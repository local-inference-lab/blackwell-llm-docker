#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -eq 0 ]]; then
  echo "ERROR: lmcache-mp-wrapper.sh requires a model server command" >&2
  exit 2
fi

mode="${LMCACHE_MODE:-off}"
mode="${mode,,}"
case "${mode}" in
  off|0)
    exec "$@"
    ;;
  ram|memory|1)
    mode=ram
    ;;
  disk|ram-disk|memory-disk)
    mode=disk
    ;;
  *)
    echo "ERROR: LMCACHE_MODE must be off, ram, or disk; got ${mode}" >&2
    exit 2
    ;;
esac

command -v lmcache >/dev/null || {
  echo "ERROR: LMCACHE_MODE=${mode}, but the lmcache CLI is not installed" >&2
  exit 2
}

lmcache_expected_source_root="${LMCACHE_EXPECTED_SOURCE_ROOT:-}"
if [[ -n "${lmcache_expected_source_root}" ]]; then
  "${PYTHON_BIN:-python3}" - "${lmcache_expected_source_root}" <<'PY'
from pathlib import Path
import sys

import lmcache

expected = Path(sys.argv[1]).resolve()
loaded = Path(lmcache.__file__).resolve()
if not loaded.is_relative_to(expected):
    raise SystemExit(
        f"LMCache imported from {loaded}; expected a module below {expected}"
    )
PY
fi

service_port="${PORT:-8000}"
port_offset=0
if [[ "${service_port}" =~ ^[0-9]+$ ]] && (( service_port >= 8000 )); then
  port_offset=$((service_port - 8000))
fi

lmcache_host="${LMCACHE_HOST:-127.0.0.1}"
# Derived ports assume each service uses a unique PORT offset. Deployments with
# custom spacing must set both LMCACHE_PORT and LMCACHE_HTTP_PORT explicitly.
lmcache_port="${LMCACHE_PORT:-$((5555 + port_offset))}"
lmcache_http_port="${LMCACHE_HTTP_PORT:-$((8099 + port_offset))}"
lmcache_chunk_size="${LMCACHE_CHUNK_SIZE:-}"
if [[ -z "${lmcache_chunk_size}" ]]; then
  # LMCache chunks must align to every effective DCP paged-cache block.
  # TP6 uses 192-token (DCP3) or 384-token (DCP6) manager blocks; 512 is
  # invalid for both. Power-of-two DCP layouts retain the established 512.
  case "${DCP_SIZE:-${DCP:-1}}" in
    3|6) lmcache_chunk_size=384 ;;
    *) lmcache_chunk_size=512 ;;
  esac
fi
lmcache_l1_gb="${LMCACHE_L1_GB:-24}"
lmcache_l1_init_gb="${LMCACHE_L1_INIT_GB:-${lmcache_l1_gb}}"
# Every TP rank registers an independent GPU client, including at DCP1. Give
# each client its own affinity worker so rank transfers are not serialized.
# Constrained hosts can still override this explicitly.
lmcache_gpu_workers="${LMCACHE_MAX_GPU_WORKERS:-${TP_SIZE:-${TP:-1}}}"
lmcache_cpu_workers="${LMCACHE_MAX_CPU_WORKERS:-4}"
lmcache_log="${LMCACHE_LOG:-/tmp/lmcache-mp-${service_port}.log}"
lmcache_l2_prefetch_policy="${LMCACHE_L2_PREFETCH_POLICY:-retain}"
lmcache_l2_prefetch_policy="${lmcache_l2_prefetch_policy,,}"
case "${lmcache_l2_prefetch_policy}" in
  default|retain) ;;
  *)
    echo "ERROR: LMCACHE_L2_PREFETCH_POLICY must be default or retain; got ${lmcache_l2_prefetch_policy}" >&2
    exit 2
    ;;
esac
lmcache_transfer_mode="${LMCACHE_TRANSFER_MODE:-auto}"
lmcache_transfer_mode="${lmcache_transfer_mode,,}"
case "${lmcache_transfer_mode}" in
  auto|lmcache_driven|engine_driven) ;;
  *)
    echo "ERROR: LMCACHE_TRANSFER_MODE must be auto, lmcache_driven, or engine_driven; got ${lmcache_transfer_mode}" >&2
    exit 2
    ;;
esac

lmcache_separate_object_groups="${LMCACHE_SEPARATE_OBJECT_GROUPS:-0}"
lmcache_separate_object_groups="${lmcache_separate_object_groups,,}"
case "${lmcache_separate_object_groups}" in
  1|true|yes|on) lmcache_separate_object_groups=1 ;;
  0|false|no|off) lmcache_separate_object_groups=0 ;;
  *)
    echo "ERROR: LMCACHE_SEPARATE_OBJECT_GROUPS must be a boolean; got ${lmcache_separate_object_groups}" >&2
    exit 2
    ;;
esac

server_args=(
  server
  --host "${lmcache_host}"
  --port "${lmcache_port}"
  --chunk-size "${lmcache_chunk_size}"
  --max-gpu-workers "${lmcache_gpu_workers}"
  --max-cpu-workers "${lmcache_cpu_workers}"
  --supported-transfer-mode "${lmcache_transfer_mode}"
  --l1-size-gb "${lmcache_l1_gb}"
  --l1-init-size-gb "${lmcache_l1_init_gb}"
  --l1-write-ttl-seconds 600
  --l1-read-ttl-seconds 300
  --eviction-policy LRU
  --eviction-trigger-watermark 0.90
  --eviction-ratio 0.10
  --l2-store-policy default
  --l2-prefetch-policy "${lmcache_l2_prefetch_policy}"
  --http-port "${lmcache_http_port}"
)

# An explicitly empty shared-memory name selects bounded per-request transfer
# buffers. This avoids mapping and pinning the entire L1 pool in every vLLM
# worker when the engine-driven transfer path is used.
if [[ -v LMCACHE_SHM_NAME ]]; then
  server_args+=(--shm-name "${LMCACHE_SHM_NAME}")
fi
if [[ "${lmcache_separate_object_groups}" == 1 ]]; then
  server_args+=(--separate-object-groups)
fi

lmcache_l2_path=disabled
if [[ "${mode}" == "disk" ]]; then
  lmcache_l2_path="${LMCACHE_L2_PATH:-/cache/lmcache/${service_port}}"
  lmcache_l2_gb="${LMCACHE_L2_GB:-256}"
  lmcache_l2_workers="${LMCACHE_L2_WORKERS:-4}"
  mkdir -p "${lmcache_l2_path}"
  l2_config="$(
    LMCACHE_JSON_PATH="${lmcache_l2_path}" \
    LMCACHE_JSON_WORKERS="${lmcache_l2_workers}" \
    LMCACHE_JSON_CAPACITY_GB="${lmcache_l2_gb}" \
      python3 - <<'PY'
import json
import os

print(
    json.dumps(
        {
            "type": "fs_native",
            "base_path": os.environ["LMCACHE_JSON_PATH"],
            "num_workers": int(os.environ["LMCACHE_JSON_WORKERS"]),
            "use_odirect": False,
            "max_capacity_gb": float(os.environ["LMCACHE_JSON_CAPACITY_GB"]),
        },
        separators=(",", ":"),
    )
)
PY
  )"
  server_args+=(--l2-adapter "${l2_config}")
fi

transfer_config="$(
  LMCACHE_JSON_HOST="${lmcache_host}" \
  LMCACHE_JSON_PORT="${lmcache_port}" \
  LMCACHE_JSON_TRANSFER_MODE="${lmcache_transfer_mode}" \
    python3 - <<'PY'
import json
import os

print(
    json.dumps(
        {
            "kv_connector": "LMCacheMPConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "lmcache.mp.host": f"tcp://{os.environ['LMCACHE_JSON_HOST']}",
                "lmcache.mp.port": int(os.environ["LMCACHE_JSON_PORT"]),
                "lmcache.mp.mq_timeout": 60,
                "lmcache.mp.heartbeat_interval": 5,
                "lmcache.mp.mp_transfer_mode": os.environ[
                    "LMCACHE_JSON_TRANSFER_MODE"
                ],
            },
        },
        separators=(",", ":"),
    )
)
PY
)"

# LMCache registers KV storage by address. PyTorch expandable segments can
# remap those virtual addresses, so vLLM intentionally rejects this pairing.
# Keep any unrelated allocator settings while forcing the incompatible option
# off before the downstream model helper applies its normal default.
allocator_config="${PYTORCH_CUDA_ALLOC_CONF:-}"
if [[ -z "${allocator_config}" ]]; then
  allocator_config="expandable_segments:False"
elif [[ "${allocator_config}" =~ (^|,)expandable_segments:True(,|$) ]]; then
  allocator_config="${allocator_config//expandable_segments:True/expandable_segments:False}"
  echo "LMCache requires PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False; overriding expandable_segments:True"
fi
export PYTORCH_CUDA_ALLOC_CONF="${allocator_config}"

rm -f "${lmcache_log}"
export LMCACHE_DISABLE_BANNER="${LMCACHE_DISABLE_BANNER:-1}"
lmcache_server_command=(lmcache)
if [[ "${lmcache_transfer_mode}" == engine_driven ]]; then
  # GPU visibility is removed only from the standalone cache server. GPU
  # gather/scatter operations remain in the existing vLLM worker processes.
  lmcache_server_env="${LMCACHE_SERVER_ENV-CUDA_VISIBLE_DEVICES= CUDA_MODULE_LOADING=LAZY}"
  read -r -a lmcache_server_env_args <<< "${lmcache_server_env}"
  for assignment in "${lmcache_server_env_args[@]}"; do
    if [[ ! "${assignment}" =~ ^[A-Za-z_][A-Za-z0-9_]*=.*$ ]]; then
      echo "ERROR: invalid LMCACHE_SERVER_ENV assignment: ${assignment}" >&2
      exit 2
    fi
  done
  lmcache_server_command=(env "${lmcache_server_env_args[@]}" lmcache)
fi
"${lmcache_server_command[@]}" "${server_args[@]}" >"${lmcache_log}" 2>&1 &
lmcache_pid=$!
model_pid=""
shutdown_requested=0

stop_children() {
  if [[ -n "${model_pid}" ]] && kill -0 "${model_pid}" 2>/dev/null; then
    kill -TERM "${model_pid}" 2>/dev/null || true
  fi
  if kill -0 "${lmcache_pid}" 2>/dev/null; then
    kill -TERM "${lmcache_pid}" 2>/dev/null || true
  fi
}

request_shutdown() {
  shutdown_requested=1
  stop_children
}
trap request_shutdown INT TERM HUP

ready=0
for _ in $(seq 1 "${LMCACHE_START_TIMEOUT:-120}"); do
  if ! kill -0 "${lmcache_pid}" 2>/dev/null; then
    break
  fi
  if curl --fail --silent --show-error --max-time 1 \
      "http://127.0.0.1:${lmcache_http_port}/healthcheck" \
      >/dev/null 2>&1; then
    ready=1
    break
  fi
  if grep -Fq "${LMCACHE_READY_LOG_TEXT:-LMCache ZMQ cache server is running}" \
      "${lmcache_log}"; then
    ready=1
    break
  fi
  sleep 1
done
if [[ "${ready}" != 1 ]]; then
  echo "ERROR: LMCache did not become ready; log follows" >&2
  sed -n '1,320p' "${lmcache_log}" >&2
  stop_children
  wait "${lmcache_pid}" 2>/dev/null || true
  exit 1
fi

printf 'LMCache ready: mode=%s transfer=%s L1=%sGB chunk=%s L2=%s health=http://%s:%s/healthcheck metrics=http://%s:%s/metrics log=%s\n' \
  "${mode}" "${lmcache_transfer_mode}" "${lmcache_l1_gb}" "${lmcache_chunk_size}" \
  "${lmcache_l2_path}" "${lmcache_host}" "${lmcache_http_port}" \
  "${lmcache_host}" "${lmcache_http_port}" "${lmcache_log}"

"$@" --kv-transfer-config "${transfer_config}" &
model_pid=$!

set +e
completed_pid=""
wait -n -p completed_pid "${lmcache_pid}" "${model_pid}"
first_status=$?
set -e

if [[ "${shutdown_requested}" == 1 ]]; then
  set +e
  if [[ "${completed_pid:-}" == "${model_pid}" ]]; then
    model_status=${first_status}
  else
    wait "${model_pid}" 2>/dev/null
    model_status=$?
  fi
  if [[ "${completed_pid:-}" != "${lmcache_pid}" ]]; then
    wait "${lmcache_pid}" 2>/dev/null
  fi
  set -e
  exit "${model_status}"
fi

if [[ "${completed_pid:-}" == "${lmcache_pid}" ]]; then
  echo "ERROR: LMCache exited while the model server was running" >&2
  sed -n '1,320p' "${lmcache_log}" >&2
  stop_children
  wait "${model_pid}" 2>/dev/null || true
  exit 1
fi

stop_children
wait "${lmcache_pid}" 2>/dev/null || true
exit "${first_status}"
