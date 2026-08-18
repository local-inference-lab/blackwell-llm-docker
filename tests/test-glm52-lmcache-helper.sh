#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
lmcache_wrapper="${repo_root}/launchers/lmcache-mp-wrapper.sh"
tmp_root="$(mktemp -d)"
trap 'rm -rf "${tmp_root}"' EXIT
mkdir -p "${tmp_root}/bin"

cat >"${tmp_root}/bin/lmcache" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
printf '%q ' "$@" >"${LMCACHE_TEST_SERVER_ARGS}"
printf '\n' >>"${LMCACHE_TEST_SERVER_ARGS}"
if [[ -n "${LMCACHE_TEST_SERVER_ENV:-}" ]]; then
  printf '%s\n%s\n' \
    "${CUDA_VISIBLE_DEVICES-<unset>}" \
    "${CUDA_MODULE_LOADING-<unset>}" \
    >"${LMCACHE_TEST_SERVER_ENV}"
fi
echo 'LMCache ZMQ cache server is running'
if [[ "${LMCACHE_TEST_EXIT_AFTER_READY:-0}" == 1 ]]; then
  sleep "${LMCACHE_TEST_EXIT_DELAY:-2}"
  exit 23
fi
trap 'exit 0' INT TERM
while true; do
  sleep 1 &
  wait $! || true
done
SH
chmod +x "${tmp_root}/bin/lmcache"

cat >"${tmp_root}/bin/curl" <<'SH'
#!/usr/bin/env bash
if [[ "${LMCACHE_TEST_HTTP_READY:-0}" == 1 ]]; then
  exit 0
fi
exit 22
SH
chmod +x "${tmp_root}/bin/curl"

cat >"${tmp_root}/model-server" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$@" >"${LMCACHE_TEST_MODEL_ARGS}"
if [[ -n "${LMCACHE_TEST_MODEL_ENV:-}" ]]; then
  printf '%s\n' "${PYTORCH_CUDA_ALLOC_CONF-<unset>}" >"${LMCACHE_TEST_MODEL_ENV}"
fi
if [[ -n "${LMCACHE_TEST_MODEL_CUDA_ENV:-}" ]]; then
  printf '%s\n' "${CUDA_VISIBLE_DEVICES-<unset>}" \
    >"${LMCACHE_TEST_MODEL_CUDA_ENV}"
fi
if [[ "${LMCACHE_TEST_MODEL_HANDLE_TERM:-0}" == 1 ]]; then
  trap 'exit 0' INT TERM HUP
  sleep "${LMCACHE_TEST_MODEL_SLEEP:-0}" &
  wait $! || true
else
  sleep "${LMCACHE_TEST_MODEL_SLEEP:-0}"
fi
SH
chmod +x "${tmp_root}/model-server"

PATH="${tmp_root}/bin:${PATH}" \
LMCACHE_MODE=ram \
PORT=8002 \
TP_SIZE=8 \
DCP_SIZE=4 \
LMCACHE_L1_GB=2 \
LMCACHE_LOG="${tmp_root}/ram.log" \
LMCACHE_TEST_SERVER_ARGS="${tmp_root}/ram-server.args" \
LMCACHE_TEST_MODEL_ARGS="${tmp_root}/ram-model.args" \
LMCACHE_TEST_MODEL_ENV="${tmp_root}/ram-model.env" \
bash "${lmcache_wrapper}" \
  "${tmp_root}/model-server" --model-arg value

grep -Fq -- '--port 5557' "${tmp_root}/ram-server.args"
grep -Fq -- '--http-port 8101' "${tmp_root}/ram-server.args"
if grep -Fq -- '--prometheus-port' "${tmp_root}/ram-server.args"; then
  echo 'LMCache helper unexpectedly configured a standalone metrics port' >&2
  exit 1
fi
grep -Fq -- '--l1-size-gb 2' "${tmp_root}/ram-server.args"
grep -Fq -- '--max-gpu-workers 8' "${tmp_root}/ram-server.args"
grep -Fq -- '--chunk-size 512' "${tmp_root}/ram-server.args"
grep -Fq -- '--supported-transfer-mode auto' "${tmp_root}/ram-server.args"
if grep -Fq -- '--shm-name' "${tmp_root}/ram-server.args"; then
  echo 'Default LMCache mode unexpectedly configured shared memory' >&2
  exit 1
fi
if grep -Fq -- '--separate-object-groups' "${tmp_root}/ram-server.args"; then
  echo 'Default LMCache mode unexpectedly separated object groups' >&2
  exit 1
fi
if grep -Fq -- '--l2-adapter' "${tmp_root}/ram-server.args"; then
  echo 'RAM-only mode unexpectedly enabled L2' >&2
  exit 1
fi
grep -Fq -- '--kv-transfer-config' "${tmp_root}/ram-model.args"
grep -Fq -- '"lmcache.mp.port":5557' "${tmp_root}/ram-model.args"
grep -Fq -- '"lmcache.mp.mp_transfer_mode":"auto"' \
  "${tmp_root}/ram-model.args"
grep -Fxq 'expandable_segments:False' "${tmp_root}/ram-model.env"

# Engine-driven transfers keep the standalone server CPU-only. GPU visibility
# remains unchanged for the model process that performs gather/scatter work.
PATH="${tmp_root}/bin:${PATH}" \
LMCACHE_MODE=ram \
LMCACHE_TRANSFER_MODE=engine_driven \
LMCACHE_SHM_NAME= \
LMCACHE_SEPARATE_OBJECT_GROUPS=1 \
LMCACHE_L1_GB=2 \
LMCACHE_LOG="${tmp_root}/engine.log" \
LMCACHE_TEST_SERVER_ARGS="${tmp_root}/engine-server.args" \
LMCACHE_TEST_SERVER_ENV="${tmp_root}/engine-server.env" \
LMCACHE_TEST_MODEL_ARGS="${tmp_root}/engine-model.args" \
LMCACHE_TEST_MODEL_CUDA_ENV="${tmp_root}/engine-model-cuda.env" \
CUDA_VISIBLE_DEVICES=0,1 \
bash "${lmcache_wrapper}" \
  "${tmp_root}/model-server" engine-driven
grep -Fq -- '--supported-transfer-mode engine_driven' \
  "${tmp_root}/engine-server.args"
grep -Fq -- "--shm-name ''" "${tmp_root}/engine-server.args"
grep -Fq -- '--separate-object-groups' "${tmp_root}/engine-server.args"
grep -Fq -- '"lmcache.mp.mp_transfer_mode":"engine_driven"' \
  "${tmp_root}/engine-model.args"
sed -n '1p' "${tmp_root}/engine-server.env" | grep -Fxq ''
sed -n '2p' "${tmp_root}/engine-server.env" | grep -Fxq 'LAZY'
grep -Fxq '0,1' "${tmp_root}/engine-model-cuda.env"

# HTTP health is the primary readiness contract; this test intentionally uses
# a log string that cannot satisfy the fallback.
PATH="${tmp_root}/bin:${PATH}" \
LMCACHE_MODE=ram \
PORT=8008 \
LMCACHE_L1_GB=2 \
LMCACHE_READY_LOG_TEXT='not-present' \
LMCACHE_TEST_HTTP_READY=1 \
LMCACHE_LOG="${tmp_root}/http-ready.log" \
LMCACHE_TEST_SERVER_ARGS="${tmp_root}/http-ready-server.args" \
LMCACHE_TEST_MODEL_ARGS="${tmp_root}/http-ready-model.args" \
bash "${lmcache_wrapper}" \
  "${tmp_root}/model-server" http-ready
grep -Fxq 'http-ready' "${tmp_root}/http-ready-model.args"

PATH="${tmp_root}/bin:${PATH}" \
LMCACHE_MODE=ram \
PORT=8006 \
TP=6 \
DCP=3 \
LMCACHE_L1_GB=2 \
LMCACHE_LOG="${tmp_root}/tp6.log" \
LMCACHE_TEST_SERVER_ARGS="${tmp_root}/tp6-server.args" \
LMCACHE_TEST_MODEL_ARGS="${tmp_root}/tp6-model.args" \
bash "${lmcache_wrapper}" \
  "${tmp_root}/model-server"
grep -Fq -- '--max-gpu-workers 6' "${tmp_root}/tp6-server.args"
grep -Fq -- '--chunk-size 384' "${tmp_root}/tp6-server.args"

PATH="${tmp_root}/bin:${PATH}" \
LMCACHE_MODE=ram \
PORT=8007 \
TP=6 \
DCP=6 \
LMCACHE_CHUNK_SIZE=768 \
LMCACHE_MAX_GPU_WORKERS=2 \
LMCACHE_L1_GB=2 \
LMCACHE_LOG="${tmp_root}/override.log" \
LMCACHE_TEST_SERVER_ARGS="${tmp_root}/override-server.args" \
LMCACHE_TEST_MODEL_ARGS="${tmp_root}/override-model.args" \
bash "${lmcache_wrapper}" \
  "${tmp_root}/model-server"
grep -Fq -- '--max-gpu-workers 2' "${tmp_root}/override-server.args"
grep -Fq -- '--chunk-size 768' "${tmp_root}/override-server.args"

# The prefetch policy is independently configurable. Retain remains the
# generic wrapper default; profiles with sustained L1 pressure can select the
# temporary-staging policy explicitly.
PATH="${tmp_root}/bin:${PATH}" \
LMCACHE_MODE=ram \
LMCACHE_L2_PREFETCH_POLICY=default \
LMCACHE_L1_GB=2 \
LMCACHE_LOG="${tmp_root}/prefetch.log" \
LMCACHE_TEST_SERVER_ARGS="${tmp_root}/prefetch-server.args" \
LMCACHE_TEST_MODEL_ARGS="${tmp_root}/prefetch-model.args" \
bash "${lmcache_wrapper}" \
  "${tmp_root}/model-server"
grep -Fq -- '--l2-prefetch-policy default' \
  "${tmp_root}/prefetch-server.args"

if PATH="${tmp_root}/bin:${PATH}" \
  LMCACHE_MODE=ram \
  LMCACHE_L2_PREFETCH_POLICY=invalid \
  bash "${lmcache_wrapper}" \
    "${tmp_root}/model-server"; then
  echo 'Invalid LMCache L2 prefetch policy unexpectedly succeeded' >&2
  exit 1
fi

PATH="${tmp_root}/bin:${PATH}" \
LMCACHE_MODE=ram \
PORT=8005 \
LMCACHE_L1_GB=2 \
LMCACHE_LOG="${tmp_root}/allocator.log" \
LMCACHE_TEST_SERVER_ARGS="${tmp_root}/allocator-server.args" \
LMCACHE_TEST_MODEL_ARGS="${tmp_root}/allocator-model.args" \
LMCACHE_TEST_MODEL_ENV="${tmp_root}/allocator-model.env" \
PYTORCH_CUDA_ALLOC_CONF='max_split_size_mb:256,expandable_segments:True' \
bash "${lmcache_wrapper}" \
  "${tmp_root}/model-server" >"${tmp_root}/allocator-wrapper.log"
grep -Fxq 'max_split_size_mb:256,expandable_segments:False' \
  "${tmp_root}/allocator-model.env"
grep -Fxq \
  'LMCache requires PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False; overriding expandable_segments:True' \
  "${tmp_root}/allocator-wrapper.log"

PATH="${tmp_root}/bin:${PATH}" \
LMCACHE_MODE=disk \
PORT=8003 \
LMCACHE_L1_GB=2 \
LMCACHE_L2_GB=7 \
LMCACHE_L2_PATH="${tmp_root}/l2" \
LMCACHE_HTTP_PORT=8181 \
LMCACHE_LOG="${tmp_root}/disk.log" \
LMCACHE_TEST_SERVER_ARGS="${tmp_root}/disk-server.args" \
LMCACHE_TEST_MODEL_ARGS="${tmp_root}/disk-model.args" \
bash "${lmcache_wrapper}" \
  "${tmp_root}/model-server"

grep -Fq -- '--l2-adapter' "${tmp_root}/disk-server.args"
grep -Fq -- '--http-port 8181' "${tmp_root}/disk-server.args"
if grep -Fq -- '--prometheus-port' "${tmp_root}/disk-server.args"; then
  echo 'LMCache disk mode unexpectedly configured a standalone metrics port' >&2
  exit 1
fi
grep -Fq -- 'fs_native' "${tmp_root}/disk-server.args"
grep -Fq -- 'use_odirect' "${tmp_root}/disk-server.args"
grep -Fq -- 'max_capacity_gb' "${tmp_root}/disk-server.args"
grep -Fq -- '--l2-prefetch-policy retain' "${tmp_root}/disk-server.args"

if PATH="${tmp_root}/bin:${PATH}" \
  LMCACHE_MODE=invalid \
  bash "${lmcache_wrapper}" \
    "${tmp_root}/model-server"; then
  echo 'Invalid LMCache mode unexpectedly succeeded' >&2
  exit 1
fi

LMCACHE_MODE=off \
LMCACHE_TEST_MODEL_ARGS="${tmp_root}/off-model.args" \
LMCACHE_TEST_MODEL_ENV="${tmp_root}/off-model.env" \
bash "${repo_root}/launchers/glm52-lmcache-wrapper.sh" \
  "${tmp_root}/model-server" untouched
grep -Fxq 'untouched' "${tmp_root}/off-model.args"
grep -Fxq '<unset>' "${tmp_root}/off-model.env"

# The EXL3 preset uses the same GLM-5.2 MLA KV-transfer contract. Exercise the
# unified entrypoint so this allowlist cannot drift from the model preset above.
for model_family in glm52-exl3 exl3; do
  PATH="${tmp_root}/bin:${PATH}" \
  LMCACHE_MODE=ram \
  MODEL_FAMILY="${model_family}" \
  GLM52_LMCACHE_WRAPPER="${repo_root}/launchers/glm52-lmcache-wrapper.sh" \
  GLM52_SERVER="${tmp_root}/model-server" \
  DRY_RUN=1 \
  PORT=8010 \
  TP=4 \
  DCP=2 \
  LMCACHE_L1_GB=2 \
  LMCACHE_LOG="${tmp_root}/${model_family}.log" \
  LMCACHE_TEST_SERVER_ARGS="${tmp_root}/${model_family}-server.args" \
  LMCACHE_TEST_MODEL_ARGS="${tmp_root}/${model_family}-model.args" \
  bash "${repo_root}/launchers/serve-gilded-gnosis.sh"

  grep -Fq -- '--kv-transfer-config' \
    "${tmp_root}/${model_family}-model.args"
  grep -Fq 'LMCacheMPConnector' \
    "${tmp_root}/${model_family}-model.args"
done

if LMCACHE_MODE=ram \
  MODEL_FAMILY=ds4 \
  bash "${repo_root}/launchers/serve-gilded-gnosis.sh"; then
  echo 'LMCache unexpectedly accepted an unvalidated model family' >&2
  exit 1
fi

crash_stderr="${tmp_root}/crash.stderr"
if PATH="${tmp_root}/bin:${PATH}" \
  LMCACHE_MODE=ram \
  PORT=8004 \
  LMCACHE_L1_GB=2 \
  LMCACHE_LOG="${tmp_root}/crash.log" \
  LMCACHE_TEST_EXIT_AFTER_READY=1 \
  LMCACHE_TEST_MODEL_SLEEP=5 \
  LMCACHE_TEST_SERVER_ARGS="${tmp_root}/crash-server.args" \
  LMCACHE_TEST_MODEL_ARGS="${tmp_root}/crash-model.args" \
  bash "${lmcache_wrapper}" \
    "${tmp_root}/model-server" 2>"${crash_stderr}"; then
  echo 'Wrapper unexpectedly survived an LMCache server failure' >&2
  exit 1
fi
grep -Fq 'ERROR: LMCache exited while the model server was running' \
  "${crash_stderr}"

shutdown_stdout="${tmp_root}/shutdown.stdout"
shutdown_stderr="${tmp_root}/shutdown.stderr"
PATH="${tmp_root}/bin:${PATH}" \
LMCACHE_MODE=ram \
PORT=8011 \
LMCACHE_L1_GB=2 \
LMCACHE_LOG="${tmp_root}/shutdown.log" \
LMCACHE_TEST_MODEL_SLEEP=30 \
LMCACHE_TEST_MODEL_HANDLE_TERM=1 \
LMCACHE_TEST_SERVER_ARGS="${tmp_root}/shutdown-server.args" \
LMCACHE_TEST_MODEL_ARGS="${tmp_root}/shutdown-model.args" \
bash "${lmcache_wrapper}" \
  "${tmp_root}/model-server" >"${shutdown_stdout}" 2>"${shutdown_stderr}" &
shutdown_pid=$!
for _ in $(seq 1 100); do
  if [[ -s "${tmp_root}/shutdown-model.args" ]]; then
    break
  fi
  sleep 0.05
done
if [[ ! -s "${tmp_root}/shutdown-model.args" ]]; then
  echo 'LMCache wrapper shutdown test did not start the model server' >&2
  kill -TERM "${shutdown_pid}" 2>/dev/null || true
  wait "${shutdown_pid}" 2>/dev/null || true
  exit 1
fi
kill -TERM "${shutdown_pid}"
wait "${shutdown_pid}"
if grep -Fq 'unbound variable' "${shutdown_stderr}"; then
  echo 'LMCache wrapper accessed an unset wait result during shutdown' >&2
  exit 1
fi
if grep -Fq 'LMCache exited while the model server was running' \
    "${shutdown_stderr}"; then
  echo 'LMCache wrapper reported an expected shutdown as a server failure' >&2
  exit 1
fi

echo 'LMCache multiprocessing helper: PASS'
