#!/usr/bin/env bash
set -euo pipefail

# NCCL interprets an empty graph-file value as a path and fails communicator
# initialization. The runtime does not require an external NCCL topology file.
unset NCCL_GRAPH_FILE

model=${MODEL:-local-inference-lab/GLM-5.3-Flash-NVFP4}
if (($# > 0)) && [[ "$1" != -* ]]; then
  model=$1
  shift
fi
model_revision=${MODEL_REVISION-520de24eabf507659eaef7c70f14fd584527facc}

served_model_name=${SERVED_MODEL_NAME:-GLM-5.3-Flash-NVFP4}
host=${HOST:-0.0.0.0}
port=${PORT:-8000}
tp=${TP:-4}
max_num_seqs=${MAX_NUM_SEQS:-16}
max_model_len=${MAX_MODEL_LEN:-262144}
max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS:-4096}
gpu_memory_utilization=${GPU_MEMORY_UTILIZATION:-0.90}
load_format=${LOAD_FORMAT:-instanttensor}
speculator=${SPECULATOR:-mtp}
dflash_model=${DFLASH_MODEL:-local-inference-lab/GLM-5.3-Flash-DFlash2-MXFP8}
dflash_model_revision=${DFLASH_MODEL_REVISION-b6d33aa93fc1ac5b23a88251a1c0ce0bfe2ad17c}
attention_backend=${ATTENTION_BACKEND:-B12X}
moe_backend=${MOE_BACKEND:-b12x}
linear_backend=${LINEAR_BACKEND:-b12x}
mtp_attention_backend=${MTP_ATTENTION_BACKEND:-B12X}
mtp_moe_backend=${MTP_MOE_BACKEND:-humming}
b12x_pcie_allreduce=${B12X_PCIE_ALLREDUCE:-1}
kda_decode_backend=${GLM53_KDA_DECODE_BACKEND:-auto}
cudagraph_mode=${CUDAGRAPH_MODE:-FULL}

case "${speculator}" in
  mtp)
    num_speculative_tokens=${NUM_SPECULATIVE_TOKENS:-${MTP:-0}}
    ;;
  dflash | dflash2)
    # DFlash2 verifies one target token and drafts the remaining seven tokens
    # in its serialized eight-token block.
    num_speculative_tokens=${NUM_SPECULATIVE_TOKENS:-7}
    ;;
  *)
    printf 'SPECULATOR must be mtp, dflash, or dflash2; got %s\n' \
      "${speculator}" >&2
    exit 2
    ;;
esac

if [[ ! "${num_speculative_tokens}" =~ ^[0-9]+$ ]]; then
  printf 'NUM_SPECULATIVE_TOKENS/MTP must be a non-negative integer; got %s\n' \
    "${num_speculative_tokens}" >&2
  exit 2
fi

case "${b12x_pcie_allreduce}" in
  0 | 1) ;;
  *)
    printf 'B12X_PCIE_ALLREDUCE must be 0 or 1; got %s\n' \
      "${b12x_pcie_allreduce}" >&2
    exit 2
    ;;
esac

case "${kda_decode_backend}" in
  auto | b12x | triton) ;;
  *)
    printf 'GLM53_KDA_DECODE_BACKEND must be auto, b12x, or triton; got %s\n' \
      "${kda_decode_backend}" >&2
    exit 2
    ;;
esac

# Zero retains the target checkpoint's NVFP4 W4A4 routed-expert path.
export VLLM_B12X_MOE_FP4_FORCE_A16="${VLLM_B12X_MOE_FP4_FORCE_A16:-0}"
export VLLM_ENABLE_PCIE_ALLREDUCE="${b12x_pcie_allreduce}"
export VLLM_PCIE_ALLREDUCE_BACKEND=b12x
export VLLM_PLUGINS=

revision_args=()
if [[ -n "${model_revision}" && "${model}" != /* ]]; then
  revision_args=(--revision "${model_revision}")
fi

cmd=(
  /opt/venv/bin/vllm serve "${model}"
  "${revision_args[@]}"
  --served-model-name "${served_model_name}"
  --host "${host}"
  --port "${port}"
  --tensor-parallel-size "${tp}"
  --pipeline-parallel-size 1
  --decode-context-parallel-size 1
  --max-num-seqs "${max_num_seqs}"
  --max-model-len "${max_model_len}"
  --max-num-batched-tokens "${max_num_batched_tokens}"
  --gpu-memory-utilization "${gpu_memory_utilization}"
  --mamba-cache-mode align
  --enable-chunked-prefill
  --dtype bfloat16
  --kv-cache-dtype fp8
  --quantization modelopt_mixed
  --block-size 256
  --load-format "${load_format}"
  --attention-backend "${attention_backend}"
  --moe-backend "${moe_backend}"
  --linear-backend "${linear_backend}"
  --no-enable-flashinfer-autotune
  --enable-auto-tool-choice
  --tool-call-parser glm47
  --reasoning-parser glm45
  --additional-config
  "{\"glm53_kda_decode_backend\":\"${kda_decode_backend}\"}"
  --compilation-config
  "{\"cudagraph_mode\":\"${cudagraph_mode}\"}"
)

if ((b12x_pcie_allreduce == 0)); then
  cmd+=(--disable-custom-all-reduce)
fi

case "${ENABLE_PREFIX_CACHING:-1}" in
  0) ;;
  1) cmd+=(--enable-prefix-caching) ;;
  *)
    printf 'ENABLE_PREFIX_CACHING must be 0 or 1; got %s\n' \
      "${ENABLE_PREFIX_CACHING}" >&2
    exit 2
    ;;
esac

if ((num_speculative_tokens > 0)); then
  case "${speculator}" in
    mtp)
      cmd+=(
        --speculative-config
        "{\"method\":\"mtp\",\"num_speculative_tokens\":${num_speculative_tokens},\"moe_backend\":\"${mtp_moe_backend}\",\"attention_backend\":\"${mtp_attention_backend}\"}"
      )
      ;;
    dflash | dflash2)
      if [[ -n "${dflash_model_revision}" && "${dflash_model}" != /* ]]; then
        cmd+=(
          --speculative-config
          "{\"method\":\"dflash\",\"model\":\"${dflash_model}\",\"revision\":\"${dflash_model_revision}\",\"num_speculative_tokens\":${num_speculative_tokens},\"kv_cache_dtype\":\"auto\"}"
        )
      else
        cmd+=(
          --speculative-config
          "{\"method\":\"dflash\",\"model\":\"${dflash_model}\",\"num_speculative_tokens\":${num_speculative_tokens},\"kv_cache_dtype\":\"auto\"}"
        )
      fi
      ;;
  esac
fi

cmd+=("$@")

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf 'GLM-5.3-Flash NVFP4 launch:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  exit 0
fi

exec "${cmd[@]}"
