#!/usr/bin/env bash
set -euo pipefail

# This single-node TP launcher does not use an external NCCL topology file.
# NCCL interprets an empty graph-file value as a path, so remove it completely.
unset NCCL_GRAPH_FILE

# Serving is GPU-bound for the qualified TP4 paths. One OpenMP thread avoids
# spin-wait contention; measurements with 2, 4, and 8 threads do not improve
# target-step or output throughput. Operators may still override the value.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

model=${MODEL:-local-inference-lab/GLM-5.3-Flash-NVFP4}
if (($# > 0)) && [[ "$1" != -* ]]; then
  model=$1
  shift
fi
model_revision=${MODEL_REVISION-}

served_model_name=${SERVED_MODEL_NAME:-GLM-5.3-Flash-NVFP4}
host=${HOST:-0.0.0.0}
port=${PORT:-8000}
tp=${TP:-4}
dcp=${DCP:-1}
cp_kv_cache_interleave_size=${CP_KV_CACHE_INTERLEAVE_SIZE:-4}
max_num_seqs=${MAX_NUM_SEQS:-32}
max_model_len=${MAX_MODEL_LEN:-1048576}
max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS:-4096}
prefill_schedule_interval=${PREFILL_SCHEDULE_INTERVAL:-8}
prefill_interval_from_cli=0
model_len_from_cli=0
chat_defaults_from_cli=0
generation_defaults_from_cli=0
for argument in "$@"; do
  case "${argument}" in
    --max-model-len | --max-model-len=*)
      model_len_from_cli=1
      ;;
    --prefill-schedule-interval | --prefill-schedule-interval=*)
      prefill_interval_from_cli=1
      ;;
    --default-chat-template-kwargs | --default-chat-template-kwargs=*)
      chat_defaults_from_cli=1
      ;;
  esac
  case "${argument//_/-}" in
    --generation-config | --generation-config=* | \
      --override-generation-config | --override-generation-config=* | \
      --override-generation-config.* | --config | --config=*)
      generation_defaults_from_cli=1
      ;;
  esac
done
max_cudagraph_capture_size=${MAX_CUDAGRAPH_CAPTURE_SIZE:-128}
gpu_memory_utilization=${GPU_MEMORY_UTILIZATION:-0.93}
load_format=${LOAD_FORMAT:-instanttensor}
speculator=${SPECULATOR:-mtp}
dflash_model=${DFLASH_MODEL:-local-inference-lab/GLM-5.3-Flash-DFlash2}
dflash_model_revision=${DFLASH_MODEL_REVISION-}
dflash_kv_cache_dtype=${DFLASH_KV_CACHE_DTYPE:-auto}
dflash_attention_backend=${DFLASH_ATTENTION_BACKEND:-FLASH_ATTN}
kv_cache_dtype=${KV_CACHE_DTYPE:-fp8}
attention_backend=${ATTENTION_BACKEND:-B12X}
moe_backend=${MOE_BACKEND:-b12x}
linear_backend=${LINEAR_BACKEND:-b12x}
mtp_attention_backend=${MTP_ATTENTION_BACKEND:-B12X}
mtp_moe_backend=${MTP_MOE_BACKEND:-marlin}
b12x_pcie_allreduce=${B12X_PCIE_ALLREDUCE:-1}
kda_decode_backend=${GLM53_KDA_DECODE_BACKEND:-auto}
kda_prefill_backend=${GLM53_KDA_PREFILL_BACKEND:-flashkda}
cudagraph_mode=${CUDAGRAPH_MODE:-FULL_AND_PIECEWISE}
dcp_ckv_gather=${DCP_CKV_GATHER:-auto}

if [[ ! "${tp}" =~ ^[1-9][0-9]*$ || ! "${dcp}" =~ ^[1-9][0-9]*$ ]]; then
  printf 'TP and DCP must be positive integers; got TP=%s DCP=%s\n' \
    "${tp}" "${dcp}" >&2
  exit 2
fi
if ((tp % dcp != 0)); then
  printf 'DCP must divide TP; got TP=%s DCP=%s\n' "${tp}" "${dcp}" >&2
  exit 2
fi
if [[ ! "${cp_kv_cache_interleave_size}" =~ ^[1-9][0-9]*$ ]]; then
  printf 'CP_KV_CACHE_INTERLEAVE_SIZE must be a positive integer; got %s\n' \
    "${cp_kv_cache_interleave_size}" >&2
  exit 2
fi
if ((prefill_interval_from_cli == 0)) &&
  [[ ! "${prefill_schedule_interval}" =~ ^[1-9][0-9]*$ ]]; then
  printf 'PREFILL_SCHEDULE_INTERVAL must be a positive integer; got %s\n' \
    "${prefill_schedule_interval}" >&2
  exit 2
fi

case "${dcp_ckv_gather}" in
  auto)
    # Independent target and recurrent pages let MTP and DFlash2 amortize
    # full-CKV gathering at the qualified 4096-token scheduler budget.
    if ((dcp > 1)); then
      dcp_ckv_gather=1
    else
      dcp_ckv_gather=0
    fi
    ;;
  0 | 1) ;;
  *)
    printf 'DCP_CKV_GATHER must be auto, 0, or 1; got %s\n' \
      "${dcp_ckv_gather}" >&2
    exit 2
    ;;
esac

# B12X DMA remains beneficial for DCP1. DCP full-CKV prefill uses PyNCCL for
# messages above the one-shot/two-shot ranges because the measured B12X DMA
# crossover is slower for that workload. An explicit operator value remains
# authoritative.
pcie_dma_min_bytes=${VLLM_PCIE_DMA_MIN_BYTES:-}
if [[ -z ${pcie_dma_min_bytes} ]]; then
  if ((dcp > 1)); then
    pcie_dma_min_bytes=off
  else
    pcie_dma_min_bytes=6MB
  fi
fi

case "${speculator}" in
  mtp)
    num_speculative_tokens=${NUM_SPECULATIVE_TOKENS:-${MTP:-0}}
    ;;
  dflash | dflash2)
    # DFlash2 is trained for an eight-token block: one verified token and
    # seven draft tokens. NUM_SPECULATIVE_TOKENS can qualify another width.
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

# A single-token target step is faster with quantized NVFP4 activations on the
# qualified RTX PRO 6000 path. Multi-token speculative verifiers retain B12X's
# profile-driven selection, which is faster for their four- and eight-row
# GEMMs. Explicit precision settings remain authoritative.
if [[ ${speculator} == mtp && ${num_speculative_tokens} == 0 &&
  -z ${VLLM_B12X_DENSE_ACTIVATION_MODE+x} &&
  -z ${VLLM_B12X_NVFP4_ACTIVATION_MODE+x} ]]; then
  export VLLM_B12X_NVFP4_ACTIVATION_MODE=quantized
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

case "${kda_prefill_backend}" in
  auto | b12x | flashkda | triton) ;;
  *)
    printf 'GLM53_KDA_PREFILL_BACKEND must be auto, b12x, flashkda, or triton; got %s\n' \
      "${kda_prefill_backend}" >&2
    exit 2
    ;;
esac

case "${dflash_attention_backend}" in
  FLASHINFER | FLASH_ATTN) ;;
  *)
    printf 'DFLASH_ATTENTION_BACKEND must be FLASHINFER or FLASH_ATTN; got %s\n' \
      "${dflash_attention_backend}" >&2
    exit 2
    ;;
esac

case "${kv_cache_dtype}" in
  fp8 | fp8_e4m3 | fp8_ds_mla | nvfp4_ds_mla) ;;
  *)
    printf 'KV_CACHE_DTYPE must be fp8, fp8_e4m3, fp8_ds_mla, or nvfp4_ds_mla; got %s\n' \
      "${kv_cache_dtype}" >&2
    exit 2
    ;;
esac

# Zero retains the target checkpoint's NVFP4 W4A4 routed-expert path.
export VLLM_B12X_MOE_FP4_FORCE_A16="${VLLM_B12X_MOE_FP4_FORCE_A16:-0}"
export VLLM_ENABLE_PCIE_ALLREDUCE="${b12x_pcie_allreduce}"
export VLLM_PCIE_ALLREDUCE_BACKEND=b12x
export VLLM_PCIE_DMA_MIN_BYTES="${pcie_dma_min_bytes}"
export VLLM_B12X_MLA_CKV_GATHER="${dcp_ckv_gather}"
# GLM-5.3 stores sparse target KV and recurrent state in separate allocations.
# The 2,048-token target page gives the qualified GPU-cache capacity without a
# measurable prefill or decode penalty. The recurrent page follows the target
# unless an external-cache launcher derives both from its transfer interval.
export VLLM_GLM53_SPLIT_TARGET_BLOCK_SIZE="${VLLM_GLM53_SPLIT_TARGET_BLOCK_SIZE:-2048}"
export VLLM_GLM53_SPLIT_MAMBA_BLOCK_SIZE="${VLLM_GLM53_SPLIT_MAMBA_BLOCK_SIZE:-auto}"
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
  --decode-context-parallel-size "${dcp}"
  # Glm5Next B12X C4 consumes cp_kv_cache_interleave_size, while generic
  # FlashInfer DCP consumes dcp_kv_cache_interleave_size. Equal values preserve
  # one rank-local cache layout for target and draft attention.
  --cp-kv-cache-interleave-size "${cp_kv_cache_interleave_size}"
  --dcp-kv-cache-interleave-size "${cp_kv_cache_interleave_size}"
  --max-num-seqs "${max_num_seqs}"
  --max-num-batched-tokens "${max_num_batched_tokens}"
  --max-cudagraph-capture-size "${max_cudagraph_capture_size}"
  --gpu-memory-utilization "${gpu_memory_utilization}"
  --mamba-cache-mode align
  --enable-chunked-prefill
  --dtype bfloat16
  --kv-cache-dtype "${kv_cache_dtype}"
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
  "{\"glm53_kda_decode_backend\":\"${kda_decode_backend}\",\"kda_prefill_backend\":\"${kda_prefill_backend}\"}"
  --compilation-config
  "{\"cudagraph_mode\":\"${cudagraph_mode}\"}"
)

if ((model_len_from_cli == 0)); then
  cmd+=(--max-model-len "${max_model_len}")
fi

if ((chat_defaults_from_cli == 0)); then
  # Preserve reasoning at its original assistant turn for agent continuations.
  # Request template kwargs remain authoritative over these server defaults.
  cmd+=(--default-chat-template-kwargs '{"reasoning_effort":"high","clear_thinking":false}')
fi

if ((generation_defaults_from_cli == 0)); then
  # The quantized checkpoint can omit Z.ai's generation defaults. Request
  # sampling remains authoritative; an explicit CLI configuration replaces
  # this launcher preset without creating duplicate JSON options.
  cmd+=(--override-generation-config '{"temperature":1.0,"top_p":0.95}')
fi

if ((prefill_interval_from_cli == 0)); then
  cmd+=(--prefill-schedule-interval "${prefill_schedule_interval}")
fi

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
      if [[ -n "${model_revision}" && "${model}" != /* ]]; then
        cmd+=(
          --speculative-config
          "{\"method\":\"mtp\",\"revision\":\"${model_revision}\",\"num_speculative_tokens\":${num_speculative_tokens},\"draft_sample_method\":\"probabilistic\",\"rejection_sample_method\":\"standard\",\"moe_backend\":\"${mtp_moe_backend}\",\"attention_backend\":\"${mtp_attention_backend}\"}"
        )
      else
        cmd+=(
          --speculative-config
          "{\"method\":\"mtp\",\"num_speculative_tokens\":${num_speculative_tokens},\"draft_sample_method\":\"probabilistic\",\"rejection_sample_method\":\"standard\",\"moe_backend\":\"${mtp_moe_backend}\",\"attention_backend\":\"${mtp_attention_backend}\"}"
        )
      fi
      ;;
    dflash | dflash2)
      if [[ -n "${dflash_model_revision}" && "${dflash_model}" != /* ]]; then
        cmd+=(
          --speculative-config
          "{\"method\":\"dflash\",\"model\":\"${dflash_model}\",\"revision\":\"${dflash_model_revision}\",\"num_speculative_tokens\":${num_speculative_tokens},\"draft_sample_method\":\"probabilistic\",\"rejection_sample_method\":\"standard\",\"kv_cache_dtype\":\"${dflash_kv_cache_dtype}\",\"attention_backend\":\"${dflash_attention_backend}\"}"
        )
      else
        cmd+=(
          --speculative-config
          "{\"method\":\"dflash\",\"model\":\"${dflash_model}\",\"num_speculative_tokens\":${num_speculative_tokens},\"draft_sample_method\":\"probabilistic\",\"rejection_sample_method\":\"standard\",\"kv_cache_dtype\":\"${dflash_kv_cache_dtype}\",\"attention_backend\":\"${dflash_attention_backend}\"}"
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
