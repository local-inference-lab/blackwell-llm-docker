#!/usr/bin/env bash
# Serve Kimi-K3 MXFP4 or serialized QSRT-K2 through installed packages.
set -euo pipefail
unset NCCL_GRAPH_FILE PYTHONPATH

readonly mode=${KIMI_SPECULATOR:-none}
readonly sequences=${KIMI_MAX_SEQS:-1}
readonly format=${KIMI_QUANT_FORMAT:-mxfp4}
quant_args=()
case "$format" in
  mxfp4)
    checkpoint=${KIMI_CHECKPOINT:-moonshotai/Kimi-K3}
    revision=2496450e92e425c886db095102a52a6682ca3970
    served_model=${KIMI_SERVED_MODEL:-Kimi-K3-MXFP4}
    tp=${KIMI_TP:-16}
    quant_args=(--quantization-config '{"linear":"mxfp8","ignore":["re:^(?!.*(?:self_attn\\.(?:q_proj|k_proj|v_proj|b_proj|f_a_proj|in_proj_qkvgfab)|vision_tower\\..*|mm_projector\\..*)$).*$"]}')
    ;;
  qsrt_k2)
    checkpoint=${KIMI_CHECKPOINT:?QSRT-K2 requires a local checkpoint directory}
    revision=3b98114115f1d41ce7963ba346c3fca19918b0bd
    served_model=${KIMI_SERVED_MODEL:-Kimi-K3-QSRT-K2}
    tp=${KIMI_TP:-10}
    # QSRT stores dense MXFP8 weights and BF16 gates; online conversion changes them.
    quant_args=(--quantization qsrt_k2)
    : "${KIMI_KV_BYTES:?Set the QSRT-K2 per-rank KV allocation explicitly}"
    ;;
  *) echo 'KIMI_QUANT_FORMAT must be mxfp4 or qsrt_k2' >&2; exit 2 ;;
esac
readonly dcp=${KIMI_DCP:-$tp}
if ! [[ "$sequences" =~ ^[1-9][0-9]*$ ]]; then
  echo 'KIMI_MAX_SEQS must be a positive integer' >&2
  exit 2
fi
spec_args=()
draft_graph_width=0
case "$mode" in
  none)
    width=1
    kv_bytes=${KIMI_KV_BYTES:-910000000}
    ;;
  dspark)
    width=8
    kv_bytes=${KIMI_KV_BYTES:-1270000000}
    export VLLM_K3_KV_GROUP_SIZE=${VLLM_K3_KV_GROUP_SIZE:-6}
    export VLLM_DSPARK_DRAFT_KV_WINDOW=${VLLM_DSPARK_DRAFT_KV_WINDOW:-32768}
    export VLLM_DSPARK_COMPACT_ROPE=1
    export VLLM_DSPARK_SHARD_MARKOV_HEAD=1
    export VLLM_DSPARK_REPLICATE_MARKOV_W1=1
    export VLLM_KIMI_K3_B12X_DSPARK_ARGMAX=1
    spec_args=(--speculative-config '{"method":"dspark","model":"Inferact/Kimi-K3-DSpark","revision":"cf6b8244620e7ea4b0651d214f28e89eac75bed6","num_speculative_tokens":7,"attention_backend":"B12X_MLA","kv_cache_dtype":"fp8","draft_sample_method":"greedy","rejection_sample_method":"block","quantization":"mxfp8","quantization_config":{"linear":"mxfp8","ignore":["re:.*fused_qkv_a_proj$"]}}')
    ;;
  dspark-redhat)
    width=6
    draft_graph_width=5
    kv_bytes=${KIMI_KV_BYTES:-0}
    export VLLM_K3_KV_GROUP_SIZE=${VLLM_K3_KV_GROUP_SIZE:-3}
    export VLLM_DSPARK_DRAFT_KV_WINDOW=${VLLM_DSPARK_DRAFT_KV_WINDOW:-32768}
    export VLLM_DSPARK_COMPACT_ROPE=1
    export VLLM_DSPARK_SHARD_MARKOV_HEAD=1
    export VLLM_DSPARK_REPLICATE_MARKOV_W1=1
    export VLLM_KIMI_K3_B12X_DSPARK_ARGMAX=1
    export VLLM_KV_CACHE_LAYOUT=BLHNC
    export VLLM_DFLASH_SHARD_AUX_PROJECTION=1
    export VLLM_DFLASH_AUX_BF16_STAGING=1
    export VLLM_DFLASH_AUX_MXFP8_STREAMING=0
    export VLLM_DFLASH_COMPACT_ROPE=1
    draft=${KIMI_REDHAT_DSPARK_CHECKPOINT:-RedHatAI/Kimi-K3-speculator.dspark}
    spec_args=(--speculative-config "{\"method\":\"dspark\",\"model\":\"$draft\",\"revision\":\"38a88101e0d46bb22134b9da340f381b954d40d4\",\"num_speculative_tokens\":5,\"attention_backend\":\"TRITON_ATTN\",\"kv_cache_dtype\":\"auto\",\"draft_sample_method\":\"greedy\",\"rejection_sample_method\":\"block\",\"draft_load_config\":{\"load_format\":\"auto\"},\"quantization\":\"mxfp8\",\"quantization_config\":{\"linear\":\"mxfp8\",\"ignore\":[\"re:.*qkv_proj$\",\"re:.*markov_head.*\"]}}")
    ;;
  dflash)
    width=8
    kv_bytes=${KIMI_KV_BYTES:-1180000000}
    export VLLM_K3_KV_GROUP_SIZE=${VLLM_K3_KV_GROUP_SIZE:-6}
    export VLLM_DFLASH_AUX_MXFP8_STREAMING=${VLLM_DFLASH_AUX_MXFP8_STREAMING:-0}
    export VLLM_DFLASH_AUX_BF16_STAGING=${VLLM_DFLASH_AUX_BF16_STAGING:-1}
    export VLLM_DFLASH_COMPACT_ROPE=1
    export VLLM_DFLASH_SHARD_AUX_PROJECTION=1
    # Replicated draft and sharded target pages require block-major pool views.
    export VLLM_KV_CACHE_LAYOUT=BLHNC
    spec_args=(--speculative-config '{"method":"dflash","model":"modal-labs/Kimi-K3-DFlash","revision":"c192d15a43407bf758b5ae0880d5c72052fef1de","num_speculative_tokens":7,"attention_backend":"TRITON_ATTN","draft_load_config":{"load_format":"auto"},"quantization":"mxfp8","quantization_config":{"linear":"mxfp8","ignore":["re:.*qkv_proj$"]}}')
    ;;
  dflash2)
    proposals=${KIMI_DFLASH2_SPEC_TOKENS:-7}
    if ! [[ "$proposals" =~ ^[1-7]$ ]]; then
      echo 'KIMI_DFLASH2_SPEC_TOKENS must be an integer from 1 through 7' >&2
      exit 2
    fi
    width=$((proposals+1))
    kv_bytes=${KIMI_KV_BYTES:?MLA DFlash2 requires an explicit per-rank KV allocation}
    export VLLM_K3_KV_GROUP_SIZE=${VLLM_K3_KV_GROUP_SIZE:-3}
    export VLLM_DSPARK_COMPACT_ROPE=1
    draft_quant_fields=''
    case "${KIMI_DFLASH2_QUANTIZATION:-bf16}" in
      bf16) ;;
      mxfp8)
        # The streamed context-KV projection consumes BF16 weight slices.
        draft_quant_fields=',"quantization":"mxfp8","quantization_config":{"linear":"mxfp8","ignore":["re:.*fused_qkv_a_proj$"]}'
        ;;
      *) echo 'KIMI_DFLASH2_QUANTIZATION must be bf16 or mxfp8' >&2; exit 2 ;;
    esac
    # Preserve the checkpoint's four sliding-attention and one full-attention layers.
    spec_args=(--speculative-config "{\"method\":\"dflash\",\"model\":\"lightseekorg/kimi-k3-dflash2\",\"revision\":\"e77935fb4804e17eb55085bffd045eae1d779769\",\"num_speculative_tokens\":$proposals,\"attention_backend\":\"B12X_MLA\",\"kv_cache_dtype\":\"fp8\",\"draft_sample_method\":\"probabilistic\",\"rejection_sample_method\":\"block\",\"draft_load_config\":{\"load_format\":\"auto\"}$draft_quant_fields}")
    ;;
  *) echo 'KIMI_SPECULATOR must be none, dspark, dspark-redhat, dflash or dflash2' >&2; exit 2 ;;
esac
graph_sizes=()
for ((i=1; i<=sequences; i++)); do
  graph_sizes+=("$((i*width))")
  if ((draft_graph_width)); then graph_sizes+=("$((i*draft_graph_width))"); fi
done
mapfile -t graph_sizes < <(printf '%s\n' "${graph_sizes[@]}" | sort -nu)
graphs='['
for ((i=0; i<${#graph_sizes[@]}; i++)); do
  if ((i>0)); then graphs+=','; fi
  graphs+=${graph_sizes[i]}
done
graphs+=']'

# This profile preserves A16 expert activations and the draft's BF16 inputs.
# Cache hints and stream overlap do not modify weights or reduction order.
profile_args=()
mm_encoder_mode=${KIMI_MM_ENCODER_TP_MODE:-weights}
if [[ $format == qsrt_k2 && $mode == dspark-redhat && $tp == 9 && $dcp == 9 ]]; then
  export VLLM_KIMI_ALIGNED_DECODE_PROJECTIONS=1
  export VLLM_MLA_CHUNKED_PREFILL_WORKSPACE_SIZE=${VLLM_MLA_CHUNKED_PREFILL_WORKSPACE_SIZE:-65536}
  export VLLM_MLA_PREFILL_DCP_OVERLAP=1
  export VLLM_MLA_PREFILL_DCP_FP8_TRANSPORT=1
  export VLLM_KIMI_L2_PREFETCH=${VLLM_KIMI_L2_PREFETCH:-1}
  export VLLM_DISABLE_SHARED_EXPERTS_STREAM=${VLLM_DISABLE_SHARED_EXPERTS_STREAM:-0}
  export VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD=8
  export B12X_DYNAMIC_DETERMINISTIC_OUTPUT=1
  mm_encoder_mode=${KIMI_MM_ENCODER_TP_MODE:-data}
  profile_args=(--gpu-memory-utilization "${KIMI_GPU_MEMORY_UTILIZATION:-0.970}"
    --attention-config '{"mla_prefill_backend":"B12X"}')
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,large_segment_size_mb:12}
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_USE_BREAKABLE_CUDAGRAPH=1
export VLLM_B12X_MOE_FP4_FORCE_A16=1
export VLLM_B12X_MXFP8_ACTIVATION_MODE=a16
export B12X_MOE_WORKSPACE_TOKEN_LIMIT=4096
export B12X_W4A16_PREFILL_FUSED_SUM=1
export B12X_W4A16_STABLE_ROUTE_PACK=1
export B12X_W4A16_SMALL_M_HOST_BARRIER_RESET=0
export VLLM_DISABLE_SHARED_EXPERTS_STREAM=${VLLM_DISABLE_SHARED_EXPERTS_STREAM:-1}
export VLLM_ENABLE_PCIE_ALLREDUCE=1
export VLLM_PCIE_ALLREDUCE_BACKEND=b12x
export VLLM_PCIE_ONESHOT_SINGLE_CHANNEL=1
export B12X_PCIE_ALLREDUCE_ALGORITHM=island_rs
export B12X_PCIE_HIERARCHICAL_DEFERRED_CONSUMPTION=1
export B12X_PCIE_HIERARCHICAL_DOUBLE_BUFFER=0
export B12X_PCIE_HIERARCHICAL_THREADS=256
export B12X_PCIE_HIERARCHICAL_NANOSLEEP_CYCLES=24
export B12X_PCIE_HIERARCHICAL_BF16X2=1
export B12X_PCIE_HIERARCHICAL_BF16X2_MAX_ELEMENTS=7168
export B12X_PCIE_DCP_THREADS=512
export B12X_PCIE_DCP_BLOCK_LIMIT=8
export B12X_PCIE_KIMI_TOPK_THREADS=384
export VLLM_KIMI_SHARD_QKV_A=1
export VLLM_KIMI_SHARD_AUXILIARY_PROJECTIONS=${VLLM_KIMI_SHARD_AUXILIARY_PROJECTIONS:-1}
if [[ $VLLM_KIMI_SHARD_AUXILIARY_PROJECTIONS != 0 && $VLLM_KIMI_SHARD_AUXILIARY_PROJECTIONS != 1 ]]; then
  echo 'VLLM_KIMI_SHARD_AUXILIARY_PROJECTIONS must be 0 or 1' >&2
  exit 2
fi
export VLLM_KIMI_K3_AUX_ATTN_RES_STREAM=1
export VLLM_USE_B12X_DCP_A2A=1
export VLLM_MLA_CHUNKED_PREFILL_WORKSPACE_SIZE=${VLLM_MLA_CHUNKED_PREFILL_WORKSPACE_SIZE:-4096}
export INSTANTTENSOR_COPY=0
export INSTANTTENSOR_BUFFER_SIZE=536870912
export INSTANTTENSOR_IO_DEPTH=${INSTANTTENSOR_IO_DEPTH:-16}
export INSTANTTENSOR_BACKEND=AIO
export INSTANTTENSOR_MAX_FREE_MEM_USAGE=0.6
export VLLM_DISABLED_KERNELS=B12xMxfp8LinearKernel,FlashInferCutedslMxfp8LinearKernel,FlashInferCutlassMxfp8LinearKernel
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/cache/kimi-k3/triton}
export CUTE_DSL_CACHE_DIR=${CUTE_DSL_CACHE_DIR:-/cache/kimi-k3/cute}
export B12X_COMPILE_CACHE_DIR=${B12X_COMPILE_CACHE_DIR:-/cache/kimi-k3/b12x-compile}
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-/cache/kimi-k3/vllm}

command=(/opt/venv/bin/lil-runtime-bootstrap /opt/venv/bin/python
  -m vllm.entrypoints.cli.main serve
  "$checkpoint" --revision "$revision"
  --host 0.0.0.0 --port "${KIMI_PORT:-8000}"
  --served-model-name "$served_model"
  --trust-remote-code --tensor-parallel-size "$tp"
  --decode-context-parallel-size "$dcp" --dcp-comm-backend a2a
  --dcp-kv-cache-interleave-size 1 --mamba-block-size 12288
  --load-format instanttensor
  --model-loader-extra-config '{"instanttensor_priority_weight_name_prefixes":["vision_tower","mm_projector"],"instanttensor_small_checkpoint_max_bytes":8589934592}'
  --moe-backend b12x --linear-backend auto --attention-backend B12X_MLA
  --kda-prefill-backend triton --kv-cache-dtype fp8
  --kv-cache-memory-bytes "$kv_bytes"
  --max-model-len "${KIMI_MAX_MODEL_LEN:-950000}"
  --max-num-batched-tokens 4096 --max-num-scheduled-tokens 4096
  --max-num-seqs "$sequences"
  --enable-chunked-prefill --enable-prefix-caching
  --kv-offloading-size "${KIMI_NATIVE_KV_GIB:-32}" --kv-offloading-backend native
  --mm-processor-kwargs '{"in_patch_limit":40960,"patch_limit_on_one_side":512}'
  --mm-encoder-tp-mode "$mm_encoder_mode"
  --reasoning-parser kimi_k3 --tool-call-parser kimi_k3 --enable-auto-tool-choice
  --structured-outputs-config.backend xgrammar
  "${quant_args[@]}"
  --compilation-config "{\"mode\":0,\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"cudagraph_capture_sizes\":$graphs,\"pass_config\":{\"fuse_allreduce_rms\":true}}"
  "${spec_args[@]}" "${profile_args[@]}" "$@")
if [[ ${KIMI_PRINT_COMMAND:-0} == 1 ]]; then
  printf '%q ' "${command[@]}"
  printf '\n'
else
  exec "${command[@]}"
fi
