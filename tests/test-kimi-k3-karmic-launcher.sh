#!/usr/bin/env bash
# Check allocator, grammar-backend, and graph policy without loading a model.
set -euo pipefail
readonly recipe=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
for mode in none dspark dflash dflash2; do
  env -u PYTORCH_CUDA_ALLOC_CONF KIMI_SPECULATOR="$mode" \
    KIMI_PRINT_COMMAND=1 KIMI_KV_BYTES=3221225472 bash -c '
      source "$1" >/dev/null
      [[ $PYTORCH_CUDA_ALLOC_CONF == expandable_segments:True,large_segment_size_mb:12 ]]
      [[ " ${command[*]} " == *" --structured-outputs-config.backend xgrammar "* ]]
      if [[ $KIMI_SPECULATOR == dflash ]]; then
        [[ $VLLM_DFLASH_AUX_MXFP8_STREAMING == 1 && $VLLM_DFLASH_COMPACT_ROPE == 1 ]]
        [[ $VLLM_DFLASH_SHARD_AUX_PROJECTION == 1 ]]
      fi
    ' -- "$recipe/runtime/serve-kimi-k3.sh"
  env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,large_segment_size_mb:16 \
    KIMI_SPECULATOR="$mode" KIMI_PRINT_COMMAND=1 KIMI_KV_BYTES=3221225472 bash -c '
      source "$1" >/dev/null
      [[ $PYTORCH_CUDA_ALLOC_CONF == expandable_segments:True,large_segment_size_mb:16 ]]
    ' -- "$recipe/runtime/serve-kimi-k3.sh"
done
for proposals in 3 4 7; do
  KIMI_SPECULATOR=dflash2 KIMI_PRINT_COMMAND=1 KIMI_KV_BYTES=3221225472 \
    KIMI_MAX_SEQS=2 KIMI_DFLASH2_SPEC_TOKENS="$proposals" bash -c '
      source "$1" >/dev/null
      [[ $width == $((KIMI_DFLASH2_SPEC_TOKENS+1)) ]]
      [[ $graphs == "[$width,$((2*width))]" ]]
      [[ ${spec_args[1]} == *"lightseekorg/kimi-k3-dflash2"* ]]
      [[ ${spec_args[1]} != *"quantization"* ]]
      [[ $VLLM_K3_KV_GROUP_SIZE == 3 ]]
    ' -- "$recipe/runtime/serve-kimi-k3.sh"
done
if KIMI_SPECULATOR=dflash2 KIMI_PRINT_COMMAND=1 KIMI_KV_BYTES=3221225472 \
  KIMI_DFLASH2_SPEC_TOKENS=8 bash "$recipe/runtime/serve-kimi-k3.sh" >/dev/null 2>&1; then
  echo 'An unsupported DFlash2 proposal count was accepted' >&2
  exit 1
fi
printf 'Kimi-K3 allocator, draft dtype, graph sizing, and input validation: 12 passed.\n'
