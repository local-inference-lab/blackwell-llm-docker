#!/usr/bin/env bash
# Check allocator overrides without starting the model or requiring CUDA.
set -euo pipefail
readonly recipe=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
for mode in none dspark dflash; do
  env -u PYTORCH_CUDA_ALLOC_CONF KIMI_SPECULATOR="$mode" \
    KIMI_PRINT_COMMAND=1 bash -c '
      source "$1" >/dev/null
      [[ $PYTORCH_CUDA_ALLOC_CONF == expandable_segments:True,large_segment_size_mb:12 ]]
      if [[ $KIMI_SPECULATOR == dflash ]]; then
        [[ $VLLM_DFLASH_AUX_MXFP8_STREAMING == 1 && $VLLM_DFLASH_COMPACT_ROPE == 1 ]]
      fi
    ' -- "$recipe/runtime/serve-kimi-k3.sh"
  env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,large_segment_size_mb:16 \
    KIMI_SPECULATOR="$mode" KIMI_PRINT_COMMAND=1 bash -c '
      source "$1" >/dev/null
      [[ $PYTORCH_CUDA_ALLOC_CONF == expandable_segments:True,large_segment_size_mb:16 ]]
    ' -- "$recipe/runtime/serve-kimi-k3.sh"
done
printf 'Kimi-K3 Karmic allocator defaults and overrides: 6 passed.\n'
