#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="$(mktemp -d)"
trap 'rm -rf "${tmp_root}"' EXIT

output="$({
  DRY_RUN=1 \
  XDG_CACHE_HOME="${tmp_root}/cache" \
  TMPDIR="${tmp_root}/tmp" \
  MODEL=/tmp/model \
  LOAD_FORMAT=dummy \
  bash "${repo_root}/launchers/serve-glm52-v16.sh" \
    --profiler-config.profiler=torch \
    --profiler-config.torch_profiler_dir=/tmp/profile
} 2>&1)"

grep -Fq -- '--profiler-config.profiler=torch' <<<"${output}"
grep -Fq -- '--profiler-config.torch_profiler_dir=/tmp/profile' <<<"${output}"
grep -Fxq 'VLLM_B12X_ABSORB_BMM=1' <<<"${output}"

disabled_absorb_output="$({
  DRY_RUN=1 \
  XDG_CACHE_HOME="${tmp_root}/cache" \
  TMPDIR="${tmp_root}/tmp" \
  MODEL=/tmp/model \
  LOAD_FORMAT=dummy \
  VLLM_B12X_ABSORB_BMM=0 \
  bash "${repo_root}/launchers/serve-glm52-v16.sh"
} 2>&1)"
grep -Fxq 'VLLM_B12X_ABSORB_BMM=0' <<<"${disabled_absorb_output}"

for mode in i8 i8_ring i8_a2a mx mx_ring mx_a2a; do
  mode_output="$({
    DRY_RUN=1 \
    XDG_CACHE_HOME="${tmp_root}/cache" \
    TMPDIR="${tmp_root}/tmp" \
    MODEL=/tmp/model \
    LOAD_FORMAT=dummy \
    F8_DMA="${mode}" \
    bash "${repo_root}/launchers/serve-glm52-v16.sh"
  } 2>&1)"
  grep -Fxq "VLLM_PCIE_DMA_FP8=${mode}" <<<"${mode_output}"
  grep -Fxq "SPARKINFER_PCIE_DMA_FP8=${mode}" <<<"${mode_output}"
done
