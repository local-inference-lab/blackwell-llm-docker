#!/usr/bin/env bash
# Verify that the Kimi source overlay is opt-in and aborts on invalid identity.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
overlay_dir="${repo_root}/runtime/kimi-k3-qsrt/source-overlay"
scratch_dir="$(mktemp -d)"
trap 'rm -rf "${scratch_dir}"' EXIT
missing_source="${scratch_dir}/missing-vllm-source"

grep -Fq \
  'COPY --chmod=0755 launchers/lmcache-shm-preflight.py /usr/local/bin/lmcache-shm-preflight.py' \
  "${repo_root}/Dockerfile.kimi-k3-qsrt-tp16-runtime"

output="$({
  PYTHONPATH="${overlay_dir}" \
    VLLM_SOURCE_OVERLAY_ROOT="${missing_source}" \
    python3 -c 'print("inactive overlay did not import vLLM")'
} 2>&1)"
grep -Fxq 'inactive overlay did not import vLLM' <<<"${output}"

if PYTHONPATH="${overlay_dir}" \
    VLLM_SOURCE_OVERLAY_ACTIVE=1 \
    VLLM_SOURCE_OVERLAY_ROOT="${missing_source}" \
    python3 -c 'print("invalid overlay continued")' >/dev/null 2>&1; then
  printf 'An invalid active vLLM source overlay did not abort Python startup\n' >&2
  exit 1
fi

printf 'Kimi source overlay activation policy: PASS\n'
