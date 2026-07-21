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
