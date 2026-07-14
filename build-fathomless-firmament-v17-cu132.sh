#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export IMAGE="${IMAGE:-voipmonitor/vllm:fathomless-firmament-v17-vllm0db5be0-b12x1377d5f-fi801d57a-cu132-20260714}"

export VLLM_REPO="${VLLM_REPO:-https://github.com/local-inference-lab/vllm.git}"
export VLLM_REF="${VLLM_REF:-codex/fathomless-firmament-v17-dcp-prefill-opt-20260714}"
export VLLM_COMMIT="${VLLM_COMMIT:-0db5be0a1ec21f92f6e134b372bfc27271f62995}"
export VLLM_BUILD_VERSION="${VLLM_BUILD_VERSION:-0.11.2.dev280+fathomless.firmament.v17.vllm0db5be0.b12x1377d5f.fi801d57a.cu132.20260714}"

export B12X_REPO="${B12X_REPO:-https://github.com/voipmonitor/b12x.git}"
export B12X_REF="${B12X_REF:-codex/fathomless-firmament-v17-nf3-nvfp4kv-20260714}"
export B12X_COMMIT="${B12X_COMMIT:-1377d5f22c98de0c17d9b3f35a5b56d7587992fa}"

export VLLM_REQUIRED_LAUNCHERS="serve-fathomless-firmament.sh serve-ds4-flash.sh serve-glm52-v16.sh serve-glm52-hybrid-v17.sh"

requested_push="${PUSH_IMAGE:-0}"
export PUSH_IMAGE=0

./build-fathomless-firmament-v16-cu132.sh "$@"

docker run --rm --entrypoint /usr/local/bin/serve-glm52-hybrid-v17.sh \
  -e DRY_RUN=1 \
  -e MODEL=/model \
  -e DCP=4 \
  -e MAX_BATCHED_TOKENS=3072 \
  "${IMAGE}" | tee /tmp/fathomless-firmament-v17-hybrid-dry-run.txt

grep -q -- '--tensor-parallel-size 4' /tmp/fathomless-firmament-v17-hybrid-dry-run.txt
grep -q -- '--decode-context-parallel-size 4' /tmp/fathomless-firmament-v17-hybrid-dry-run.txt
grep -q -- '--kv-cache-dtype nvfp4_ds_mla' /tmp/fathomless-firmament-v17-hybrid-dry-run.txt
grep -q -- '--quantization nvfp4_nf3_hybrid' /tmp/fathomless-firmament-v17-hybrid-dry-run.txt
grep -q -- '--load-format instanttensor' /tmp/fathomless-firmament-v17-hybrid-dry-run.txt
grep -q -- 'VLLM_DCP_PROJECT_BEFORE_MERGE=1' /tmp/fathomless-firmament-v17-hybrid-dry-run.txt
grep -q -- 'VLLM_B12X_MLA_DCP_GATHER_IN_WORKSPACE=1' /tmp/fathomless-firmament-v17-hybrid-dry-run.txt

if [[ "${requested_push}" == "1" ]]; then
  docker push "${IMAGE}"
fi

printf 'Image: %s\n' "${IMAGE}"
