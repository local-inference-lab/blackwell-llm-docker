#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export IMAGE="${IMAGE:-voipmonitor/vllm:fathomless-firmament-v17-vllm70c49aa-b12x8648a2b-fi801d57a-cu132-20260714}"

export VLLM_REPO="${VLLM_REPO:-https://github.com/local-inference-lab/vllm.git}"
export VLLM_REF="${VLLM_REF:-codex/fathomless-firmament-v17-nf3-nvfp4kv-20260714}"
export VLLM_COMMIT="${VLLM_COMMIT:-70c49aad686417aa2f15123731971d56edb4ded6}"
export VLLM_BUILD_VERSION="${VLLM_BUILD_VERSION:-0.11.2.dev280+fathomless.firmament.v17.vllm70c49aa.b12x8648a2b.fi801d57a.cu132.20260714}"

export B12X_REPO="${B12X_REPO:-https://github.com/voipmonitor/b12x.git}"
export B12X_REF="${B12X_REF:-codex/fathomless-firmament-v17-nf3-nvfp4kv-20260714}"
export B12X_COMMIT="${B12X_COMMIT:-8648a2bb751b2d695f24f2a229f98c42be394a00}"

export VLLM_REQUIRED_LAUNCHERS="serve-fathomless-firmament.sh serve-ds4-flash.sh serve-glm52-v16.sh serve-glm52-hybrid-v17.sh"

requested_push="${PUSH_IMAGE:-0}"
export PUSH_IMAGE=0

./build-fathomless-firmament-v16-cu132.sh "$@"

docker run --rm --entrypoint /usr/local/bin/serve-glm52-hybrid-v17.sh \
  -e DRY_RUN=1 \
  -e MODEL=/model \
  -e DCP=4 \
  "${IMAGE}" | tee /tmp/fathomless-firmament-v17-hybrid-dry-run.txt

grep -q -- '--tensor-parallel-size 4' /tmp/fathomless-firmament-v17-hybrid-dry-run.txt
grep -q -- '--decode-context-parallel-size 4' /tmp/fathomless-firmament-v17-hybrid-dry-run.txt
grep -q -- '--kv-cache-dtype nvfp4_ds_mla' /tmp/fathomless-firmament-v17-hybrid-dry-run.txt
grep -q -- '--quantization nvfp4_nf3_hybrid' /tmp/fathomless-firmament-v17-hybrid-dry-run.txt
grep -q -- '--load-format instanttensor' /tmp/fathomless-firmament-v17-hybrid-dry-run.txt

if [[ "${requested_push}" == "1" ]]; then
  docker push "${IMAGE}"
fi

printf 'Image: %s\n' "${IMAGE}"
