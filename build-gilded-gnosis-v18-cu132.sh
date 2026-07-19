#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Unified GLM 5.2 and DS4/DSpark image built from dev/gilded-gnosis plus the
# explicitly pinned DSpark, SM120 PCIe, DCP-prefill, and NF3 decode stacks.
export IMAGE="${IMAGE:-voipmonitor/vllm:gilded-gnosis-v18-vllm264bce1-b12xbc85ef3-fi801d57a-cu132-20260718}"

export VLLM_REPO="${VLLM_REPO:-https://github.com/local-inference-lab/vllm.git}"
export VLLM_REF="${VLLM_REF:-build/gilded-gnosis-v18-final-20260718}"
export VLLM_COMMIT="${VLLM_COMMIT:-264bce1da81e27d638e7cf265b4cbd125d023c38}"
export VLLM_BUILD_VERSION="${VLLM_BUILD_VERSION:-0.11.2.dev280+gilded.gnosis.v18.vllm264bce1.b12xbc85ef3.fi801d57a.cu132.20260718}"

# Current lukealonso/b12x master plus the ready-for-review Grid188 PR #36.
# Replace the ref with upstream master after merge; keep the immutable commit.
export B12X_REPO="${B12X_REPO:-https://github.com/voipmonitor/b12x.git}"
export B12X_REF="${B12X_REF:-codex/nf3-grid188-decode-20260717}"
export B12X_COMMIT="${B12X_COMMIT:-bc85ef36192cb6e444d42ba7be86e1e125cca98a}"

export VLLM_REQUIRED_LAUNCHERS="${VLLM_REQUIRED_LAUNCHERS:-serve-gilded-gnosis.sh serve-fathomless-firmament.sh serve-ds4-flash.sh serve-glm52-v16.sh serve-glm52-v18.sh serve-glm52-hybrid-v17.sh serve-glm52-hybrid-v18.sh}"

requested_push="${PUSH_IMAGE:-0}"
export PUSH_IMAGE=0

./build-fathomless-firmament-v16-cu132.sh "$@"

labels="$(docker image inspect "${IMAGE}" --format '{{json .Config.Labels}}')"
jq -e --arg value "${VLLM_COMMIT}" '."local-inference.vllm.commit" == $value' <<<"${labels}" >/dev/null
jq -e --arg value "${B12X_COMMIT}" '."local-inference.b12x.commit" == $value' <<<"${labels}" >/dev/null
jq -e --arg value "801d57a08958c13d375ddbb6be3be4808f48a708" '."local-inference.flashinfer.commit" == $value' <<<"${labels}" >/dev/null

docker run --rm --gpus device=0 -i --entrypoint /opt/venv/bin/python "${IMAGE}" - <<'PY'
from typing import get_args

import torch
from vllm import _custom_ops  # noqa: F401
from vllm import envs
from vllm.config.cache import CacheDType
from vllm.config.quantization import resolve_quantization_config
from vllm.config.speculative import SpeculativeConfig
from vllm.v1.attention.backends.mla import b12x_mla_sparse
from vllm.v1.attention.ops import dcp_alltoall
from b12x.moe.fused.w4a16 import kernel as w4a16_kernel

assert hasattr(envs, "VLLM_DCP_QUERY_SPLIT")
assert hasattr(envs, "VLLM_B12X_MLA_CKV_GATHER")
assert hasattr(envs, "VLLM_NF3_GRID188_DECODE")
assert hasattr(b12x_mla_sparse, "_global_causal_lens_for_ckv_gather")
assert hasattr(SpeculativeConfig, "_inherit_target_revision_for_mtp")
assert hasattr(dcp_alltoall, "_DCP_A2A_GRAPH_BUFFERS")
assert "nvfp4_ds_mla" in get_args(CacheDType)
assert hasattr(torch.ops._C_cache_ops, "concat_and_cache_nvfp4_mla")
assert resolve_quantization_config("nvfp4_nf3_hybrid", {"linear": {"weight": "mxfp8"}})
if hasattr(w4a16_kernel, "w4a16_hybrid_mapped_grid188_mapping_proof"):
    proof = w4a16_kernel.w4a16_hybrid_mapped_grid188_mapping_proof()
    assert proof["grid_x"] == 188
    assert len(proof["fc1_tasks"]) == 128
    assert len(proof["fc2_tasks"]) == 768
else:
    assert hasattr(w4a16_kernel, "compile_w4a16_fused_moe_hybrid")
    assert hasattr(w4a16_kernel, "run_w4a16_moe_hybrid")
    tier_map = w4a16_kernel.build_w4a16_tier_local_map(
        [1, 3], [0, 2], map_slots=4
    )
    assert tier_map.tolist() == [256, 0, 257, 1]

# Exercise the SM100+ writer, not just its Python/C++ registration.  The
# stable-libtorch extension is loaded lazily and requires a CUDA driver.
torch.manual_seed(7)
num_blocks, block_size, num_tokens = 2, 16, 5
kv_c = torch.randn(num_tokens, 512, dtype=torch.bfloat16, device="cuda")
k_pe = torch.randn(num_tokens, 64, dtype=torch.bfloat16, device="cuda")
slots = torch.tensor([0, 3, 7, 18, 29], dtype=torch.long, device="cuda")
scale = torch.tensor(1.0, dtype=torch.float32, device="cuda")
cache = torch.zeros(
    num_blocks, block_size, 432, dtype=torch.uint8, device="cuda"
)
_custom_ops.concat_and_cache_mla(
    kv_c, k_pe, cache, slots, "nvfp4_ds_mla", scale
)
torch.cuda.synchronize()
flat = cache.reshape(-1, 432)
selected = flat[slots]
assert (selected[:, :288] != 0).any(dim=1).all()
assert torch.equal(selected[:, 288:304], torch.zeros_like(selected[:, 288:304]))
assert torch.equal(
    selected[:, 304:432],
    k_pe.contiguous().view(torch.uint8).reshape(num_tokens, 128),
)
untouched = torch.ones(flat.shape[0], dtype=torch.bool, device="cuda")
untouched[slots] = False
assert torch.count_nonzero(flat[untouched]) == 0
print("nvfp4_ds_mla CUDA writer: PASS")
PY

dry_run() {
  local name="$1"
  shift
  docker run --rm --entrypoint /usr/local/bin/serve-glm52-v18.sh \
    -e DRY_RUN=1 \
    -e MODEL=/model \
    "$@" \
    "${IMAGE}" | tee "/tmp/gilded-gnosis-v18-${name}.txt"
}

dry_run tp8-dcp4 -e TP=8 -e DCP=4
grep -q '^VLLM_DCP_QUERY_SPLIT=1$' /tmp/gilded-gnosis-v18-tp8-dcp4.txt
grep -q '^VLLM_B12X_MLA_CKV_GATHER=1$' /tmp/gilded-gnosis-v18-tp8-dcp4.txt
grep -q -- '--load-format instanttensor' /tmp/gilded-gnosis-v18-tp8-dcp4.txt

dry_run tp8-dcp8 -e TP=8 -e DCP=8
grep -q '^VLLM_DCP_QUERY_SPLIT=1$' /tmp/gilded-gnosis-v18-tp8-dcp8.txt
grep -q '^VLLM_B12X_MLA_CKV_GATHER=1$' /tmp/gilded-gnosis-v18-tp8-dcp8.txt

dry_run tp6-dcp3-mtp3 -e TP=6 -e DCP=3 -e MTP=3
grep -q -- '--tensor-parallel-size 6' /tmp/gilded-gnosis-v18-tp6-dcp3-mtp3.txt
grep -q -- '--decode-context-parallel-size 3' /tmp/gilded-gnosis-v18-tp6-dcp3-mtp3.txt
grep -q 'num_speculative_tokens.*3' /tmp/gilded-gnosis-v18-tp6-dcp3-mtp3.txt

dry_run tp6-dcp6-mtp3 -e TP=6 -e DCP=6 -e MTP=3
grep -q -- '--tensor-parallel-size 6' /tmp/gilded-gnosis-v18-tp6-dcp6-mtp3.txt
grep -q -- '--decode-context-parallel-size 6' /tmp/gilded-gnosis-v18-tp6-dcp6-mtp3.txt

dry_run forced-off \
  -e TP=8 \
  -e DCP=4 \
  -e DCP_QUERY_SPLIT=0 \
  -e DCP_CKV_GATHER=0
grep -q '^VLLM_DCP_QUERY_SPLIT=0$' /tmp/gilded-gnosis-v18-forced-off.txt
grep -q '^VLLM_B12X_MLA_CKV_GATHER=0$' /tmp/gilded-gnosis-v18-forced-off.txt

docker run --rm --entrypoint /usr/local/bin/serve-glm52-hybrid-v18.sh \
  -e DRY_RUN=1 \
  -e MODEL=/model \
  "${IMAGE}" | tee /tmp/gilded-gnosis-v18-nf3.txt
grep -q '^ONLINE_QUANT=nf3-mxfp8$' /tmp/gilded-gnosis-v18-nf3.txt
grep -q '^VLLM_NF3_GRID188_DECODE=1$' /tmp/gilded-gnosis-v18-nf3.txt
grep -q 'shared_experts' /tmp/gilded-gnosis-v18-nf3.txt
grep -q -- '--quantization nvfp4_nf3_hybrid' /tmp/gilded-gnosis-v18-nf3.txt
grep -q -- '--kv-cache-dtype nvfp4_ds_mla' /tmp/gilded-gnosis-v18-nf3.txt
grep -q -- '--load-format instanttensor' /tmp/gilded-gnosis-v18-nf3.txt

docker run --rm --entrypoint /usr/local/bin/serve-gilded-gnosis.sh \
  -e MODEL_FAMILY=glm52-hybrid \
  -e DRY_RUN=1 \
  -e MODEL=/model \
  "${IMAGE}" | grep -q '^ONLINE_QUANT=nf3-mxfp8$'

ds4_dry_run() {
  local mode="$1"
  local backend="$2"
  local tp="$3"
  docker run --rm --entrypoint /usr/local/bin/serve-ds4-flash.sh \
    -e DRY_RUN=1 \
    -e MODE="${mode}" \
    -e BACKEND="${backend}" \
    -e TP_SIZE="${tp}" \
    -e MODEL=/model \
    "${IMAGE}" 2>&1 | tee "/tmp/gilded-gnosis-v18-ds4-${mode}-${backend}.txt"
}

ds4_dry_run mtp0 b12x-a16 2
ds4_dry_run mtp2 b12x-a8 2
ds4_dry_run mtp3 b12x-a8-dglin 2
ds4_dry_run dspark lucifer-cutlass 4
grep -q 'graph=512 load_format=instanttensor instanttensor_backend=BUFFERED' \
  /tmp/gilded-gnosis-v18-ds4-mtp2-b12x-a8.txt
grep -q 'graph=384 load_format=instanttensor instanttensor_backend=BUFFERED' \
  /tmp/gilded-gnosis-v18-ds4-dspark-lucifer-cutlass.txt
if grep -q -- '--revision' /tmp/gilded-gnosis-v18-ds4-mtp0-b12x-a16.txt; then
  echo 'ERROR: DS4 helper injected an HF revision for a local model path' >&2
  exit 1
fi

if [[ "${requested_push}" == "1" ]]; then
  docker push "${IMAGE}"
fi

printf 'Image: %s\n' "${IMAGE}"
