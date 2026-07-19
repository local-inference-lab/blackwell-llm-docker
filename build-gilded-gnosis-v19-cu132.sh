#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Unified GLM 5.2 and DS4/DSpark image built from canonical Gilded Gnosis plus
# the independently reviewable SM120 CUTLASS DSL pin and DCP A2A prewarm fix.
export IMAGE="${IMAGE:-voipmonitor/vllm:gilded-gnosis-v19-vllm6b74780-b12xc7dc733-fi801d57a-cu132-20260719}"

export VLLM_REPO="${VLLM_REPO:-https://github.com/voipmonitor/vllm.git}"
export VLLM_REF="${VLLM_REF:-build/gilded-gnosis-v19-final2-20260719}"
export VLLM_COMMIT="${VLLM_COMMIT:-6b74780750013b52d82cb767578445c8fa96f8ed}"
export VLLM_BUILD_VERSION="${VLLM_BUILD_VERSION:-0.11.2.dev280+gilded.gnosis.v19.vllm6b74780.b12xc7dc733.fi801d57a.cu132.20260719}"

# Canonical B12X master including the merged stable CuTe cache-key fix.
export B12X_REPO="${B12X_REPO:-https://github.com/voipmonitor/b12x.git}"
export B12X_REF="${B12X_REF:-build/gilded-gnosis-v19-final-b12x-canonical-20260719}"
export B12X_COMMIT="${B12X_COMMIT:-c7dc73322cc50609f843fa2bbcc53283a90003b3}"

# Keep the rebased upstream CUTLASS C++ source pin used to build FlashInfer.
# CuTe DSL 4.6.0 regresses the B12X W4A16 prefill kernel on SM120 through
# register spilling, so the runtime compiler is deliberately pinned to 4.5.3.
export CUTLASS_REF="${CUTLASS_REF:-e6233cbac5d7c7a865c19c91cd684ceece19513c}"
export CUTLASS_COMMIT="${CUTLASS_COMMIT:-e6233cbac5d7c7a865c19c91cd684ceece19513c}"
export CUTLASS_DSL_VERSION="${CUTLASS_DSL_VERSION:-4.5.3}"
export TOKENSPEED_MLA_VERSION="${TOKENSPEED_MLA_VERSION:-0.1.8}"
export TVM_FFI_VERSION="${TVM_FFI_VERSION:-0.1.10}"

export VLLM_REQUIRED_LAUNCHERS="serve-gilded-gnosis.sh serve-fathomless-firmament.sh serve-ds4-flash.sh serve-glm52-v16.sh serve-glm52-v18.sh serve-glm52-v19.sh serve-glm52-hybrid-v17.sh serve-glm52-hybrid-v18.sh serve-glm52-hybrid-v19.sh"

requested_push="${PUSH_IMAGE:-0}"
export PUSH_IMAGE=0

# Reuse the audited CUDA 13.2/NCCL/InstantTensor/FlashInfer build substrate and
# its v18 compatibility assertions while overriding all mutable source pins.
./build-gilded-gnosis-v18-cu132.sh "$@"

labels="$(docker image inspect "${IMAGE}" --format '{{json .Config.Labels}}')"
jq -e --arg value "${VLLM_COMMIT}" '."local-inference.vllm.commit" == $value' <<<"${labels}" >/dev/null
jq -e --arg value "${B12X_COMMIT}" '."local-inference.b12x.commit" == $value' <<<"${labels}" >/dev/null
jq -e --arg value "801d57a08958c13d375ddbb6be3be4808f48a708" '."local-inference.flashinfer.commit" == $value' <<<"${labels}" >/dev/null

docker run --rm --entrypoint /opt/venv/bin/python "${IMAGE}" - <<'PY'
import importlib.metadata as md
import hashlib
from pathlib import Path

import cutlass.cute as cute
from b12x.moe.fused.w4a16 import kernel as w4a16_kernel

from vllm import envs
from vllm.distributed.device_communicators import symm_mem_pcie_barrier
from vllm.model_executor.layers import fp8_draft_head
from vllm.v1.worker.gpu.spec_decode import capacity
from vllm.v1.worker.gpu.spec_decode.dspark import online_sts

assert md.version("nvidia-cutlass-dsl") == "4.5.3"
assert md.version("nvidia-cutlass-dsl-libs-base") == "4.5.3"
assert md.version("nvidia-cutlass-dsl-libs-cu13") == "4.5.3"
assert hasattr(cute.nvgpu.warp, "MmaMXF8Op")
mma = Path(cute.__file__).parent / "nvgpu" / "warp" / "mma.py"
assert hashlib.sha256(mma.read_bytes()).hexdigest() == (
    "cccf48864bae9554acdf708fd66a7b2fe729948ca3768205df3e251c8dc71fd2"
)
assert md.version("tokenspeed-mla") == "0.1.8"
assert md.version("apache-tvm-ffi") == "0.1.10"
assert hasattr(envs, "VLLM_ALLOW_CUSTOM_ALLREDUCE_PCIE")
assert hasattr(envs, "VLLM_SYMM_MEM_PCIE_SAFE_BARRIER")
assert hasattr(envs, "VLLM_DSPARK_FP8_DRAFT_HEAD")
assert hasattr(envs, "VLLM_DSPARK_DYNAMIC_DRAFT_DEPTH")
assert hasattr(envs, "VLLM_DSPARK_CAPACITY_ACTIVATION_BATCH_SIZE")
assert hasattr(symm_mem_pcie_barrier, "install_pcie_safe_barrier")
assert hasattr(fp8_draft_head, "Fp8DraftHead")
assert hasattr(capacity, "DSparkDynamicDraftDepthController")
assert hasattr(capacity, "CapacityBasedVerificationManager")
assert hasattr(online_sts, "DSparkOnlineSTS")
assert hasattr(w4a16_kernel, "compile_w4a16_fused_moe_hybrid")
assert hasattr(w4a16_kernel, "run_w4a16_moe_hybrid")
assert not hasattr(w4a16_kernel, "w4a16_hybrid_mapped_grid188_mapping_proof")
tier_map = w4a16_kernel.build_w4a16_tier_local_map(
    [1, 3], [0, 2], map_slots=4
)
assert tier_map.tolist() == [256, 0, 257, 1]
print("v19 rebased runtime symbols: PASS")
PY

dry_run() {
  local name="$1"
  shift
  docker run --rm --entrypoint /usr/local/bin/serve-gilded-gnosis.sh \
    -e DRY_RUN=1 \
    -e MODEL_FAMILY=glm52 \
    -e MODEL=/model \
    "$@" \
    "${IMAGE}" | tee "/tmp/gilded-gnosis-v19-${name}.txt"
}

for profile in tp4-dcp2 tp4-dcp4 tp8-dcp2 tp8-dcp4 tp8-dcp8; do
  tp="${profile#tp}"
  tp="${tp%%-*}"
  dcp="${profile##*-dcp}"
  dry_run "${profile}" -e TP="${tp}" -e DCP="${dcp}"
  grep -q '^VLLM_DCP_QUERY_SPLIT=1$' "/tmp/gilded-gnosis-v19-${profile}.txt"
  grep -q '^VLLM_B12X_MLA_CKV_GATHER=1$' "/tmp/gilded-gnosis-v19-${profile}.txt"
done
grep -q -- '--load-format instanttensor' /tmp/gilded-gnosis-v19-tp8-dcp4.txt

dry_run tp6-dcp3 -e TP=6 -e DCP=3
grep -q '^VLLM_DCP_QUERY_SPLIT=0$' /tmp/gilded-gnosis-v19-tp6-dcp3.txt
grep -q '^VLLM_B12X_MLA_CKV_GATHER=0$' /tmp/gilded-gnosis-v19-tp6-dcp3.txt

dry_run tp6-dcp6-mtp3 -e TP=6 -e DCP=6 -e MTP=3
grep -q -- '--tensor-parallel-size 6' /tmp/gilded-gnosis-v19-tp6-dcp6-mtp3.txt
grep -q -- '--decode-context-parallel-size 6' /tmp/gilded-gnosis-v19-tp6-dcp6-mtp3.txt
grep -q 'num_speculative_tokens.*3' /tmp/gilded-gnosis-v19-tp6-dcp6-mtp3.txt

docker run --rm --entrypoint /usr/local/bin/serve-gilded-gnosis.sh \
  -e MODEL_FAMILY=glm52-hybrid \
  -e DRY_RUN=1 \
  -e MODEL=/model \
  "${IMAGE}" | tee /tmp/gilded-gnosis-v19-nf3.txt
grep -q '^ONLINE_QUANT=nf3-mxfp8$' /tmp/gilded-gnosis-v19-nf3.txt
grep -q '^VLLM_NF3_GRID188_DECODE=1$' /tmp/gilded-gnosis-v19-nf3.txt
grep -q -- '--kv-cache-dtype nvfp4_ds_mla' /tmp/gilded-gnosis-v19-nf3.txt
grep -q -- '--load-format instanttensor' /tmp/gilded-gnosis-v19-nf3.txt

ds4_dry_run() {
  local name="$1"
  shift
  docker run --rm --entrypoint /usr/local/bin/serve-gilded-gnosis.sh \
    -e MODEL_FAMILY=ds4 \
    -e DRY_RUN=1 \
    -e MODEL=/model \
    "$@" \
    "${IMAGE}" 2>&1 | tee "/tmp/gilded-gnosis-v19-ds4-${name}.txt"
}

ds4_dry_run default \
  -e MODE=dspark \
  -e BACKEND=lucifer-cutlass \
  -e TP_SIZE=4
grep -q 'load_format=instanttensor instanttensor_backend=BUFFERED' \
  /tmp/gilded-gnosis-v19-ds4-default.txt

ds4_dry_run adaptive \
  -e MODE=dspark \
  -e BACKEND=lucifer-cutlass \
  -e TP_SIZE=4 \
  -e DSPARK_CAPACITY=1 \
  -e DSPARK_DYNAMIC_DRAFT_DEPTH=1 \
  -e DSPARK_FP8_DRAFT_HEAD=1
grep -q 'dspark_capacity_verification_mode' \
  /tmp/gilded-gnosis-v19-ds4-adaptive.txt

docker run --rm --entrypoint /bin/bash "${IMAGE}" -lc \
  'grep -q "export VLLM_DSPARK_FP8_DRAFT_HEAD" /usr/local/bin/serve-ds4-flash.sh && grep -q "export VLLM_DSPARK_DYNAMIC_DRAFT_DEPTH" /usr/local/bin/serve-ds4-flash.sh'

if [[ "${requested_push}" == "1" ]]; then
  docker push "${IMAGE}"
fi

printf 'Image: %s\n' "${IMAGE}"
