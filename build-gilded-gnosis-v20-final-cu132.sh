#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# GG v20 canonical-head release candidate. The vLLM integration source is
# exactly dev/gilded-gnosis plus unmerged PRs #145 and #164. SparkInfer is the
# canonical master containing the reviewed v20 fixes, including #48; closed
# overlap experiments #60/#150 and sparse-CKV/QBMM experiments are excluded.
export IMAGE="${IMAGE:-voipmonitor/vllm:gilded-gnosis-v20-vllm3e731bc-si1a88b38-fi801d57a-cu132-20260722}"
export SYSTEM_BASE_IMAGE="${SYSTEM_BASE_IMAGE:-voipmonitor/vllm:glm-kimi-cu132-system-base-20260626}"
export BUILD_BASE_IMAGE_TAG="${BUILD_BASE_IMAGE_TAG:-voipmonitor/vllm:glm-kimi-cu132-build-base-20260626}"
export BUILD_BASE_IMAGE="${BUILD_BASE_IMAGE:-0}"
export PUSH_BASE_IMAGE="${PUSH_BASE_IMAGE:-0}"
export MAX_JOBS="${MAX_JOBS:-64}"
export VLLM_MAX_JOBS="${VLLM_MAX_JOBS:-64}"
export NVCC_THREADS="${NVCC_THREADS:-1}"
export VLLM_NVCC_THREADS="${VLLM_NVCC_THREADS:-1}"
export PIN_SOURCE_COMMITS=1

export NCCL_REPO="${NCCL_REPO:-https://github.com/local-inference-lab/nccl-canonical.git}"
export NCCL_REF="${NCCL_REF:-canonical/cu132-nccl2304-amd-noxml}"
export NCCL_COMMIT="${NCCL_COMMIT:-dfab7c1ace32da250ba97757879429c341b7bcf9}"

export FLASHINFER_REPO="${FLASHINFER_REPO:-https://github.com/voipmonitor/flashinfer.git}"
export FLASHINFER_REF="${FLASHINFER_REF:-codex/sm120-dspark-stack-20260711}"
export FLASHINFER_COMMIT="${FLASHINFER_COMMIT:-801d57a08958c13d375ddbb6be3be4808f48a708}"
export FLASHINFER_BUILD_CUBIN="${FLASHINFER_BUILD_CUBIN:-0}"

export DEEPGEMM_REPO="${DEEPGEMM_REPO:-https://github.com/deepseek-ai/DeepGEMM.git}"
export DEEPGEMM_REF="${DEEPGEMM_REF:-a6b593d2826719dcf4892609af7b84ee23aaf32a}"
export DEEPGEMM_COMMIT="${DEEPGEMM_COMMIT:-a6b593d2826719dcf4892609af7b84ee23aaf32a}"

export VLLM_REPO="${VLLM_REPO:-https://github.com/voipmonitor/vllm.git}"
export VLLM_REF="${VLLM_REF:-build/gilded-gnosis-v20-canonical-145-164-20260722}"
export VLLM_COMMIT="${VLLM_COMMIT:-3e731bc043d23ec21277fb76d3e15fe6da91b23b}"
export VLLM_BUILD_VERSION="${VLLM_BUILD_VERSION:-0.11.2.dev280+gilded.gnosis.v20.vllm3e731bc.si1a88b38.fi801d57a.cu132.20260722}"
export VLLM_PATCH_URL=
export VLLM_PATCH_SHA256=
export VLLM_PATCH_FILE=

export SPARKINFER_REPO="${SPARKINFER_REPO:-https://github.com/local-inference-lab/sparkinfer.git}"
export SPARKINFER_REF="${SPARKINFER_REF:-master}"
export SPARKINFER_COMMIT="${SPARKINFER_COMMIT:-1a88b389a8d14f26dbe4c157965938cfd8f1bf51}"

export LAUNCHER_REPO="${LAUNCHER_REPO:-https://github.com/local-inference-lab/blackwell-llm-docker.git}"
export LAUNCHER_REF="${LAUNCHER_REF:-build/gilded-gnosis-v20-canonical-release-20260722}"
export LAUNCHER_COMMIT="${LAUNCHER_COMMIT:-cf9a0f1e04ad9f029bccd3c46caa1ed3f49528ec}"
export VLLM_REQUIRED_LAUNCHERS="serve-gilded-gnosis.sh serve-fathomless-firmament.sh serve-glm52-v16.sh serve-glm52-v18.sh serve-glm52-v19.sh serve-glm52-hybrid-v17.sh serve-glm52-hybrid-v18.sh serve-glm52-hybrid-v19.sh"

export CUTLASS_REF="${CUTLASS_REF:-e6233cbac5d7c7a865c19c91cd684ceece19513c}"
export CUTLASS_COMMIT="${CUTLASS_COMMIT:-e6233cbac5d7c7a865c19c91cd684ceece19513c}"
export CUTLASS_DSL_VERSION="${CUTLASS_DSL_VERSION:-4.6.0}"
export TOKENSPEED_MLA_VERSION="${TOKENSPEED_MLA_VERSION:-0.1.8}"
export TVM_FFI_VERSION="${TVM_FFI_VERSION:-0.1.10}"
export TRITON_KERNELS_REF=
export TRITON_KERNELS_COMMIT=

export INSTANTTENSOR_REPO="${INSTANTTENSOR_REPO:-https://github.com/scitix/InstantTensor.git}"
export INSTANTTENSOR_REF="${INSTANTTENSOR_REF:-85e7c5f5539d9c006ee0c26bc1b5233c65251b6b}"
export INSTANTTENSOR_COMMIT="${INSTANTTENSOR_COMMIT:-85e7c5f5539d9c006ee0c26bc1b5233c65251b6b}"
export HUMMING_KERNELS_SPEC="${HUMMING_KERNELS_SPEC:-humming-kernels[cu13]==0.1.10}"
export VLLM_RUNTIME_EXTRA_PACKAGES="${VLLM_RUNTIME_EXTRA_PACKAGES:-nvtx==0.2.15 PyNvVideoCodec==2.0.4 nccl4py==0.3.1}"

requested_push="${PUSH_IMAGE:-0}"
export PUSH_IMAGE=0

./build-vllm-sparkinfer-cu132.sh "$@"

labels="$(docker image inspect "${IMAGE}" --format '{{json .Config.Labels}}')"
jq -e --arg value "${VLLM_COMMIT}" '."local-inference.vllm.commit" == $value' <<<"${labels}" >/dev/null
jq -e --arg value "${SPARKINFER_COMMIT}" '."local-inference.sparkinfer.commit" == $value' <<<"${labels}" >/dev/null
jq -e --arg value "${FLASHINFER_COMMIT}" '."local-inference.flashinfer.commit" == $value' <<<"${labels}" >/dev/null
jq -e --arg value "${CUTLASS_DSL_VERSION}" '."local-inference.cutlass_dsl.version" == $value' <<<"${labels}" >/dev/null
jq -e '."local-inference.vllm.patch_file" == "" and ."local-inference.vllm.patch_url" == ""' <<<"${labels}" >/dev/null

cache_fingerprint="$(jq -r '."local-inference.cache.fingerprint"' <<<"${labels}")"
[[ "${cache_fingerprint}" =~ ^vllm3e731bc043-b12x1a88b389a8-[0-9a-f]{16}$ ]]

image_env="$(docker image inspect "${IMAGE}" --format '{{range .Config.Env}}{{println .}}{{end}}')"
grep -Fxq "XDG_CACHE_HOME=/cache/jit/${cache_fingerprint}" <<<"${image_env}"
grep -Fxq "VLLM_CACHE_ROOT=/cache/jit/${cache_fingerprint}/vllm" <<<"${image_env}"
grep -Fxq "SPARKINFER_COMPILE_CACHE_DIR=/cache/jit/${cache_fingerprint}/sparkinfer/compile" <<<"${image_env}"

docker run --rm --gpus device=0 -i --entrypoint /opt/venv/bin/python "${IMAGE}" - <<'PY'
import importlib.metadata as md
import inspect

import torch
from sparkinfer.attention.sparse_mla._scratch import SPARKINFERSparseMLAScratchCaps
from sparkinfer.comm.pcie import DcpAllToAllPool
from sparkinfer.gemm import bmm, can_implement_bmm, prewarm_bmm
from sparkinfer.moe.fused_moe import _impl as fused_moe_impl
from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator
from vllm.model_executor.layers.attention import mla_attention
from vllm.model_executor.layers.attention.mla_attention import MLAAttention
from vllm.v1.attention.backends.mla.b12x_mla_sparse import B12xMLASparseImpl
from vllm.v1.attention.ops.common import cp_lse_ag_out_rs

assert md.version("sparkinfer") == "1.0.1"
assert md.version("nvidia-cutlass-dsl") == "4.6.0"
assert torch.__version__.startswith("2.12.0+cu132")
assert torch.version.cuda == "13.2"
assert fused_moe_impl._dynamic_kernel_intermediate_size(352, "w4a8_mx") == 384
assert callable(bmm) and callable(can_implement_bmm) and callable(prewarm_bmm)
assert "head_major_output" in inspect.signature(cp_lse_ag_out_rs).parameters
assert hasattr(CudaCommunicator, "reduce_scatter_head_major")
assert "head_major_output=True" in inspect.getsource(MLAAttention.forward_impl)
assert "ensure_cublas_tail_padding" not in inspect.getsource(MLAAttention._v_up_proj)
assert "out" in inspect.signature(DcpAllToAllPool.lse_reduce_scatter).parameters
assert hasattr(mla_attention, "_preallocate_absorbed_mla_weights")
assert not hasattr(mla_attention, "_release_b12x_mxfp8_kv_b_proj")
spec_source = inspect.getsource(B12xMLASparseImpl)
assert 'os.getenv("VLLM_B12X_MLA_SPEC_EXTEND_AS_DECODE", "auto")' in spec_source
assert "attn_metadata.is_spec_decode" in spec_source
caps = SPARKINFERSparseMLAScratchCaps(
    device="cuda:0",
    dtype=torch.bfloat16,
    num_q_heads=8,
    max_q_rows=6,
    max_width=2048,
    head_dim=576,
    v_head_dim=512,
    head_major_output=True,
)
assert caps.head_major_output is True
assert __import__("pathlib").Path(
    "/opt/vllm/kv-scales/glm52-nvfp4-nf3-hybrid_mla_outer_scales_v1.json"
).is_file()
print("v20 final runtime contracts: PASS")
PY

dry_run_file="/tmp/gilded-gnosis-v20-final-dcp1.txt"
docker run --rm --entrypoint /usr/local/bin/serve-gilded-gnosis.sh \
  -e DRY_RUN=1 \
  -e MODEL_FAMILY=glm52 \
  -e MODEL=/model \
  -e TP=8 \
  -e DCP=1 \
  -e MTP=0 \
  -e MOE_MODE=a16 \
  -e MAX_NUM_SEQS=1 \
  -e GRAPH=6 \
  "${IMAGE}" | tee "${dry_run_file}"

grep -q -- '--max-num-seqs 1' "${dry_run_file}"
grep -q -- '--max-cudagraph-capture-size 6' "${dry_run_file}"
grep -q -- '--load-format instanttensor' "${dry_run_file}"
grep -q -- '--max-model-len 262144' "${dry_run_file}"
grep -q -- '--gpu-memory-utilization 0.96' "${dry_run_file}"

mxfp8_dry_run_file="/tmp/gilded-gnosis-v20-final-mxfp8.txt"
docker run --rm --entrypoint /usr/local/bin/serve-gilded-gnosis.sh \
  -e DRY_RUN=1 \
  -e MODEL_FAMILY=glm52 \
  -e MODEL=/model \
  -e TP=8 \
  -e DCP=1 \
  -e MTP=0 \
  -e MOE_MODE=a16 \
  -e ONLINE_QUANT=mxfp8 \
  "${IMAGE}" | tee "${mxfp8_dry_run_file}"

grep -Fq 'QUANTIZATION_CONFIG_JSON=\{\"linear\":\{\"weight\":\"mxfp8\"\}\}' "${mxfp8_dry_run_file}"
if grep -q 'kv_b_proj' "${mxfp8_dry_run_file}"; then
  printf 'Unexpected kv_b_proj ignore in the default MXFP8 preset\n' >&2
  exit 1
fi

mxfp8_ignore_json='{"linear":{"weight":"mxfp8"},"ignore":["re:.*[.]q_a_proj$","re:.*[.]kv_a_proj_with_mqa$"]}'
mxfp8_ignore_dry_run_file="/tmp/gilded-gnosis-v20-final-mxfp8-ignore.txt"
docker run --rm --entrypoint /usr/local/bin/serve-gilded-gnosis.sh \
  -e DRY_RUN=1 \
  -e MODEL_FAMILY=glm52 \
  -e MODEL=/model \
  -e TP=8 \
  -e DCP=1 \
  -e MTP=0 \
  -e MOE_MODE=a16 \
  -e ONLINE_QUANT=mxfp8 \
  -e QUANTIZATION_CONFIG_JSON="${mxfp8_ignore_json}" \
  "${IMAGE}" | tee "${mxfp8_ignore_dry_run_file}"

grep -Fq 're:.\*\[.\]q_a_proj\$' "${mxfp8_ignore_dry_run_file}"
grep -Fq 're:.\*\[.\]kv_a_proj_with_mqa\$' "${mxfp8_ignore_dry_run_file}"

if [[ "${requested_push}" == "1" ]]; then
  docker push "${IMAGE}"
fi

printf 'Image: %s\n' "${IMAGE}"
