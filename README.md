# Blackwell LLM Docker

Docker images for LLM inference on NVIDIA Blackwell GPUs (SM120).

## Images

| Image | Dockerfile | Stack |
|-------|-----------|-------|
| `voipmonitor/sglang:cu130` | `Dockerfile.sglang-cu130` | CUDA 13.0, torch 2.11 stable cu130, FlashInfer source (PR #2913), SGLang + b12x + PCIe allreduce |
| `voipmonitor/sglang:cu132` | `Dockerfile.sglang-cu132` | CUDA 13.2, torch 2.12 from source, FlashInfer source (PR #2913), SGLang + b12x |
| `voipmonitor/vllm:cu130` | `Dockerfile.vllm-cu130` | CUDA 13.0, torch 2.11 stable cu130, FlashInfer source (PR #2913), vLLM + cherry-picks |
| `voipmonitor/vllm:vllm-b12x-cu132` | `Dockerfile.vllm-b12x-cu132` | Clean CUDA 13.2.1, PyTorch 2.12 cu132 wheels, patched NCCL 2.30.4, FlashInfer, DeepGEMM, B12X, vLLM |
| `voipmonitor/vllm:lucifer` | `Dockerfile.vllm-b12x-cu132` | Lucifer DS4 Flash/CUTLASS vLLM branch on the same CUDA 13.2.1 base, FlashInfer, DeepGEMM, and Triton kernels source hook |

Base image for cu132 (torch + FlashInfer compiled from source):

| Image | Dockerfile | Stack |
|-------|-----------|-------|
| `voipmonitor/torch:cu132` | `Dockerfile.torch-cu132` | CUDA 13.2, torch 2.12 from source (no pip nvidia-*), FlashInfer from source |

## Quick start

```bash
# Qwen3.5-397B NVFP4 on 4x Blackwell GPUs
docker compose -f examples/docker-compose-qwen35.yml up -d

# GLM-5 NVFP4 on 8x Blackwell GPUs
docker compose -f examples/docker-compose-glm5.yml up -d
```

See `examples/` for full docker-compose files with hardware requirements and configuration options.

## Run

### With model profile

```bash
docker run --gpus all --ipc=host --shm-size=8g \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v jit-cache:/cache/jit -p 5000:5000 \
  -e MODEL_PROFILE=qwen35-b12x \
  voipmonitor/sglang:cu130
```

Available profiles: `qwen35-b12x`, `glm5-nvfp4` (see `profiles/` directory).

### Direct command

```bash
docker run --gpus all --ipc=host --shm-size=8g \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v jit-cache:/cache/jit -p 5000:5000 \
  voipmonitor/sglang:cu130 \
  python -m sglang.launch_server --model-path <model> --tp 8 --host 0.0.0.0 --port 5000
```

### vLLM

```bash
docker run --gpus all --ipc=host --shm-size=8g \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 5000:5000 \
  voipmonitor/vllm:cu130 \
  --model <model> --tensor-parallel-size 4 --host 0.0.0.0 --port 5000
```

## Build

```bash
# SGLang cu130
docker build --build-arg CACHEBUST=$(date +%s) -f Dockerfile.sglang-cu130 -t voipmonitor/sglang:cu130 .

# SGLang cu132 (requires torch base first)
docker build -f Dockerfile.torch-cu132 -t voipmonitor/torch:cu132 .
docker build --build-arg CACHEBUST=$(date +%s) -f Dockerfile.sglang-cu132 -t voipmonitor/sglang:cu132 .

# vLLM cu130
docker build --build-arg CACHEBUST=$(date +%s) -f Dockerfile.vllm-cu130 -t voipmonitor/vllm:cu130 .

# Clean vLLM+B12X cu132. This builds the reusable system/build base images
# first, then builds the final vLLM image from those base images.
IMAGE=voipmonitor/vllm:vllm-b12x-cu132 ./build-vllm-b12x-cu132.sh

# Reproduce the pushed black-benediction PR11 image exactly.
./build-black-benediction-b12xpr11-cu132.sh

# Build the Lucifer DS4 Flash/CUTLASS image from local-inference-lab/vllm:lucifer.
./build-lucifer-cu132.sh

# Build the unified GLM-5.2 and DS4/DSpark v16 image from immutable vLLM,
# B12X, FlashInfer, DeepGEMM, CUTLASS, InstantTensor, and NCCL commits.
./build-fathomless-firmament-v16-cu132.sh
```

The unified image installs `/usr/local/bin/serve-fathomless-firmament.sh`, which
dispatches to the GLM or DS4 helper through `MODEL_FAMILY`. Start either model
with a minimal environment-only Compose file and override only the serving
choices you need:

```bash
MODE=dspark BACKEND=lucifer-cutlass TP_SIZE=2 GPUS=0,1 \
  docker compose -f examples/docker-compose-ds4-v10.yml up -d

MTP=0 DCP=1 MOE_MODE=a16 ONLINE_QUANT=mxfp8 \
  docker compose -f examples/docker-compose-glm52-v16.yml up -d
```

Supported modes are `mtp0`, `mtp2`, `mtp3`, and `dspark`. Supported backend
profiles are `b12x-a16`, `b12x-a8`, `b12x-a8-dglin`, `lucifer-default`, and
`lucifer-cutlass`; the helper derives the CUDA graph cap from
`MAX_NUM_SEQS`. The GLM helper likewise derives `GRAPH=4*MAX_NUM_SEQS` unless
explicitly overridden. Both helpers default to InstantTensor with the
page-cache-aware `BUFFERED` backend.

### Current vLLM+B12X CUDA 13.2 base image

The vLLM+B12X build uses two reusable base images:

- `voipmonitor/vllm:vllm-b12x-cu132-system-base`: CUDA 13.2.1 cuDNN devel base, cuBLAS 13.4.1, cuDNN 9.22, Python 3.12, build/runtime OS packages, and patched NCCL 2.30.4.
- `voipmonitor/vllm:vllm-b12x-cu132-build-base`: the system base plus `/opt/venv` with PyTorch `2.12.0+cu132`, torchvision `0.27.0+cu132`, CUDA tile, and CUTLASS DSL `4.5.2`.

The final image is built `FROM` the system base and copies the completed vLLM
venv from the build stages. This keeps the final image from carrying a stale
base venv while avoiding repeated apt/PyTorch downloads on normal source-only
rebases. The historical 2026-06-08 black-benediction build reused the already
published `glm-kimi-cu132-system-base-20260608` and
`glm-kimi-cu132-build-base-20260608` tags; the preset below preserves that exact
input stack.

```bash
git clone https://github.com/local-inference-lab/blackwell-llm-docker.git
cd blackwell-llm-docker

SYSTEM_BASE_IMAGE=voipmonitor/vllm:vllm-b12x-cu132-system-base \
BUILD_BASE_IMAGE_TAG=voipmonitor/vllm:vllm-b12x-cu132-build-base \
IMAGE=voipmonitor/vllm:vllm-b12x-cu132 \
./build-vllm-b12x-cu132.sh

# Push the reusable base images when publishing a new stack baseline.
SYSTEM_BASE_IMAGE=voipmonitor/vllm:vllm-b12x-cu132-system-base \
BUILD_BASE_IMAGE_TAG=voipmonitor/vllm:vllm-b12x-cu132-build-base \
IMAGE=voipmonitor/vllm:vllm-b12x-cu132 \
PUSH_BASE_IMAGE=1 \
./build-vllm-b12x-cu132.sh

# Exact black-benediction PR11 image from 2026-06-08.
./build-black-benediction-b12xpr11-cu132.sh

# Lucifer DS4 Flash/CUTLASS image. This reuses the same cu132 system/build bases
# and builds vLLM from local-inference-lab/vllm branch `lucifer`, which contains
# the rebased Lucifer SM120 sparse MLA patch and CUTLASS MoE fix from
# procr1337/llm-bench. It also enables the Triton kernels source hook used by
# that stack.
./build-lucifer-cu132.sh
```

Useful sanity check after the build:

```bash
docker run --rm voipmonitor/vllm:vllm-b12x-cu132-system-base bash -lc '
python --version
nvcc --version | tail -n 1
strings /opt/libnccl-local-inference.so.2.30.4 | grep "NCCL version 2.30.4 compiled with CUDA 13.2"
dpkg-query -W \
  "cuda-compat-13-2" \
  "cublas-cuda-13" \
  "libcublas13-cuda-13" \
  "libcublas13-dev-cuda-13" \
  "libcudnn9-cuda-13" \
  "libcudnn9-dev-cuda-13" \
  "libcudnn9-headers-cuda-13"
'

docker run --rm voipmonitor/vllm:vllm-b12x-cu132-build-base bash -lc '
python - <<PY
import torch
import cutlass
print(torch.__version__, torch.version.cuda)
print(cutlass.__file__)
PY
'
```

## Hardware

- NVIDIA RTX PRO 6000 Blackwell Server Edition (SM120) or compatible
- CUDA driver 575+
- 96 GB VRAM per GPU

## Key features

- **FlashInfer from source** with PR #2913 (GDC for SM120) — no prebuilt cubin/jit-cache that would override patched kernels
- **b12x backend** (lukealonso) — TP-only NVFP4 MoE/GEMM for SM120
- **PCIe allreduce** — custom allreduce for PCIe topologies (cu130 only)
- **nvidia-cublas pinned to 13.1** (cu130) — 13.3 causes illegal memory access on CUDA 13.0 toolkit
- **Model profiles** — preconfigured launch configs via `MODEL_PROFILE` env var
- **Adaptive speculative decoding** (PR #21599) — dynamically adjusts num_steps
- Pre-tuned Triton MoE configs for RTX PRO 6000 Blackwell

## vLLM+B12X CUDA 13.2 Image

`Dockerfile.vllm-b12x-cu132` is intentionally based on reusable base images that
are themselves built from `nvidia/cuda:13.2.1-cudnn-devel-ubuntu24.04`, not from
an older `voipmonitor/vllm` image. The system base keeps the CUDA toolkit on
13.2.1, overlays the latest CUDA 13 library packages currently used by this
image (`cuBLAS` 13.4.1, `cuDNN` 9.22, `cuda-compat-13-2` 595.71), and includes
patched NCCL `2.30.4` from `local-inference-lab/nccl-canonical`. The build base
adds PyTorch `2.12.0+cu132` from the official PyTorch wheel index and CUTLASS
DSL. The final image then builds FlashInfer, DeepGEMM, B12X and the selected
vLLM branch on top of those bases.

The final image defaults to `/usr/local/bin/run-kimi26-vllm`; GLM is available
through `/usr/local/bin/run-glm51-vllm`.
