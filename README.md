# Blackwell LLM Docker

Docker images for LLM inference on NVIDIA Blackwell GPUs (SM120).

## Images

| Image | Dockerfile | Stack |
|-------|-----------|-------|
| `voipmonitor/sglang:cu130` | `Dockerfile.sglang-cu130` | CUDA 13.0, torch 2.11 stable cu130, FlashInfer source (PR #2913), SGLang + b12x + PCIe allreduce |
| `voipmonitor/sglang:cu132` | `Dockerfile.sglang-cu132` | CUDA 13.2, torch 2.12 from source, FlashInfer source (PR #2913), SGLang + b12x |
| `voipmonitor/vllm:cu130` | `Dockerfile.vllm-cu130` | CUDA 13.0, torch 2.11 stable cu130, FlashInfer source (PR #2913), vLLM + cherry-picks |
| `voipmonitor/vllm:glm-kimi-cu132` | `Dockerfile.glm-kimi-cu132` | Clean CUDA 13.2.1, PyTorch 2.12 cu132 wheels, patched NCCL 2.30.4, FlashInfer, B12X, canonical GLM/Kimi vLLM |

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

# Clean GLM/Kimi vLLM cu132
IMAGE=voipmonitor/vllm:glm-kimi-cu132-20260518 ./build-glm-kimi-cu132.sh
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

## GLM/Kimi CUDA 13.2 Image

`Dockerfile.glm-kimi-cu132` is intentionally self-contained: it starts from
`nvidia/cuda:13.2.1-cudnn-devel-ubuntu24.04` and does not inherit from any
previous `voipmonitor/vllm` image. It installs PyTorch `2.12.0+cu132` from the
official PyTorch wheel index, builds patched NCCL `2.30.4` from
`local-inference-lab/nccl-canonical`, then builds FlashInfer, B12X and the
canonical GLM/Kimi vLLM branch.

The final image defaults to `/usr/local/bin/run-kimi26-vllm`; GLM is available
through `/usr/local/bin/run-glm51-vllm`.
