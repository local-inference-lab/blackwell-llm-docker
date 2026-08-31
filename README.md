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

# Build the Kimi-K3 TP16/DCP16 image from pinned Heraldic Harbinger and B12X
# source compositions. The image defaults to the DSpark launcher and also
# contains target-only and DFlash launchers.
./build-kimi-k3-hh-runtime.sh

# Build the Kimi-K3 TP16/DCP16 runtime pinned to the
# dev/infernal-invocation-derived integration source and B12X composition.
./build-kimi-k3-infernal-invocation-cu133-torch213.sh

# Build the source-neutral Kimi-K3 dependency foundation once. Subsequent
# source-only vLLM, B12X, and LMCache changes reuse its compiled dependencies.
./build-kimi-k3-runtime-foundation.sh

# Build the qualified Kimi-K3 source composition from the pinned foundation.
./build-kimi-k3-upstream-aligned-runtime.sh

# Build the DeepSeek-V4-Flash-0731 runtime from the source-locked Infernal
# Invocation, B12X, and LMCache integration trees on CUDA 13.3/PyTorch 2.13.
./build-deepseek-infernal-invocation-cu133-torch213.sh

# Build the unified GLM-5.2 and DS4/DSpark v16 image from immutable vLLM,
# B12X, FlashInfer, DeepGEMM, CUTLASS, InstantTensor, and NCCL commits.
./build-fathomless-firmament-v16-cu132.sh

# Build the current unified v17 image with NF3/NVFP4-KV support and the
# validated TP4/TP6/TP8 sparse-MLA DCP prefill workspace paths.
./build-fathomless-firmament-v17-cu132.sh

# Build a new GG v20 candidate. This always resolves the current clean
# dev/gilded-gnosis and SparkInfer master heads and composes both pinned PR
# manifests from scratch.
./build-gilded-gnosis-v20-final-cu132.sh
```

### Kimi-K3 source-locked production runtime

Status: **qualified**. `build-kimi-k3-upstream-aligned-runtime.sh` compiles
vLLM, B12X, and LMCache from the source locks under
`patches/releases/kimi-k3-upstream-aligned-20260822/`. It reuses the generic
CUDA 13.3/PyTorch 2.13 dependencies in the source-neutral image produced by
`build-kimi-k3-runtime-foundation.sh`. Updating a source lock therefore does
not rebuild FlashInfer, DeepGEMM, ExLlamaV3, InstantTensor, or the Rust
toolchain.

Published images:

```text
Foundation:
voipmonitor/vllm:kimi-k3-runtime-foundation-cu133-torch213-fi1ac6942-rust195-20260822-r1
sha256:03b67e53dda73c3fa317d4cb529ad38a220c51c7365ee8d54c16e5063fcc54e2

Production runtime:
voipmonitor/vllm:kimi-k3-upstream-aligned-dspark-nativekv-vllme755f87-b12x2d466e3-cu133-torch213-20260822-r36
sha256:c41bf15095b2316c7335d305115ad26bab14ec4234f3109b1d1ebb807895a3ea
```

The production image contains target-only, Inferact DSpark, and modal-labs
DFlash entrypoints. The default entrypoint enables DSpark, native vLLM host KV
offload, vision, TP16/DCP16, and a 4,096-token scheduler chunk. Qualification
against the preceding source-identical image measured 55.801 target-only,
122.695 DSpark, and 155.069 DFlash decode tokens per second. All deterministic
output hashes and speculative acceptance rates were identical. The complete
source identities, cache capacities, and measurement receipts are recorded in
`validation/kimi-k3-upstream-aligned-r36-20260822.json`.

### Kimi-K3 Heraldic Harbinger runtime

`build-kimi-k3-hh-runtime.sh` applies hash-verified integration patches to a
pinned `dev/heraldic-harbinger` vLLM commit and a pinned B12X `master` commit.
The included changes are restricted to the Kimi-K3 runtime:

- vLLM PRs #242 and #278;
- B12X PRs #124, #138, and #139.

The committed integration locks under
`patches/releases/kimi-k3-hh-runtime-r1/` record the base commits, PR heads,
resulting Git trees, and patch hashes. A publication build requires a clean
repository and uses those locks by default. Set `KIMI_K3_COMPOSITION=compose`
only to regenerate candidate locks from the manifests under `manifests/`.

Published image:

```text
voipmonitor/vllm:kimi-k3-hh-vllm138eccd-b12x7617005-cu132-20260811-r2
sha256:7ca3d4ffc6d5812984b3164e1ec821104bfa5ae85a5467aea9e86e7462943092
```

The image contains three entrypoints:

| Runtime profile | Entrypoint |
|---|---|
| Full MXFP4 target without speculation | `/usr/local/bin/serve-kimi-k3-nospec` |
| Full MXFP4 target with BF16 DSpark K7 | `/usr/local/bin/serve-kimi-k3-dspark` |
| Full MXFP4 target with online-MXFP8 DFlash K7 | `/usr/local/bin/serve-kimi-k3-dflash` |

Each profile must use a separate persistent `/cache/jit` host directory. The
vLLM, Triton, CuTe DSL, and B12X caches are source-fingerprinted inside that
directory. Profile separation also prevents incompatible CUDA-graph and
generated-kernel state from being reused across target-only, DSpark, and
DFlash processes. The machine-readable qualification receipt is
`validation/kimi-k3-hh-runtime-20260811.json`.

### Kimi-K3 dev/infernal-invocation runtime

Status: **implemented**. The build script
`build-kimi-k3-infernal-invocation-cu133-torch213.sh` composes the vLLM Git tree
recorded by
`patches/releases/kimi-k3-infernal-invocation-runtime-r1/vllm/integration.lock.json`
with the B12X Git tree recorded by
`patches/releases/kimi-k3-hh-runtime-r1/b12x/integration.lock.json`. The image
uses CUDA 13.3, PyTorch 2.13.0, NCCL 2.31.2, and InstantTensor 0.1.9.

The image contains these serving interfaces:

| Runtime profile | Entrypoint | Draft checkpoint |
|---|---|---|
| Full MXFP4 target without speculation | `/usr/local/bin/serve-kimi-k3-nospec` | none |
| Full MXFP4 target with seven-token DSpark | `/usr/local/bin/serve-kimi-k3-dspark` | `Inferact/Kimi-K3-DSpark` |
| Full MXFP4 target with seven-token DFlash | `/usr/local/bin/serve-kimi-k3-dflash` | `modal-labs/Kimi-K3-DFlash` |

DSpark is the image entrypoint. Start that profile with:

```bash
docker run --rm --name kimi-k3-infernal-dspark \
  --gpus all --ipc=host --network=host --shm-size=32g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /root/.cache/huggingface:/root/.cache/huggingface:ro \
  -v kimi-k3-infernal-dspark-jit:/cache/jit \
  voipmonitor/vllm:kimi-k3-infernal-vllmde04f08-b12x2e6092a-cu133-torch213-20260812-r1
```

Use a distinct `/cache/jit` volume for each entrypoint. The three profiles have
different graph shapes and generated kernels; sharing a writable JIT directory
between them is unsupported.

The source-locked production image also includes the language-only QSRT-K2
launcher. Build its immutable vLLM/B12X/LMCache source bundle with
`./build-kimi-k3-qsrt-tp16-runtime.sh`. For eight 96 GiB GPUs, use
`/usr/local/bin/serve-kimi-k3-qsrt-dspark-tp8`. It reserves four worker batch
slots for DSpark depth 3 while keeping the target scheduler aligned to Kimi's
1,536-token recurrent block (`1540` total / `1536` target / two sequences).
The generic QSRT launcher validates this reserve before starting and verifies
that the active B12X package supports both QSRT atoms-v2 and inactive W4A16
routes. See `docs/kimi-k3-qsrt-tp8-scheduler-safety.md` for the failure analysis,
source identities, compatibility limits, and qualification evidence.

### Infernal Invocation CUDA 13.3 runtime for DeepSeek-V4-Flash and GLM-5.2

Status: **qualified for the DeepSeek-V4-Flash profile and source-qualified for
the GLM-5.2 profiles listed below**.
The build script `build-deepseek-infernal-invocation-cu133-torch213.sh`
composes immutable vLLM, B12X, and LMCache trees. The runtime uses CUDA 13.3,
PyTorch 2.13.0, NCCL 2.31.2, FlashInfer 0.6.18, CUTLASS DSL 4.6.2, XGrammar
0.2.5, InstantTensor 0.1.9, and retains `pytest 8.4.1` for focused runtime
validation.

```text
voipmonitor/vllm:infernal-invocation-vllmf0fa1ce-b12x75787c7-fi1ac6942-cu133-torch213-20260818-r18
sha256:414ec7d0d28358cfd8af0697f330f5c8acbb80e4dc4e5ba69c9fd5b5855ea804
```

| Model profile | Status | Configuration | Compose file |
|---|---|---|---|
| DeepSeek-V4-Flash-0731 | qualified | TP2/DCP1, fixed probabilistic DSpark K5, B12X W4A8, FP8 compressed MLA KV | `examples/docker-compose-ds4-infernal-invocation-cu133-r18.yml` |
| GLM-5.2 NVFP4 | source-qualified | TP8/DCP1/MTP3, B12X W4A16, online MXFP8, FP8 MLA KV | `examples/docker-compose-glm52-nvfp4-infernal-invocation-r18.yml` |
| GLM-5.2 EXL3 R7 3.5bpw | source-qualified | TP4/DCP1/MTP3, mixed Trellis K3/K4/K5 experts, online K6, NVFP4 DS-MLA KV | `examples/docker-compose-glm52-exl3-infernal-invocation-r18.yml` |

Start one profile with its committed Compose specification:

```bash
docker compose \
  -f examples/docker-compose-ds4-infernal-invocation-cu133-r18.yml \
  up -d
```

The DeepSeek-V4-Flash runtime sizes C128A graph metadata from physical graph
capacity, limits sparse top-k to active rows, and performs GPU-native sparse
metadata updates without per-step host scalar extraction. Hybrid external-KV
load failure restores cache-group block tables independently. MRV2
logits-processing state is reset for each request.

The GLM entrypoint accepts borrowed InstantTensor buffers. Its deferred
layerwise online quantizer owns tensors that outlive the iterator step, so
`INSTANTTENSOR_COPY=0` cannot expose retained weights to staging-buffer reuse.
The EXL3 profile preserves per-projection MCG K3/K4/K5 payloads and encodes
eligible BF16 dense projections as persistent K6 payloads.

The qualified DeepSeek-V4-Flash run used TP2/DCP1 and fixed probabilistic
DSpark K5. Target decode, DSpark draft decode, and DFlash context-KV execution
captured FULL CUDA graphs. The 20-second C1 gate measured 164.46 aggregate
tok/s and 64.40 target steps/s. Infernal Invocation r16 measured 64.03 target
steps/s with the same scheduler configuration, so target execution did not
regress. A structured-output soak completed 160 of 160 requests at concurrency
8. Native filesystem KV restored all 695 objects after process restart.
LMCache restored all 94 disk chunks and 24,064 cached tokens after a complete
LMCache and vLLM restart.

GLM-5.2 EXL3 and NVFP4 remain source-qualified for this image. The EXL3
checkpoint declares rank slices for TP4 and correctly rejects TP2; only GPUs 6
and 7 were available during r18 qualification. Infernal Invocation r17 retains
the measured TP4/DCP1/MTP3 GLM-5.2 EXL3 serving receipt.

The exact image identity, source trees, runtime packages, configurations,
measurements, and qualification limits are recorded in
`validation/infernal-invocation-r18-remote-gpu.json`. A profile is qualified
only on the image and geometry named by its receipt.

### Clean GG release composition

New Gilded Gnosis images must not use an earlier `build/*` integration branch
as their source. `build-gilded-gnosis-v20-final-cu132.sh` reads the manifests
under `manifests/vllm/` and `manifests/b12x/`, resolves the current
`dev/gilded-gnosis` and B12X `master` heads, verifies every pinned PR
head, and creates fresh integration patches and lockfiles. The build stops if
either base advances, a PR changes, or a PR conflicts. The Dockerfile
independently verifies that applying each patch produces its locked Git tree
and records both bases, PR heads, trees, patch hashes, and lock hashes in image
labels.

The old r4 source can only be selected explicitly for reproducibility:

```bash
VLLM_RELEASE_COMPOSITION=reproduce-r4 \
  ./build-gilded-gnosis-v20-final-cu132.sh
```

The validated r5 image is reproducible from its archived, hash-verified source
locks and integration patches even after either upstream branch advances:

```bash
VLLM_RELEASE_COMPOSITION=reproduce-r5 \
  ./build-gilded-gnosis-v20-final-cu132.sh
```

The r6 LMCache runtime uses the same immutable vLLM and SparkInfer trees as r5:

```bash
VLLM_RELEASE_COMPOSITION=reproduce-r6 \
  ./build-gilded-gnosis-v20-final-cu132.sh
```

The r7 candidate keeps those model stacks, but builds LMCache directly from
the merged `local-inference-lab/LMCache` release commit instead of applying a
container-local source patch:

```bash
VLLM_RELEASE_COMPOSITION=reproduce-r7 \
  ./build-gilded-gnosis-v20-final-cu132.sh
```

This historical mode still verifies the pinned base commits, patch hashes, and
resulting Git trees. It only skips the normal requirement that the current
remote branch heads remain equal to the archived base commits.

Current clean r8 candidates also build XGrammar `0.2.5` from the immutable
`v0.2.5` source commit. The image build verifies GLM `tool_choice=required`
semantics: at least one tool call is required, while multiple calls and normal
termination after a call remain valid. XGrammar caps Transformers below 5 for
tokenizer regressions in other model families; this GLM image removes only that
package-metadata cap and validates the pinned GLM tokenizer with its
Transformers 5 runtime. The override is recorded in the image labels.
Historical r4-r7 reproduction modes keep the XGrammar version and metadata
supplied by their original vLLM requirements. Reproduce the exact r8 source
composition with:

```bash
VLLM_RELEASE_COMPOSITION=reproduce-r8 \
  ./build-gilded-gnosis-v20-final-cu132.sh
```

The r9 release retains the r8 runtime and adds three independently verifiable
changes:

- optional dynamic per-token NVFP4 MLA KV scaling from the paired vLLM #189
  and SparkInfer #86 ABI change;
- exact adaptive sparse-indexer folding from SparkInfer #87, which keeps the
  two-level reduction when it fits the configured workspace budget and falls
  back to exact streaming carry otherwise;
- `pytest==8.4.1` in the final `/opt/venv`, so focused tests can run against a
  deployed image without copying the repository test tree into the image.

The default cache format is unchanged. Dynamic NVFP4 scaling is enabled only
when all three settings below are selected, and it must not be combined with a
static outer-scale file:

```bash
KV_CACHE_DTYPE=nvfp4_ds_mla \
KV_FP8_ROPE=1 \
VLLM_NVFP4_MLA_DYNAMIC_SCALE=1 \
VLLM_NVFP4_MLA_SCALES_FILE= \
  docker compose up -d
```

Adaptive folding defaults to `auto` with a 256 MiB temporary-workspace budget.
Override it only for diagnosis with
`SPARKINFER_INDEXER_TWO_LEVEL_FOLD=0|1` or change the budget with
`SPARKINFER_INDEXER_TWO_LEVEL_FOLD_MAX_MIB`.

Reproduce the exact r9 source composition from the archived lock files and
hash-verified integration patches with:

```bash
VLLM_RELEASE_COMPOSITION=reproduce-r9 \
  ./build-gilded-gnosis-v20-final-cu132.sh
```

Published r9 image:

```text
voipmonitor/vllm:gilded-gnosis-v20-vllm34f26c2-side7739a-fi801d57a-cu132-20260728-r9
```

The r15 release adds the `DeepSeek-V4-Flash-0731` DSpark serving profile and
keeps the standard-checkpoint MTP modes separate. It composes vLLM #212, #213,
#214 and SparkInfer #106 over current clean GG/master sources. The paired cache
changes accept exact and padded compressed-MLA pages without copying, while the
V2 warmup change keeps FlashInfer autotune enabled before KV initialization.

Build or reproduce the exact release with:

```bash
./build-gilded-gnosis-v20-final-cu132.sh

VLLM_RELEASE_COMPOSITION=reproduce-r15 \
  ./build-gilded-gnosis-v20-final-cu132.sh
```

Published r15 image:

```text
voipmonitor/vllm:gilded-gnosis-v20-vllm0bc48c5-sieec30ff-fi801d57a-cu132-20260731-r15
```

Start the pinned 0731 checkpoint in the measured fixed-K7 mode:

```bash
GPUS=0,1 docker compose -f examples/docker-compose-ds4-v20-r15.yml up -d
```

Set `DSPARK_DEPTH_MODE=dynamic` for load-aware draft depth, or `MODE=dspark-mtp0`
for a no-speculation baseline on the same 0731 checkpoint. `MODE=mtp2|mtp3`
selects the historical standard checkpoint because 0731 does not provide the
standard MTP serving contract.

The r16 release adds native CPU KV offload for DS4 without the pinned-host
power-of-two allocation restriction and preserves SWA, MTP, and shared-prefix
replay boundaries. It also changes the DS4 Compose profile to fixed K5, which
was faster in sustained decode and more reliable than K7 in local validation.

Build or reproduce the exact release with:

```bash
./build-gilded-gnosis-v20-final-cu132.sh

VLLM_RELEASE_COMPOSITION=reproduce-r16 \
  ./build-gilded-gnosis-v20-final-cu132.sh
```

r16 release image:

```text
voipmonitor/vllm:gilded-gnosis-v20-vllm1e9c9c3-sieec30ff-fi801d57a-cu132-20260731-r16
sha256:48518e91cf87dd0c0483c76ff86e81dfc0f46de7e364b46f7a82c481ce08188f
```

Start DSpark K5 on two GPUs:

```bash
GPUS=0,1 docker compose -f examples/docker-compose-ds4-v20-r16.yml up -d
```

Native offload is opt-in. `KV_OFFLOADING_SIZE` is the total host capacity in
GiB across all TP ranks; decimal, non-power-of-two values are supported:

```bash
KV_OFFLOADING_SIZE=48.5 GPUS=0,1 \
  docker compose -f examples/docker-compose-ds4-v20-r16.yml up -d
```

The exact r16 image passed TP2 K5 E2E with and without native offload. The
no-offload baseline reached 220.6 tok/s; a repeated 5.5 GiB offload run reached
222.9 tok/s. A 70k/80k/100k prefix sequence transferred 5.22 GB from GPU to
CPU, then restored 635.5 MB and 69,888 prefix tokens from CPU on replay.

The r20 release is a strict r19 superset. It adds the qualified block-32
SparkInfer prefill path for mixed Trellis B3/B4 experts and an opt-in online
EXL3 K6 path for eligible dense matrices. Online K6 artifacts are written as
per-rank safetensors under `/cache/exl3-online`; the cache key includes the
checkpoint, encoder, TP geometry, tensor shape, and quantization parameters.
Mount `/cache` persistently so later starts load the artifacts instead of
encoding them again.

```bash
VLLM_RELEASE_COMPOSITION=reproduce-r20 \
  ./build-gilded-gnosis-v20-final-cu132.sh
```

Published r20 image:

```text
voipmonitor/vllm:gilded-gnosis-v20-vllm72c35f1-si2b9bf2a-fi801d57a-cu132-20260802-r20
```

Enable online K6 only with an EXL3 checkpoint and backend:

```bash
MODEL_FAMILY=glm52-exl3 \
QUANTIZATION=exl3 \
ONLINE_QUANT=exl3-b6 \
  docker compose up -d
```

On the TP4 release validation host, a fully empty EXL3 and JIT cache produced
1,644 EXL3 files (11.90 GB) and reached `/health` in 827 seconds. Restarting
the identical container with the same cache mount produced 1,644 cache hits,
no encodes, an identical EXL3 cache manifest, and reached `/health` in 113.94
seconds. MTP0 CC1 decode was 53.23/53.08 tok/s; uncached prefill was 3,635.67
tok/s at 8k and 3,512.62 tok/s at 64k.

The r24 DS4 release hardens compressed MLA workspace sizing, InstantTensor
host registration, native KV-offload shared-region lifetime, replay/IPC graph
state, and the MHC compile boundary. It also includes capacity-bounded LMCache
storage. Reproduce the exact source trees with:

```bash
VLLM_RELEASE_COMPOSITION=reproduce-r24 \
  ./build-gilded-gnosis-v20-final-cu132.sh
```

Published r24 image:

```text
voipmonitor/vllm:gilded-gnosis-v20-vllmf5981f1-si2b9bf2a-fi801d57a-cu132-20260803-r24
manifest: sha256:64b94299abdd3bcf5bb5050ca91b378f9ee4e0b0eff4748375b95352371d7cb2
```

The DS4 deployment recipe is
`examples/docker-compose-ds4-v20-r24.yml`. Native KV offload is disabled by
default. When enabled, its shared host region is unlinked after every worker
has mapped it, so it is reclaimed automatically when the final worker exits.
LMCache's disk-capacity fix is included, but LMCache remains experimental for
DS4 because long-context output correctness is not yet closed.

The r27 DS4 release fixes a deterministic native L2-offload corruption found
after r24. Tiering workers and the delayed scheduler now map the same named L1
backing, while the scheduler skips a redundant full-region prefault. It also
aligns the 0731 low/high/max reasoning contract and tool placement with the
official template and adds an InstantTensor runtime-pinned fallback for hosts
where segmented host registration is unavailable.

Reproduce the exact source trees with:

```bash
VLLM_RELEASE_COMPOSITION=reproduce-r27 \
  ./build-gilded-gnosis-v20-final-cu132.sh
```

Published r27 image:

```text
voipmonitor/vllm:gilded-gnosis-v20-vllm966d57c-sibbbdccc-fi801d57a-cu132-20260803-r27
Docker manifest: sha256:2605fda01797f33239af4c95ec7449505fe57d9b9de9687792f5b8273d3201a7
```

Use `examples/docker-compose-ds4-v20-r27.yml`. Native L2 offload remains
opt-in, and host `/dev/shm` must fit the configured L1 region plus runtime
headroom. Fixed K5 remains the release default. Batch-wide BOS output under
deep concurrent DSpark workloads is still tracked in rtx6kpro issue #53;
LMCache long-context correctness is also not yet a qualified DS4 path.

The r27 Compose exposes `SHM_SIZE` and the helper's `EXTRA_VLLM_ARGS` so a
tiered native L1/L2 configuration can be supplied without editing the file.
Keep the transfer-config JSON compact because the helper treats this variable
as an argument list.

The r25 GLM release adds the runtime-dynamic mixed-Trellis partition from
[SparkInfer #117](https://github.com/local-inference-lab/sparkinfer/pull/117).
The 3.36 bpw checkpoint uses 206 K3 + 50 K4 experts in layer 3 and 160 K3 +
96 K4 afterward; those counts are now launch arguments instead of stale
compiled-kernel state. The change preserves one compiled kernel across both
partitions and keeps Michel Belleau's original #114 authorship.

```bash
VLLM_RELEASE_COMPOSITION=reproduce-r25 \
  ./build-gilded-gnosis-v20-final-cu132.sh
```

Published r25 image:

```text
voipmonitor/vllm:gilded-gnosis-v20-vllmf5981f1-si978cdb3-fi801d57a-cu132-20260803-r25
Docker manifest: sha256:042936fd8d9e4c2aa579ab9b736dd0a2faf2678c6ba36bf4dfce7db566c6fd11
```

Remote validation pins the actual image ID, not only its source trees. Python
wheel archives are not bit-reproducible, so rebuilding after validation can
produce a different image ID. Publish an already validated local image without
rebuilding it by using the checked receipt path:

```bash
VLLM_RELEASE_COMPOSITION=reproduce-r25 \
  LOCAL_GPU_VALIDATION=0 \
  USE_EXISTING_VALIDATED_IMAGE=1 \
  REMOTE_GPU_VALIDATION_RECEIPT=validation/gilded-gnosis-v20-r25-remote-gpu.json \
  PUSH_IMAGE=1 \
  ./build-gilded-gnosis-v20-final-cu132.sh
```

The measured TP4/DCP4/MTP3 recipe is
`examples/docker-compose-glm52-exl3-v20-r25.yml`. The exact image passed
source/runtime contracts, cold model startup, coherent output, CUDA graphs,
CC1/CC8 decode, and 8k/64k prefill on GPUs 4-7 of the root-port validation
host. Verifier throughput was 33.70 steps/s at CC1 and 111.78 steps/s at CC8,
within 1.04% and 0.20% of the source-overlay reference. Prefill reached 2,407
tok/s at 8k and 2,018 tok/s at 64k; KV capacity was 770,048 tokens. The exact
remote validation receipt is
`validation/gilded-gnosis-v20-r25-remote-gpu.json`.

The r26 GLM release fixes the TP4/DCP4 prefill policy. TP4/DCP4 has one query
partition, so the exact owner-exchange path only adds row routing and an output
all-gather; it cannot remove duplicate query work. The helper now selects query
split, full CKV gather, two indexer shards, and depth-1 CKV prefetch, while
leaving owner merge disabled for this topology. Explicit environment overrides
remain available for diagnosis.

```bash
VLLM_RELEASE_COMPOSITION=reproduce-r26 \
  ./build-gilded-gnosis-v20-final-cu132.sh
```

Published r26 image:

```text
voipmonitor/vllm:gilded-gnosis-v20-vllmf5981f1-sibbbdccc-fi801d57a-cu132-20260803-r26
Docker manifest: sha256:c7a202cf3ccd155973a151235acb9677aa98f61765372f839bb0c193ff594ec4
```

The exact image passed TP4/DCP4/MTP3 startup, automatic lossless PCIe
calibration, CUDA graph capture, coherent 256-token decode, and uncached
8k/64k prefill on physical GPUs 4-7 of the root-port validation host. The
3.36 bpw checkpoint reached 3,125 tok/s at 8k and 2,988 tok/s at 64k, up
29.8% and 48.1% over the matched r25 release gate. The calibrated policy was
`query-split=1`, `CKV-gather=1`, `owner-merge=0`, `indexer-shards=2`, and
`prefetch-depth=1`; no lossy transport was enabled. Use
`examples/docker-compose-glm52-exl3-v20-r26.yml`. The exact receipt is
`validation/gilded-gnosis-v20-r26-remote-gpu.json`.

The r28 GLM release adds the shared-H EXL3 checkpoint contract and completes
runtime-dynamic mixed K3/K4 execution. The 3.42 bpw checkpoint uses 206/50 in
layer 3 and 148/108 in layers 4-77. SparkInfer receives those counts and the
per-tier broadcast-H flags as runtime metadata, so one compiled kernel remains
valid across layers without expanding a shared rotation row per expert.

```bash
VLLM_RELEASE_COMPOSITION=reproduce-r28 \
  ./build-gilded-gnosis-v20-final-cu132.sh
```

Published r28 image:

```text
voipmonitor/vllm:gilded-gnosis-v20-vllme1e9426-si200c1db-fi801d57a-cu132-20260804-r28
Docker manifest: sha256:501e10e79b4bc854237804d215e454c531ac9c2d354a8fa1a93e450fe7ba6ce0
```

The checkpoint's physical shared-H layout saves 672.36 MiB/GPU at MTP0 and
681.33 MiB/GPU at MTP3. The default online K6 profile loaded at 79.47 GiB/GPU
in the KLD run, 4.61 GiB below the no-online profile. On the TP4 root-port
validation host, DCP1/MTP0 reached 53.29 tok/s decode and 3,587/3,386 tok/s
uncached prefill at 8k/64k. DCP1/MTP3 reached 113.40 tok/s. DCP4/MTP3 reached
93.76 tok/s at CC1, 240.17 aggregate tok/s at CC4, 336.35 at CC8, and
3,489/3,337 tok/s prefill at 8k/64k. DCP4 batch checks passed 24/24 at c8 and
32/32 at c16.

Teacher-forced KLD uses the same 2,047 positions and BF16 reference logits for
both profiles. Lower is better:

| Profile | KV format | Mean KLD |
|---|---|---:|
| Checkpoint only, no online quantization | FP8, matched to reference | 0.074145973 |
| Release default, online K6 | NVFP4 MLA | 0.108828284 |

Most of the headline delta is the runtime KV format, not online K6: with the
same NVFP4 KV format, K6 adds only 0.000856839 mean KLD over the no-online
run. Use `examples/docker-compose-glm52-exl3-v20-r28.yml`; the exact receipt
is `validation/gilded-gnosis-v20-r28-remote-gpu.json`.

The r29 release completes the SparkInfer-to-B12X package rename, fixes the
compressed-MLA verifier capacity contract, and captures DSpark context-KV work
in dedicated FULL CUDA graphs. The split planner and scratch allocator now
receive the physical verifier row capacity: 384 rows for MNS64 fixed K5 and
512 for fixed K7. This replaces the old fixed 256-row assumption that could
select an invalid graph contract above C24.

An identical TP2/DCP1 fixed-K5 C64 gate reproduced the r28 failure at 140/192
correct responses and passed r29 at 192/192. Fixed K7 and dynamic K7 also
passed the original 192/192 gate. The final clean image passed a second K5
C64 gate at 192/192 and a K7 C32 smoke gate at 96/96.

The context-KV graph path maps actual verifier rows to the smallest captured
bucket and makes padded rows inert. It keeps profiling, dummy runs,
out-of-envelope shapes, and non-DSpark execution on the eager path. A matched
60-second A/B on the same TP2 GPUs measured:

| Fixed K5 test | Eager context-KV | FULL context-KV | Result |
|---|---:|---:|---:|
| CC1 server decode | 182.82 tok/s | 190.68 tok/s | +4.3% |
| CC32 aggregate median | 1,257.73 tok/s | 1,253.69 tok/s | -0.3% (noise) |
| Clean release image | - | 190.66 / 1,259.85 tok/s | CC1 / CC32 |

At CC32 the trace reduced DSpark proposal time from 2.408 to 1.510 ms per
step and removed 108 eager launches over four profiled steps, but target/MoE
work already saturates the server, so end-to-end throughput stays flat. Graph
memory increased from 2.14 to 2.17 GiB and the matched short-context KV pool
lost only 74 tokens. Prefill remains PIECEWISE; host metadata, input
preparation, rejection sampling, and output bookkeeping remain eager.

Fixed K5 remains the default. K7 can be faster in predictable code phases,
but did not establish a repeatable mixed-workload advantage after prompt,
reasoning-effort, and acceptance accounting were matched.

The same image also includes the one-shard DCP query-split topology. A final
TP4/DCP4/MTP0 test on the 3.42 bpw GLM checkpoint confirmed one indexer shard,
lossless full CKV gather, and 3,352/3,393 tok/s uncached prefill at 8k/64k.

Reproduce the exact source composition with:

```bash
VLLM_RELEASE_COMPOSITION=reproduce-r29 \
  ./build-gilded-gnosis-v20-final-cu132.sh
```

Validated r29 image:

```text
voipmonitor/vllm:gilded-gnosis-v20-vllm55db472-b12x6bc35fd-fi801d57a-cu132-20260807-r29
Local validated image ID: sha256:3b6057d254da1a4115dc31f742d0206820b30ac0e411fa1860ea918cda90af89
Registry digest: sha256:3441df47253919d20deb5f57a75e47142f9e0eec8a2ceb2c6f4898ebc9680e16
```

Use `examples/docker-compose-ds4-v20-r29.yml`. The immutable source receipt is
`validation/gilded-gnosis-v20-r29-remote-gpu.json`; its vLLM and B12X trees
are `55db47246a3365ca0a8f908f83a0b0ea06f0856a` and
`6bc35fdb76b6f9d11601009fe413720b461fb444`.

The r30 release rebases the same qualified DS4 paths on current GG
`e2666d9a65` and fixes native tiered KV-offload finalization. The scheduler
previously released `TieringOffloadingManager` request state before the next
step built the EOS-completed final store. Under a deep queue this caused a
deterministic `KeyError` in `prepare_store()` and terminated EngineCore. r30
builds the final store first, retains state until submitted transfers finish,
and then finalizes the request.

r30 also updates the semantic B12X CUDA-graph channel integration to GG's
current graph-manager API. Target, DSpark proposal, and DFlash context-KV
FULL graph families were all captured in the final image. Prefill remains
PIECEWISE, and profiling, dummy, unsupported-shape, and non-DSpark paths keep
their explicit fallbacks.

The final TP2/DCP1/K5 validation used the reported 500k-context native
`TieringOffloadingSpec` configuration with a 16 GiB L1 and filesystem L2.
All 1,792 concurrent stress requests completed successfully. The strongest
wave held 218 requests in deferred lookup, completed 1,184 store operations
(38.93 GB), and the replay loaded 408 batches (18.46 GB). The final queues
were zero, EngineCore remained healthy, and a post-stress request returned
exactly `42`. The release Compose was then requalified without privileged
container access: another 192/192 requests passed, including 128 filesystem
loads (5.79 GB), and the final response remained exactly `42`.

Reproduce the exact source composition with:

```bash
VLLM_RELEASE_COMPOSITION=reproduce-r30 \
  ./build-gilded-gnosis-v20-final-cu132.sh
```

Validated r30 image:

```text
voipmonitor/vllm:gilded-gnosis-v20-vllm20ed7f9-b12x6bc35fd-fi801d57a-cu132-20260807-r30
Local validated image ID: sha256:b0d592d0f4395de3936034075caf29d883d3e17e75839ff5ea800c41d6a2548c
Registry digest: sha256:a8a7a05fe8a8599b2ff86cfa2814e4a3b1ec1a28fb154b496f891115f28a6c84
```

Use `examples/docker-compose-ds4-v20-r30.yml`. The immutable source and live
qualification receipt is
`validation/gilded-gnosis-v20-r30-remote-gpu.json`; its vLLM and B12X trees
are `20ed7f98b1ab2c0e6f008a6ad34306fd3b72b33f` and
`6bc35fdb76b6f9d11601009fe413720b461fb444`.

The r31 release retains every qualified r30 path. It builds FlashInfer from the
exact qualified `main` + PR #4393 source commit and adds its PCIe IPC all-reduce
as the automatic TP2 backend. TP4 and larger keep B12X because matched kernel
and end-to-end tests favored B12X there. It also prewarms mixed-Trellis target
and native-MTP route packing before KV sizing, preventing first-request scratch
allocations from reducing the usable KV pool.

The final candidate additionally guards dynamic DSpark query lengths in the
FlashInfer persistent wrapper, fixes compiled packed-UE8M0 scale emission and
int32-packed MLA activation dtypes, registers the launcher/backend runtime
controls, serializes the DSpark draft backend under vLLM's canonical field, and
deduplicates lockstep MLA block cleanup in native KV offload.

The clean release manifest contains all required unmerged changes: vLLM #145,
#188, #213, #214, #217, #218, #228, #229, #230, #234, #235, #245, #248,
#251 through #256; B12X #125 and #126; and LMCache #7 through #17. The required
parts of superseded vLLM #197 and #250 are preserved in #254 and #228. Every
open GG PR is classified as included or reviewed-excluded and pinned by its
full head SHA. The build fails if a head moves, a new PR is unclassified, or
the clean replay over GG/B12X/LMCache conflicts.

Native L2 offload is also environment-only in r31. `KV_OFFLOADING_SIZE` sets
the host-memory L1 size, while `NATIVE_L2_PATH` and `NATIVE_L2_GB` enable a
bounded filesystem L2. The helper constructs the connector JSON and defaults
`PYTHONHASHSEED=0`; no `EXTRA_VLLM_ARGS` JSON is needed:

```bash
KV_OFFLOADING_SIZE=32 NATIVE_L2_PATH=/native-l2 NATIVE_L2_GB=1024 \
  docker compose -f examples/docker-compose-ds4-v20-r31.yml up -d
```

Reproduce the exact source composition with:

```bash
VLLM_RELEASE_COMPOSITION=reproduce-r31 \
  ./build-gilded-gnosis-v20-final-cu132.sh
```

Validated r31 image:

```text
voipmonitor/vllm:gilded-gnosis-v20-vllmfa13d33-b12xacee6e5-fi1ac6942-cu132-20260807-r31
Local validated image ID: sha256:b162476b0b3432096e9dd1d0b0d8c825ba71bf33635423c511d9bac533b12a9f
Registry digest: sha256:3230c25ff95f8678a8eeb52a463f0d3b9f96f6ad550418cc51ea12177a55b41c
```

Use `examples/docker-compose-ds4-v20-r31.yml`. `ALLREDUCE_MODE=auto` selects
`flashinfer-ipc` at TP2 and `b12x` at TP4 or larger; either backend can be
forced explicitly for diagnostics.

The release was validated on physical GPUs 4-7 of `192.168.0.69`, where the
GPUs are attached through CPU root ports rather than the PCIe switches in the
local 16-GPU server. All performance comparisons below use earlier artifacts
from that same remote host and GPU set; results from the switched server are
not used as a reference.

| Target-only profile | C1 tok/s | C32 tok/s | Prefill 8k tok/s | Prefill 64k tok/s |
|---|---:|---:|---:|---:|
| r31 TP2, FlashInfer PCIe IPC | 126.8 | 1,139.5 | 13,366 | 12,669 |
| r31 TP2, B12X | 129.9 | 1,135.7 | 14,197 | 13,421 |
| r31 TP4, B12X | 148.4 | 1,511.0 | 16,360 | 15,511 |
| Previous TP4 B12X, same host | 144.5 | 1,499.2 | 15,406 | 14,721 |

The TP4/K5 row-contract gate captured FULL target, DSpark proposal, and DFlash
context-KV graphs through the required 384-row envelope. Sustained C64 decode
was 2,540.5 tok/s. The long-context Estonia gate passed 64/64 requests with no
output-cap hits, and its 134,217-token prefill scout reached 13,352 tok/s.
Compiled packed-UE8M0 scales matched eager output bitwise at both aligned and
unaligned batch sizes. Native L2 was verified across a full engine restart: an
identical 32k prompt restored 303,586,560 bytes from filesystem storage in
0.415 seconds after process-local GPU/L1 state had been discarded.

The immutable source and live qualification receipt is
`validation/gilded-gnosis-v20-r31-remote-gpu.json`; its vLLM and B12X trees are
`fa13d334a2962756f9f7e9b562deb85387359f42` and
`acee6e504209068bd0cbb01cb2b98966bddcf042`.

The r33 release keeps the qualified r31 vLLM tree and adds four isolated B12X
changes. PR #133 adds topology-scoped fused all-reduce paths: TP8 owner
reduction is automatic, while TP2/TP4 remote push remains opt-in because
matched end-to-end runs showed gains and losses rather than a consistent win.
PR #135 preserves the dense GEMM API contract for block-FP8 callers. PR #136
restores capture-safe K6 small-M dispatch with an explicit SM120 capability
gate. PR #137 realigns mixed-Trellis execution with the QSRT ABI.

Reproduce the exact source composition with:

```bash
VLLM_RELEASE_COMPOSITION=reproduce-r33 \
  ./build-gilded-gnosis-v20-final-cu132.sh
```

Validated r33 image:

```text
voipmonitor/vllm:gilded-gnosis-v20-vllmfa13d33-b12x06db0f4-fi1ac6942-cu132-20260809-r33
Local validated image ID: sha256:60944a4ea1fbb2d1f35d7972f685d8fb0b91e77dd5aeca1dcafa3bcc29846d12
Registry digest: sha256:fdde59fed7f9fc12f9fd5ef1b3b3ea8d5097bf10ebad54b348497102c3a83f82
```

Validation used only GPUs 4-7 on `192.168.0.69`. The GLM row used
`willfalco/GLM-5.2-EXL3-TR3-3.36bpw`, TP4/DCP1/MTP3, online EXL3 K6, and
NVFP4 DS-MLA KV. The standard `llm_decode_bench` prompt produced 116.2 and
112.0 tok/s, for a two-run median of 114.1 tok/s versus the historical 113.4
tok/s result. A separate synthetic MTP-friendly prompt reached about 144
tok/s; that number is not used as the standard headline or regression gate.

| DS4 fixed probabilistic K5 | C1 tok/s | C4 tok/s | C8 tok/s | Prefill 8k tok/s |
|---|---:|---:|---:|---:|
| TP2, FlashInfer PCIe IPC auto | 180.6 | 397.1 | 580.7 | 12,849 |
| TP4, B12X auto | 247.0 | 541.9 | 804.5 | not repeated |

Both DS4 rows used the pinned `DeepSeek-V4-Flash-0731` revision, B12X W4A8,
FP8 DS-MLA KV, InstantTensor BUFFERED, and FULL+PIECEWISE CUDA graphs. The TP2
correctness request returned exactly `42`. The immutable validation receipt is
`validation/gilded-gnosis-v20-r33-remote-gpu.json`; its vLLM and B12X trees are
`fa13d334a2962756f9f7e9b562deb85387359f42` and
`06db0f4b27dbd19eb934da0da27eff7a7c49d8c4`.

### Gilded Gnosis r34 GLM-5.2 R7 mixed-Trellis profile

Status: **qualified**.

The r34 GLM-5.2 profile serves
`brandonmusic/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78@9ab9579774cc432df91567a36f6e9e863e0d4c9f`.
Routed experts execute directly from checkpoint-native K3/K4/K5 Trellis
payloads. Shared experts in layers 3-77 remain BF16 in the checkpoint and are
encoded once into merged K6 gate-up and down projections. The encoded matrices
use a content-addressed persistent cache, so warm starts do not repeat the
encoding.

```text
voipmonitor/vllm:gilded-gnosis-v20-vllm4d006a4-b12xcd3ce19-fi1ac6942-cu132-20260810-r34
Local image ID: sha256:0ff4b1de4e950cf48dd0405562908a2f81597f4524698c0291ac2c40514ae17e
Registry digest: sha256:820181fbbc975cd5291c411cda9771d58fecee1636d916f508f47230df20592b
```

Use `examples/docker-compose-glm52-r7-v20-r34.yml`. Its qualified profile is
TP4/DCP1, A16 B12X MoE, B12X sparse MLA, online EXL3 K6, NVFP4 DS-MLA KV,
InstantTensor BUFFERED, `MAX_NUM_SEQS=8`, graph cap 32, and a 65,536-token
model limit. MTP3 requires `GPU_MEMORY_UTILIZATION=0.98`; that setting exposes
75,072 KV tokens on 96 GB GPUs and leaves enough memory for graph capture.

Validation used physical GPUs 4-7 on `192.168.0.69`, where each GPU is
attached through a CPU root port. Results from the PCIe-switched 16-GPU host
are not used in the comparison.

| Profile | C1 tok/s | C4 tok/s | C8 tok/s | Prefill 8k tok/s | KV tokens |
|---|---:|---:|---:|---:|---:|
| MTP0, GMU 0.97 | 53.8 | 171.3 | 283.1 | 3,253 | 82,816 |
| MTP3, GMU 0.98 | 121.2 | 297.7 | 436.2 | 3,239 | 75,072 |

The MTP3 run accepted 15,307 of 23,391 draft tokens (65.44%). FULL decode
graphs covered every configured scheduler size; target verification and all
three draft forwards remained graph-captured. Six focused B12X GPU tests,
112 focused vLLM tests, startup, deterministic correctness, and the installed
runtime-contract verifier passed.

A paired 2,047-position full-vocabulary quality gate used FP8 KV and three
repeats per candidate. Separate shared K6 measured mean KLD 0.064467; BF16
source plus merged runtime K6 measured 0.065339. The 0.000872 difference was
smaller than the observed run variation. These KLD values do not isolate the
NVFP4 DS-MLA cache format selected by the Compose profile.

Reproduce the immutable source composition with:

```bash
VLLM_RELEASE_COMPOSITION=reproduce-r34 \
  ./build-gilded-gnosis-v20-final-cu132.sh
```

The machine-readable qualification record is
`validation/gilded-gnosis-v20-r34-remote-gpu.json`. Its vLLM and B12X trees
are `4d006a43928cdee01306691a766542c1e9bebb59` and
`cd3ce190f0f1917402cdfd5773724267cc9a63f8`.

The archived Fathomless Firmament v17 image installs
`/usr/local/bin/serve-fathomless-firmament.sh`, which dispatches to the GLM or
DS4 helper through `MODEL_FAMILY`. Start either model with a minimal
environment-only Compose file and override only the serving choices you need:

```text
voipmonitor/vllm:fathomless-firmament-v17-vllm05f50ae-b12x1377d5f-fi801d57a-cu132-20260715
sha256:9b6f1ab6db4d3a7b7b786481eb32abe82e86d185648d62c3ac1cfa6d72a55e47
```

```bash
MODE=dspark BACKEND=lucifer-cutlass TP_SIZE=2 GPUS=0,1 \
  docker compose -f examples/docker-compose-ds4-v10.yml up -d

MTP=0 DCP=1 MOE_MODE=a16 ONLINE_QUANT=mxfp8 \
  docker compose -f examples/docker-compose-glm52-v17.yml up -d
```

Supported modes are `mtp0`, `mtp2`, `mtp3`, and `dspark`. Supported backend
profiles are `b12x-a16`, `b12x-a8`, `b12x-a8-dglin`, `lucifer-default`, and
`lucifer-cutlass`; the helper derives the CUDA graph cap from
`MAX_NUM_SEQS`. The GLM helper likewise derives `GRAPH=4*MAX_NUM_SEQS` unless
explicitly overridden. Both helpers default to InstantTensor with the
page-cache-aware `BUFFERED` backend. The GLM v17 Compose also defaults
`DCP_PREFILL_WORKSPACE=auto`, which enables the optimized eager prefill path
only for its validated TP/DCP topology list.

### Automatic PCIe calibration

The v20 GLM helper runs a lossless PCIe preflight before the first model load
for each GPU order, TP/DCP geometry, CPU/NUMA placement, image fingerprint,
NCCL configuration, and probe revision. It measures the real collectives and
caches four independent decisions under
`${XDG_CACHE_HOME}/pcie-calibration`:

- `VLLM_B12X_MLA_CKV_PREFETCH_DEPTH`
- `VLLM_DCP_QUERY_SPLIT`
- `VLLM_DCP_QUERY_SPLIT_MIN_CONTEXT_TOKENS`
- `VLLM_PCIE_DMA_MIN_BYTES`, including `off` when NCCL wins the full ladder

Later starts use the cache. Explicit values always take precedence. Set
`PCIE_CALIBRATION=force` to remeasure or `PCIE_CALIBRATION=off` to retain the
conservative static policy. FP8, INT8, and MXFP8 DMA wire modes remain
explicit choices; selecting one through `F8_DMA` never enables a compressed
mode through calibration.

GPU order is resolved as `GPUS`, then an existing `CUDA_VISIBLE_DEVICES`, then
the launcher default. This ensures the probe measures the same ordered devices
that vLLM serves on, including Compose files that leave `GPUS` empty. A cold
probe may compile kernels and has a 600-second startup limit; override it with
`PCIE_CALIBRATION_TIMEOUT` when required.

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
