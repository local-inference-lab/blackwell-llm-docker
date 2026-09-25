# Model runtime configuration

Status: **implemented** in the shared CUDA 13.4 image. Model profiles select
serving, memory and cache defaults through one interface. The image's CUDA/NCCL
bootstrap prepares the native runtime before launching the selected service.
Model-level test records are kept in `runtime/validation/` and the model wiki.

## One image, explicit model selection

For an image built from `tools/jovian_wheel_runtime/Dockerfile.runtime`:

```bash
docker run --rm --init --gpus '"device=0,1,2,3"' --network host --ipc host \
  -v model-cache:/root/.cache/huggingface -v runtime-cache:/cache \
  -e PROFILE=glm53-flash -e HARDWARE_PROFILE=rtx-pro-6000-pcie \
  -e TP=4 -e SPECULATOR=mtp -e MTP_DEPTH=3 -e PORT=8000 "$LIL_IMAGE"
```

Profiles are `glm53-flash`, `qwen38-flash-next`, `ds4-flash`, `ds4-vision`
and `ds41-flash`. Select GPUs to match TP. DS4.1's Engram access also needs
the generated Compose's memory-lock and io_uring permissions. `MODEL` is an
optional checkpoint override; it does not select another model architecture.
`--print-config` displays the resolved command without starting services.
With no profile or command the container prints help, not an implicit model.
Explicit commands such as `python`, `bash` or `vllm serve` retain the ABI
bootstrap and bypass profile defaults.

Profiles keep downloaded checkpoints and saved HF credentials in
`/root/.cache/huggingface` (`HF_HOME`), independently of the runtime-keyed JIT
cache under `/cache/jit`. Updating an image therefore does not move the model
cache. To use another location, mount that directory and set `-e HF_HOME=/path`.
`XDG_CACHE_HOME` controls the profile's JIT root; it does not relocate HF data.

## Named deployment settings

`PRESET` selects a data-only overlay from `presets.yaml`. Model defaults remain
separate from deployment-specific memory constraints. The Spark TP2 preset uses
the Spark checkpoint on two 96-GB RTX PRO GPUs; it does not select DGX Spark
hardware or replace GLM TP4 defaults.

```bash
docker run -d --name glm-spark-tp2 --init --gpus '"device=0,1"' \
  --network host --ipc host --shm-size 32g \
  --ulimit memlock=-1 --ulimit stack=67108864:67108864 \
  -v model-cache:/root/.cache/huggingface -v glm-spark-runtime:/cache \
  -e PRESET=glm53-spark-tp2 -e PORT=8000 "$LIL_IMAGE"
```

This selects TP2/DCP2, MTP3 with B12X draft experts, eight request slots, a
3,072-token prefill budget, 4,557 MiB fixed KV per rank (about 1.15M tokens)
and sparse full/piecewise captures through 32 verifier rows. To make room for
the KV cache, the input embedding table lives in pinned host RAM (0.59 GiB per
GPU, read row by row over PCIe), the vision tower runs in MXFP8 (0.24 GiB) and
the TP, DCP and EP groups share one NCCL communicator (74 MiB). Decode speed is
unchanged; long prefills are 1-3% slower. The allocator uses 12 MiB large
segments; NCCL uses two channels and 1 MiB buffers. No image-count limit is
imposed.

- Sixteen request slots: add `-e MAX_NUM_SEQS=16`. Each slot above eight takes
  64 MiB from the KV allocation for larger CUDA graphs and buffers, so 16 slots
  keep 4,045 MiB per rank (about 1.02M tokens). An explicit
  `KV_CACHE_MEMORY_BYTES` is used as given.
- Qualified worst case at 8 and 16 slots: 62K-token prompts, a 3840x2160 image,
  six-image requests and decoding streams at the same time, with at least
  0.24 GiB of GPU memory still free at the peak. A larger explicit KV size or
  more slots than 16 can run out of memory under such mixed load.
- BF16 vision tower or GPU-resident embeddings: add
  `-e VLLM_GLM53_VISION_MXFP8=0` or `-e VLLM_GLM53_EMBED_HOST=0` and lower
  `KV_CACHE_MEMORY_BYTES` by 0.24 GiB or 0.59 GiB respectively.
- CPU/disk LMCache: add `-e CACHE_MODE=lmcache`. GPU-only cache is the default.
  The connector's GPU buffers take 192 MiB from the preset KV allocation
  (about 1.0M tokens at eight slots, 0.86M at sixteen).
  Text recurrent checkpoints can be restored externally; vision requests
  recompute. LMCache retains fewer KV tokens than the GPU-only configuration.
- Smaller KV allocation: add `-e KV_CACHE_MEMORY_BYTES=3758096384` for 3.5 GiB.
- Change serving choices with the same `SPECULATOR`, `MTP_DEPTH`, `TP`, `DCP`,
  `MAX_NUM_SEQS`, and `MAX_NUM_BATCHED_TOKENS` controls as other profiles.
  GLM cache object size follows a changed prefill budget unless explicitly set.
  Automatic capture capacity follows request slots and effective proposal width.
- Native form: `lil-serve --preset glm53-spark-tp2 -- --port 8000`.
- Qwen TP2: `PRESET=qwen38-tp2`; CPU PLE placement and MTP defaults still come
  from the Qwen model profile. `PROFILE=qwen38-flash-next TP=2` is equivalent.
- GLM on three GPUs: `PRESET=glm53-tp3` (TP3 with expert parallelism, MTP3,
  eight request slots). MLA and KDA heads are padded to divide by three with
  zero weights, the vision tower runs data-parallel, and the large dense
  projections decode in FP8 (`-e VLLM_GLM53_FP8_DENSE=0` keeps them BF16).
  The external cache (`CACHE_MODE=lmcache`) is not available at TP3; the VRAM
  prefix cache is. The first start tunes FlashInfer MoE kernels and stores the
  result under `/cache`.

For Qwen, the `rtx-pro-6000-pcie` hardware profile keeps residual-mixing
projections replicated on each GPU (`VLLM_QWEN3_8_FLASH_NEXT_HC_TP=0`). This
avoids two cross-GPU gathers at each residual-mixing boundary. To test sharded
projections instead, pass `-e VLLM_QWEN3_8_FLASH_NEXT_HC_TP=1`; that option uses
less projection-weight memory but can slow PCIe decode. `HARDWARE_PROFILE=native`
leaves this choice to vLLM. Target weights and activation precision are unchanged.

For the Qwen checkpoint's 524,288-token context, set
`MAX_MODEL_LEN=524288`. The Qwen profile then supplies the YaRN factor-two
text configuration to both the target and MTP draft model. The ordinary
262,144-token setting keeps the checkpoint's original positional configuration.
An explicit `HF_OVERRIDES` JSON value takes precedence over the derived YaRN
settings; use it only when the selected checkpoint needs different RoPE values.

```bash
docker run -d --name qwen38-yarn512k --init --gpus '"device=0,1"' \
  --network host --ipc host --shm-size 32g \
  -v model-cache:/root/.cache/huggingface -v qwen38-runtime:/cache \
  -e PRESET=qwen38-tp2 -e MAX_MODEL_LEN=524288 -e PORT=8000 \
  ghcr.io/local-inference-lab/vllm:karmic-kraken-beta
```

Explicit settings, environment and native arguments take precedence over preset
defaults. A preset cannot be combined with a different architecture's profile.
Credentials and host GPU selection are never stored in presets.

## Ownership

Keep deployment policy in `blackwell-llm-docker`. A separate Docker repository
would introduce another version boundary between package composition,
entrypoints, CI, and model documentation without removing a configuration
owner. The LIL client can consume the versioned launch interface; it should
not maintain another copy of kernel settings.

The configuration has four independent identities:

| Artifact | Owns | Does not own |
|---|---|---|
| Model profile, `profiles/*.yaml` | Checkpoint name, model-specific precision, speculation, cache layout, native CLI defaults | GPU selection, clocks, library builds |
| Hardware profile, `hardware/*.yaml` | Explicitly selected communication and platform tuning | Model architecture or speculation method |
| Deployment preset, `presets.yaml` | Named checkpoint/TP/memory settings layered over a model and hardware profile | Credentials, host GPU IDs or kernel implementations |
| Image contract, `image-contract.json` | Runtime-lock digest, installed profile/launcher hashes, CUDA/NCCL bootstrap executable | Mutable benchmark results or credentials |

`hardware/native.yaml` leaves communication crossovers and NCCL channels to
native vLLM/B12X selection. `hardware/rtx-pro-6000-pcie.yaml` carries the
single-node workstation deployment settings, including 16 NCCL channels and
a 2 MiB buffer. These are not claimed to be optimal on unswitched workstations,
GB10, or multi-node systems. Neither hardware profile changes GPU clocks.

`schema.json` validates profile structure. `options.yaml` owns the mapping
between managed native CLI options, typed values, and environment aliases.
There is one common layer, one model layer, one hardware layer and an optional
deployment preset. A settings file and explicit user arguments are applied
after these layers. Presets cannot inherit from other presets.

## Model policies

The [generated parameter table](generated/parameters.md) is rendered from the
same profiles as the launch command. Its values describe configuration, not
performance measurements. Source references are recorded inside each profile.

- GLM: NVFP4 target; target MoE and dense backends explicitly B12X. Serving
  defaults to MTP with three proposals and the private NVFP4 draft vocabulary
  head, Marlin draft MoE, and B12X draft attention. Use `--mode off` for
  non-speculative serving. DFlash2 defaults to seven
  proposals from `local-inference-lab/GLM-5.3-Flash-DFlash2`, FLASH_ATTN draft
  attention and `auto` draft KV, retaining the MXFP8 checkpoint policy.
  Target KV defaults to FP8; `nvfp4_ds_mla` remains an explicit GLM option.
  KDA recurrent prefill explicitly defaults to B12X in every speculation mode,
  independently of sparse MLA and MoE selection. Use the native override
  `--additional-config.kda_prefill_backend flashkda` for FlashKDA; `triton` and
  `auto` remain explicit alternatives. The `auto` policy belongs to vLLM and
  does not mean B12X.
- DS4 text/Vision: fixed DSpark K5/K3, B12X W4A8 MoE, and native dense
  selection corresponding to `BACKEND=b12x-a8-dglin`. That source launcher
  **omits** `--linear-backend`; the profile preserves the omission rather than
  claiming it proves DeepGEMM dispatch. `--linear-backend deep_gemm` and
  `--linear-backend b12x` are explicit alternatives requiring dispatch and
  performance checks. Model choice is explicit: changing speculation mode
  never silently changes the checkpoint repository.
- DS4 text sets `VLLM_ENABLE_STARTUP_PLAN=1`. The first start stores the
  measured KV budget under the JIT cache. Later starts reuse it when the vLLM
  configuration, the device and the vLLM, PyTorch and CUDA versions are
  unchanged, and skip the CUDA-graph memory estimate (about 45-50 s on TP2).
  A plan is refused when free GPU memory shrank. The key covers CLI settings,
  not environment variables: after changing an environment variable that
  affects GPU memory, start once with `-e VLLM_ENABLE_STARTUP_PLAN=0` or set
  `KV_CACHE_MEMORY_BYTES`.
  The official text/Vision checkpoint and remote-code revisions follow the
  source launcher's pinned revisions. An explicit model override does not
  inherit another repository's revision; `MODEL_REVISION` and
  `MODEL_CODE_REVISION` remain operator controls.
- DS4.1: DSpark K7 with adaptive verification, sampled proposals, standard
  rejection, B12X target/draft attention and B12X MoE/dense. Engram table
  placement selects `ram` (default) or `disk` independently of general CPU
  offload. With RAM tables the container uses about 190 GiB of host memory
  at TP4 after startup and about 230 GiB under load; use
  `ENGRAM_TABLE_MEMORY=disk` on hosts with less.
  Main/SWA pages remain 256/128. Breakable prefill graphs remain disabled;
  the native graph configuration remains FULL_AND_PIECEWISE.
  JIT monitoring defaults to `warn`: a kernel missed during warmup may compile
  on first use without the monitor aborting the request. Qualification runs can
  require complete warmup with `-e JIT_MONITOR_MODE=error` or the native
  `--jit-monitor-mode error` argument.
- Qwen: TP1 by default; TP2 is an explicit override. Preserve CPU PLE tables,
  the BF16 target vocabulary head and private NVFP4 MTP copy. Image input is
  enabled by default; use `--language-model-only` to omit the vision encoder
  and reserve more memory for text serving. The 6,019-token
  scheduler budget is intentional preservation of the published Qwen recipe,
  not a replacement of the 4,096-token GLM/DeepSeek budget. Native generation
  configuration remains authoritative; benchmark temperature 1/top-p 0.95/
  top-k 20 is a request policy, not evidence that every checkpoint has those
  server defaults. Qwen attention selection is native; GDN, MoE, and dense
  kernel selection are explicitly B12X.

GLM and DeepSeek retain the source launchers' temperature 1/top-p 0.95
server defaults. Explicit generation configuration replaces these defaults;
request sampling remains authoritative. GLM retains `reasoning_effort=high`
and `clear_thinking=false`. No profile changes target weight or KV precision
to manufacture a speedup.

The GLM attention page, recurrent checkpoint spacing, and external transfer
object are different dimensions. The GPU profile uses 2,048-token target
pages. Aligned-256 requires both recurrent storage spacing and lookup policy:

```bash
python -m runtime.launcher --profile glm53-flash --print-config -- \
  --recurrent-checkpoint-policy aligned \
  --recurrent-page-size 256 --prefix-match-unit 256
```

`--prefix-match-unit 256` alone does not create stored recurrent checkpoints.
Request boundaries remain the GLM default. No image-limit-one override is
introduced for Vision; its native multimodal count and encoder-budget rules
remain separate from the text scheduler budget.

## Interface and precedence

Use the repository's isolated Python environment with `runtime/requirements.txt`
installed. Configuration inspection neither imports vLLM/torch nor downloads
weights, initializes CUDA, writes caches, or changes a running container:

```bash
python -m runtime.launcher --profile glm53-flash \
  --hardware rtx-pro-6000-pcie --print-config -- \
  --mode mtp --draft-tokens 3 --tensor-parallel-size 4

python -m runtime.launcher --profile ds41-flash \
  --hardware rtx-pro-6000-pcie --print-config \
  --env OMP_NUM_THREADS=1 -- --engram-table-memory disk
```

Precedence, highest first:

1. Explicit managed CLI options, including native `--tensor-parallel-size`.
2. Explicit `--env NAME=VALUE` for an environment alias.
3. `--settings FILE` → `options` mapping.
4. Settings-file `environment` mapping.
5. Process environment (`docker -e`, Compose environment, shell exports).
6. Selected hardware profile, model profile, then common defaults.

Settings example:

```yaml
options:
  tensor-parallel-size: 4
  decode-context-parallel-size: 1
  mode: dflash2
  draft-tokens: 7
  max-num-batched-tokens: 4096
environment:
  OMP_NUM_THREADS: "1"
```

`TP` and `TP_SIZE` are aliases, as are `DCP` and `DCP_SIZE`. Equal aliases
are accepted; conflicting aliases at the selected precedence level fail.
A valid higher-priority CLI override does not fail because an overridden
environment alias is invalid. Managed options are emitted once. Explicit
`OMP_NUM_THREADS=1` and capture cap 256 remain explicit values for DS4.1.
`MTP_DEPTH=3` works for GLM without setting a cache variable.

Native JSON roots replace the corresponding default object. Dotted CLI
fields then refine that object; e.g. `--compilation-config.custom_ops '["all"]'`.
JSON field names retain underscores. Unknown native CLI arguments are passed
as argv, never evaluated by a shell, and are identified as unvalidated native
options in the inspection output. Opaque `--config`, external connector flags,
and shell-text `EXTRA_VLLM_ARGS` are rejected rather than allowed to bypass
layout validation. Native vLLM remains the owner of its full option schema.

Legacy `BACKEND`, `MODE`, `SPEC_MODE`, and `DS4_*` graph/OMP aliases are not
silently ignored: they fail with migration guidance. The production wrappers
in published community images continue to accept them. The installed
`serve-*.sh` names select a profile; they are not byte-for-byte replicas of
every historical shell interface. Use `SPECULATOR`, `OMP_NUM_THREADS` and
native CLI options for this interface. `FAIRNESS_ENGINE=none` is supported;
`micro_slicing` is not.

`--print-config` includes effective arguments, settings, model-relevant
environment and each value's origin. It does not dump the process environment
or authentication tokens. API-key and credential-bearing JSON fields are
redacted without modifying the private values passed to the process.

## JIT identity and image packaging

Model settings cannot remain in global Docker `ENV`. A running process cannot
distinguish an explicit `docker -e OMP_NUM_THREADS=1` from the identical baked
value. Comparing values and deleting matches is not a solution.

`runtime.packaging` audits Docker inspection metadata and refuses a foundation
containing profile-owned variables. It records all installed runtime files and
the authenticated source-lock or wheel-lock digest. The build must inspect
the **same immutable foundation** it actually uses, and repeat the environment
audit on the final image. The file is a build receipt, not a cryptographic
attestation that an arbitrary supplied inspection JSON describes an image.

```bash
python -m runtime.packaging \
  --image-inspect foundation.inspect.json \
  --runtime-lock-sha256 RUNTIME_LOCK_SHA256
```

Inside the image build, `runtime/install.sh` installs `/usr/local/bin/lil-serve`
and the package in `/opt/lil/runtime`. Serving dependencies must already exist
in `/opt/venv`. The installer does not rebuild CUDA, torch, NCCL, FlashInfer,
B12X, vLLM, or LMCache. The image entrypoint is `lil-entrypoint`, which selects
profile serving or an explicit command. Dependency correction ownership is
documented in [DEPENDENCIES.md](DEPENDENCIES.md).

The CUDA 13.4 wheel image must preserve its CUDA/NCCL bootstrap:

```bash
bash runtime/install.sh /build/foundation.inspect.json RUNTIME_LOCK_SHA256 \
  /opt/venv/bin/lil-runtime-bootstrap
```

The bootstrap is recorded in the image contract and runs before the final
vLLM interpreter. It is not bypassed by the generated Compose entrypoint.
The CUDA 13.3 community runtime can omit that bootstrap when its linker
environment is already configured by the foundation. Do not copy CUDA paths
from the community runtime into a wheel profile.

The JIT namespace includes the runtime-lock identity and profile digest.
Explicit user cache paths are honored. An inspection without an image
contract shows `UNBOUND-RUNTIME`; execution refuses an unbound namespace.
GPU architecture and kernel-specific cache keys remain owned by the libraries.

The CUDA 13.4 recipe retains 67 pinned NGC foundation layers and one application
layer. It does not use a preceding community image as its base. The separate
flattened CUDA 13.3 community recipe has two layers; that is not this recipe's
layer count. Removing `ENV` in a child Dockerfile cannot clear inherited model
policy, so foundation and final image metadata are both audited. Profile edits
do not recompile native component wheels.

Foundation defaults that overlap with profile settings are declared in
`runtime/platform-environment.json`. The build verifies their values against
the pinned source image, removes those names from a cached OCI configuration,
and uses that exact configuration as the Dockerfile's named foundation context.
All 67 filesystem layers remain unchanged. The cache key includes the source
image ID and environment policy; repeated builds reuse the layout and the same
resource-limited BuildKit builder. `LIL_FOUNDATION_CACHE` can select its directory.

Profile serving applies these platform defaults before model, hardware, preset
and explicit user settings. An explicit `-e NCCL_NET_PLUGIN=spcx` therefore still
overrides a preset selecting `none`. Raw commands intentionally bypass profile
resolution; use the generated native environment or set their NCCL policy
explicitly. Both interfaces retain the CUDA/NCCL ABI bootstrap.

## Generated Compose and wiki material

```bash
python -m runtime.generate compose --profile ds41-flash
python -m runtime.generate compose --profile qwen38-flash-next
python -m runtime.generate compose --profile qwen38-flash-next --tp 2
python -m runtime.generate compose --profile glm53-flash --mode mtp
python -m runtime.generate compose --profile glm53-flash --mode dflash2
python -m runtime.generate table
```

Checked examples are under [generated/](generated/). They require an explicit
profile-enabled image via `LIL_IMAGE`; they intentionally do not point to a
published release that lacks the entrypoint. Compose selects device IDs and
model/profile/mode. It contains no duplicated B12X/NCCL/kernel policy. Changing
TP requires matching device reservations; the generator uses the profile's
default TP unless `--tp` selects another reservation count. An image-specific release exporter must supply the immutable image
identity and qualification receipts before these examples replace wiki recipes.

Wiki pages should retain human explanations and actual benchmark records.
Only marked parameter tables, invocation examples and compatibility lists
should be generated. Each measurement must identify image, profile digest,
hardware/topology, clocks, TP/DCP, speculation width, sampling, concurrency,
context and benchmark protocol. A generated table cannot confer qualification
on another CUDA build. Documentation-only publication stays in `rtx6kpro`;
configuration implementation stays here.

## External cache preservation and migration gates

`CACHE_MODE=vram` starts no external service. `CACHE_MODE=native` selects
GLM or Qwen native CPU KV offload. Qwen uses `SimpleCPUOffloadConnector`,
DCP1 and aligned checkpoints; it does not start an LMCache service or persist
cache across restarts. `NATIVE_KV_OFFLOADING_SIZE_GB` sets the total CPU cache
capacity across all TP ranks (default 64 GiB). For example:

```bash
docker run -d --name qwen38-native --gpus '"device=0,1"' --network host --ipc host \
  -v qwen-hf:/root/.cache/huggingface -v qwen-runtime:/cache \
  -e PROFILE=qwen38-flash-next -e TP=2 -e DCP=1 -e PORT=8000 \
  -e CACHE_MODE=native -e NATIVE_KV_OFFLOADING_SIZE_GB=32 \
  "$LIL_IMAGE" --mode mtp --draft-tokens 3
```

This MTP configuration requires the
[draft CuMem configuration fix](https://github.com/local-inference-lab/vllm/pull/834).
`CACHE_MODE=lmcache` constructs an explicit service plan
for GLM, Qwen, DS4 text/Vision, or DS4.1. `LMCACHE_MODE=ram|disk|off` selects
RAM, RAM with persistent disk storage, or VRAM-only caching.
The model and cache use distinct ports, defaulting to API port plus
10000/10001/10002 for cache RPC/HTTP/metrics. Overflow and collisions are errors.

| Contract | Required preserved behavior | Resolver disposition |
|---|---|---|
| GLM atomic recurrent cache | Target/recurrent/draft all-rank bundles, request/SYSTEM boundaries, identity checks and restart restore | Implemented for text; TP2 requires engine-driven request-boundary transfer. TP4/TP8 also accept aligned transfer. |
| DS4 engine-driven cache | Worker-owned pinned SHM, CPU-only service, RAM/disk storage and coordinated shutdown | Implemented for text and authenticated image-bearing prefixes. |
| Qwen atomic recurrent cache | Complete target/GDN/draft checkpoint bundles | Implemented for text with engine-driven transfer; external image-bearing checkpoint reuse is unsupported. |
| DS4.1 engine-driven cache | Target and auxiliary cache groups with independent Engram placement | Implemented; RAM/disk Engram placement is separate from prefix offload. |

Typed settings include `cache-mode`, `cache-transfer-mode`, `cache-l1-gib`,
`cache-l2-gib`, `cache-directory` and `cache-object-tokens`. Both
`LMCACHE_L1_SIZE_GB` and `LMCACHE_L1_GB` address one setting; contradictory
values fail at the selected precedence level. The supervisor owns process
groups, readiness, exact SHM name/capacity checks and coordinated shutdown.
The model resolver owns geometry; the supervisor never re-resolves policy.
Engine-driven service processes have an empty CUDA device list. Persistent
namespaces include immutable checkpoint identity, runtime identity and layout.
HF revisions are resolved and pinned before opening persistent storage, not
during `--print-config`. GLM DFlash preserves the target scheduler budget while
reserving additional input rows for draft verification. Existing LMCache
transport, allocation and checkpoint implementations remain authoritative.

For a local model directory, the first external-cache start hashes the weight
and configuration files to keep stored cache objects tied to their exact
checkpoint. Later starts reuse those file digests from
`/cache/checkpoint-identities` when the file list, sizes, inodes and modification
metadata are unchanged. Mount `/cache` persistently to avoid repeating the
full read after a container restart. Set
`LIL_CHECKPOINT_IDENTITY_CACHE_DIR=/path/to/writable/cache` to place the small
identity records elsewhere. Missing, invalid or stale records cause a full
content hash; they never disable checkpoint identity checks.

For the GLM Spark TP2/DCP2 recipe with a 3072-token scheduling budget, replace
`CACHE_MODE=vram` with the following environment arguments:

```bash
-e LMCACHE_MODE=disk -e LMCACHE_CHUNK_SIZE=3072 \
-e LMCACHE_L1_GB=16 -e LMCACHE_L1_INIT_GB=2 -e LMCACHE_L2_GB=64
```

Use `LMCACHE_MODE=ram` to omit disk storage. Keep `/cache` on a persistent
Docker volume for disk restore. At startup the disk tier is capped to 90% of
what its filesystem can give it (free space plus what the tier already holds),
with a warning, so it cannot fill the disk. A RAM arena left in `/dev/shm` by a
container that was killed or crashed is removed automatically; an arena whose
container is still running is refused.

The disk tier lives in `/cache/lmcache/<model>/<layout>/<checkpoint>`. The
layout covers the image runtime and the cache settings, so a new image or a
changed setting starts an empty tier, and the previous one stays on disk
outside `LMCACHE_L2_GB`. The startup log names such tiers with their sizes.
`-e LMCACHE_L2_PRUNE_STALE=1` deletes them at startup, except tiers held by a
running container and tiers written in the last 30 minutes. Containers from
images older than this option do not hold that lock: do not prune a volume
that such a container still uses.

Each request-boundary checkpoint holds the complete recurrent state, about
167 MB for Qwen at TP1 and about 215 MB per request for GLM-5.3-Flash at TP4.
With disk storage, a chat turn writes two or three of them even when the next
turn cannot use them, because the chat template rewrites the previous prompt
and response. `LMCACHE_L2_CHECKPOINT_WRITES` selects what reaches the disk:

- `always` (default) writes every checkpoint to disk.
- `on-evict` writes the current checkpoint of a conversation once, when it
  leaves RAM, and again for everything still only in RAM when the container
  stops. Checkpoints that a newer turn of the same conversation has
  superseded are never written. Disk writes fall from every request to about
  one checkpoint per conversation that goes idle, so the disk holds the
  newest state of many more conversations. Give the container time to stop:
  `docker run --stop-timeout 60` or Compose `stop_grace_period: 60s`; the
  shutdown write takes up to 30 s. A crash or `docker kill` loses checkpoints
  that were only in RAM.
- `on-reuse` keeps new checkpoints in RAM and writes each one to disk only
  after a restore from the cache has used it. A follow-up served from GPU
  memory does not count, so with long conversations little reaches the disk.

With every value, RAM and disk eviction remove superseded checkpoints before
any other entry. After a restart or eviction, a restore uses the longest
checkpoint that still exists. Disk retention is roughly the disk capacity
divided by the checkpoint bytes written per minute; check
`lmcache_mp_checkpoint_retention{stat="l2_checkpoint_bytes"}` on the cache
metrics port. The engine-driven service uses CPU memory,
not a separate GPU. The profile selects request-boundary checkpoints and a
matching target scheduling budget. It does not enable aligned/direct transfer
for TP2. GLM vision remains available, but image-bearing requests recompute
instead of restoring external recurrent checkpoints. The auto-fit context
limit can differ from the VRAM-only recipe because external-cache geometry
and transfer buffers differ; inspect the reported capacity at startup.

GLM direct cuMem transfer requires a helper library built from the **same
LMCache commit** as the wheel. Its source hashes, license and compiled-library
hash are recorded in the image. The build reuses cached source and does not
recompile the LMCache wheel. Unknown shell-text extensions are not evaluated;
operators can use the raw-command interface for unsupported wrapper options.

Production cutover requires:

1. Frozen effective argv/environment fixtures for each supported model/mode
   and cache contract, including explicit overrides. Classify intentional
   differences rather than deleting them from parity checks.
2. Model-neutral final image metadata, matching installed profile hashes,
   preservation of dependency patches, and a declared native/wheel capability
   manifest. Missing FlashKDA or LMCache components are build errors for a
   profile requiring them, not silent backend substitutions.
3. GPU smoke tests of each affected model/mode, graph/backend inspection,
   matched 32K prefill and C1 decode against its source-locked reference on
   the same physical GPU group and clocks. This is a bounded migration check,
   not a rerun of every historical tuning experiment.
4. For external cache: cold/L1/restart-L2, request and shared-SYSTEM endpoints,
   aligned retention, DCP/MTP/DFlash ownership, cancellation/eviction and
   sidecar shutdown checks. Preserve bytes, state identity and geometry.
5. Qualification of legacy profile-selector aliases. No duplicated kernel
   variables in wiki Compose; both community and wheel builds install the
   same profile source revision but retain independent qualification records.

## Validation

```bash
python -m pytest -q runtime/tests
ruff check runtime
ruff format --check runtime
bash -n runtime/lil-serve runtime/install.sh
```

The CPU suite checks precedence, alias conflicts, JSON composition, graph
caps, model isolation, secret redaction, actual GLM Bash dry-run argument
parity, Engram placement, Qwen proposal-head policy, image metadata and hash
gates, and generated artifacts. It does not claim full Bash-chain environment
equivalence for every model or qualified serving on a different runtime build.
