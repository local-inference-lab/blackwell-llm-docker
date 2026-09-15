# Model runtime configuration

Status: **implemented**, with CPU configuration-contract tests. The resolver
does not replace the published community launchers or the wheel-runtime
entrypoint. GPU serving, throughput, and external-cache migration are not
qualified by these tests. In particular, this work does not fix B12X prepared
kernel startup failures or supply missing native extensions.

## Ownership

Keep deployment policy in `blackwell-llm-docker`. A separate Docker repository
would introduce another version boundary between package composition,
entrypoints, CI, and model documentation without removing a configuration
owner. The LIL client can consume the versioned launch interface; it should
not maintain another copy of kernel settings.

The configuration has three independent identities:

| Artifact | Owns | Does not own |
|---|---|---|
| Model profile, `profiles/*.yaml` | Checkpoint name, model-specific precision, speculation, cache layout, native CLI defaults | GPU selection, clocks, library builds |
| Hardware profile, `hardware/*.yaml` | Explicitly selected communication and platform tuning | Model architecture or speculation method |
| Image contract, `image-contract.json` | Runtime-lock digest, installed profile/launcher hashes, CUDA/NCCL bootstrap executable | Mutable benchmark results or credentials |

`hardware/native.yaml` leaves communication crossovers and NCCL channels to
native vLLM/B12X selection. `hardware/rtx-pro-6000-pcie.yaml` carries the
single-node workstation deployment settings, including 16 NCCL channels and
a 2 MiB buffer. These are not claimed to be optimal on unswitched workstations,
GB10, or multi-node systems. Neither hardware profile changes GPU clocks.

`schema.json` validates profile structure. `options.yaml` owns the mapping
between managed native CLI options, typed values, and environment aliases.
There is one common layer, one model layer, and one hardware layer, not an
unbounded inheritance chain. A settings file and explicit user arguments are
applied after these layers.

## Model policies

The [generated parameter table](generated/parameters.md) is rendered from the
same profiles as the launch command. Its values describe configuration, not
performance measurements. Source references are recorded inside each profile.

- GLM: NVFP4 target; target MoE and dense backends explicitly B12X. Optional
  MTP defaults to three proposals with the private NVFP4 draft vocabulary
  head, Marlin draft MoE, and B12X draft attention. DFlash2 defaults to seven
  proposals from `local-inference-lab/GLM-5.3-Flash-DFlash2`, FLASH_ATTN draft
  attention and `auto` draft KV, retaining the MXFP8 checkpoint policy.
  Target KV remains FP8. FlashKDA prefill is retained; selecting B12X MoE
  does not imply replacing recurrent prefill with B12X.
- DS4 text/Vision: fixed DSpark K5/K3, B12X W4A8 MoE, and native dense
  selection corresponding to `BACKEND=b12x-a8-dglin`. That source launcher
  **omits** `--linear-backend`; the profile preserves the omission rather than
  claiming it proves DeepGEMM dispatch. `--linear-backend deep_gemm` and
  `--linear-backend b12x` are explicit alternatives requiring dispatch and
  performance checks. Model choice is explicit: changing speculation mode
  never silently changes the checkpoint repository.
- DS4.1: DSpark K7 with adaptive verification, greedy proposals, standard
  rejection, B12X target/draft attention and B12X MoE/dense. Engram table
  placement selects `ram` or `disk` independently of general CPU offload.
  Main/SWA pages remain 256/128. Breakable prefill graphs remain disabled;
  the native graph configuration remains FULL_AND_PIECEWISE.
- Qwen: TP1 by default; TP2 is an explicit override. Preserve CPU PLE tables,
  the BF16 target vocabulary head and private NVFP4 MTP copy. The 6,019-token
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
  --env OMP_NUM_THREADS=1 -- --engram-table-memory ram
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
continue to accept them. Their eventual thin compatibility adapters must
translate them once into this interface; do not implement another precedence
resolver in shell. `FAIRNESS_ENGINE=none` is supported; `micro_slicing` is not.

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
B12X, vLLM, or LMCache, and does not replace the production entrypoints.

The CUDA 13.4 wheel image must preserve its CUDA/NCCL bootstrap:

```bash
bash runtime/install.sh /build/foundation.inspect.json RUNTIME_LOCK_SHA256 \
  /opt/venv/bin/qwen38-ngc-runtime
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

A profile-enabled image should retain the existing two-filesystem-layer
composition: immutable neutral runtime foundation plus serving sources and
profiles in the installation layer. Removing `ENV` from a child Dockerfile
does not remove values inherited from `FROM`. Produce a neutral foundation
configuration without deleting runtime files or dependency patches. Do not
add successive release-on-release layers and do not rebuild native components
merely to change model policy.

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

The direct resolver intentionally rejects LMCache/native offload execution
until lifecycle adapters are implemented and tested. This is **not** removal
of LMCache from the community image: existing source-locked cache launchers
are unchanged. They remain the executable interface for those deployments.
Do not set the image default entrypoint to `lil-serve` yet.

| Contract | Required preserved behavior | Resolver disposition |
|---|---|---|
| GLM atomic recurrent cache | Target/recurrent/draft all-rank bundles, request/SYSTEM boundaries, aligned exports, DCP-derived object geometry, identity checks, cancellation and restart restore | Lifecycle adapter required; not replaced by a generic KV connector |
| DS4 engine-driven cache | Worker-owned async pinned SHM, CPU-only sidecar, RAM/disk modes, health gating, coordinated shutdown, memory admission | Lifecycle adapter required; reuse existing transport implementation |
| Qwen external cache | Separate model/cache correctness qualification | Unsupported, matching the published recipe's qualification limit |
| DS4.1 external cache | Model-specific cache qualification | Not inferred from Engram RAM support |

Use canonical external-cache settings `cache.mode`, `cache.transfer`,
`cache.l1_gib`, `cache.l2_gib`, `cache.path`, and `cache.object_tokens` in the
adapter schema. Translate both `LMCACHE_L1_SIZE_GB` and `LMCACHE_L1_GB` to
`cache.l1_gib`, rejecting contradictory values. The adapter must produce an
immutable service plan containing sidecar command/environment, readiness
condition, final worker arguments, geometry and namespace. A supervisor owns
process lifetimes; the model resolver owns model defaults. Neither re-resolves
the other's configuration. The sidecar remains CPU-only for engine-driven
transport. Do not implement a second SHM allocator, checkpoint store, or restore
protocol as part of configuration consolidation.

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
5. Thin legacy aliases only after these gates pass. No duplicated kernel
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
