# Source-locked Jovian Judgement serving image

Status: **implemented; release qualification is tracked separately**.

The source references below include the corrected NVFP4 SwiGLU clamp. The
published R34 image does not: do not use its B12X revision as a deployment
baseline. See the [R34 preservation and correctness audit](r34-preservation.md)
for immutable artifact identities, the GPU reproducer and qualification limits.

The build installs complete committed vLLM, B12X and LMCache Git trees over an
immutable CUDA 13.3/PyTorch 2.13 runtime. It reuses source-compatible LMCache
native extensions and installs a source-authenticated FlashKDA operator with packed recurrent
checkpoint exports. The result has two filesystem layers; serving does not
require source-code bind mounts.

The runtime foundation supplies compatible PyTorch, CUDA, NCCL, InstantTensor
and unchanged native binaries. The stable vLLM extension is rebuilt together
with its Python callers because the DeepSeek query-output operator has a
caller-owned output ABI. FlashInfer Python/JIT wheels come from the immutable
DS4 Vision artifact identified in the source lock. FlashKDA remains separately
authenticated. Replacing any native dependency requires serving qualification.

## Build inputs

For a pinned branch-plus-PR reconstruction, use the
[public review composition contract](review-composition.md). It verifies all
three component trees from public review heads without private resolution
patches. The source table below identifies those composed trees. Historical
R34 identities are recorded separately in the preservation audit; do not mix
components from the two compositions.

Provide clean, complete Git checkouts as `vllm-source`, `b12x-source` and
`lmcache-source`. Use the release's published source refs and revisions, not
moving branch heads when reproducing a measured image. The image's
`/opt/glm53-flash/source.lock` records repository and package trees, build-input
hashes, Git-bundle hashes and native-artifact identity.

Host requirements: Git, Docker Buildx and `uv`; no host PyTorch or active GPU
is needed for image construction. Compilation uses the runtime foundation.

The GLM/Qwen/DS4 shared serving artifact uses the following published integration
refs. They preserve contributor history and integration resolutions; simply
merging arbitrary PR heads or changing the declared merge order does not
reproduce the source contract. The pinned, ordered manifests reproduce these
trees including their required integration resolutions.
These source refs are **implemented**, not by themselves release approval.

| Checkout | Repository | Commit |
|---|---|---|
| `vllm-source` | [voipmonitor/vllm](https://github.com/voipmonitor/vllm/tree/integration/jovian-reviewed-sources-20260911) | `de982a50c6a3e4718e5cf9f00423a92192718da1` |
| `b12x-source` | [voipmonitor/b12x](https://github.com/voipmonitor/b12x/tree/integration/jovian-reviewed-sources-20260911) | `98086604c86ec1e78977e5023ce282ecb97ab8a7` |
| `lmcache-source` | [local-inference-lab/LMCache](https://github.com/local-inference-lab/LMCache/tree/release/jovian-fp4-fs-ledger-r33-20260910) | `29bc5a2efde737c436b04499eb62cd1776cebeec` |

Clone each linked branch into its checkout directory, then verify `git rev-parse HEAD`
against the table. Complete checkouts are required; do not use shallow clones
for the source bundles. The package trees and compiled FlashKDA artifact are
reproducible inputs; OCI timestamps and archive metadata are not promised to
produce a bit-identical image digest on a different builder.

From this recipe directory, build the FlashKDA operator:

```bash
git clone --depth 1 --branch v4.4.2 https://github.com/NVIDIA/cutlass.git cutlass-source
docker buildx build \
  --build-context vllm_source=./vllm-source \
  --build-context cutlass_source=./cutlass-source \
  --output type=local,dest=./flashkda-artifact \
  -f Dockerfile.glm53-managed-checkpoint-native .
```

The native build resolves the FlashKDA revision and CUTLASS submodule declared
by vLLM, applies its recorded checkpoint patch, and exports the binary with
source and checksum records. Do not supply an unpatched `FLASH_KDA_SRC_DIR`.

Build the stable extension from the same committed vLLM tree:

```bash
uv run --no-project --python 3.12 build_jovian_stable_native.py \
  --vllm ./vllm-source --cutlass ./cutlass-source --output ./stable-artifact
cp ./stable-artifact/_C_stable_libtorch.abi3.so \
   ./stable-artifact/native-source.identity ./flashkda-artifact/
```

The source identities must match the vLLM `csrc`, `cmake`, and `CMakeLists.txt`
objects in the final source bundle. The source table includes the DS4
query-output ABI and serving contracts; a binary from a different caller ABI
is not a compatible substitute.

Freeze source bundles and install them:

```bash
uv run --no-project --python 3.12 prepare_glm53_source_bundles.py \
  --vllm ./vllm-source --b12x ./b12x-source --lmcache ./lmcache-source \
  --vllm-repository https://github.com/voipmonitor/vllm.git \
  --b12x-repository https://github.com/voipmonitor/b12x.git \
  --lmcache-repository https://github.com/local-inference-lab/LMCache.git \
  --native-artifact ./flashkda-artifact --uv "$(command -v uv)" \
  --lmcache-native-mode reuse-all \
  --output ./source-bundles \
  --release-name jovian-judgement-community-source-locked \
  --release-version source-locked
bash build_glm53_cache_contract_image.sh \
  ./source-bundles ./flashkda-artifact local/glm53:source-locked
```

The bundle directory must not already exist. Uncommitted serving changes,
bundle/tree mismatches and native-patch mismatches fail the build. Git bundles
preserve contributor history; no source squashing or filesystem overlays are
used to impersonate a committed tree.

Repository labels come from the manifest. The optional `--*-repository`
arguments identify the published composition repository rather than inferring
its owner from the package name; without them, the preparer uses the checkout's
`origin`. URLs containing credentials are rejected. A recorded URL is not
proof of publication: verify that it serves the locked commit before release.
Native compilation archives the committed CUTLASS tree and records its actual
commit and tree instead of inferring a version from the build instructions.

`source_locked_image_labels.py` clears inherited serving claims and replaces
them from the authenticated manifest. It preserves metadata for unchanged
runtime dependencies. An implementation label is not evidence of qualification.

Run the CPU-only installation and metadata contracts with:

```bash
uv run --no-project --with pytest python -m pytest -q tests/test_build_contract.py
```

## Serving contract

### GLM Spark on two GPUs

The [Spark TP2 serving contract](glm-spark-tp2.md) describes an explicit
1,048,576-token context, bounded cuBLAS workspace, native vision and worker-owned
LMCache copies for TP2/DCP2 MTP3. Automatic KV sizing and context shortening
are separate opt-in controls. Qualification identifies the tested model,
hardware and image; the fixed memory budget is not a guarantee for every host.

### GLM cache HTTP binding

Status: **implemented**. `LMCACHE_HTTP_HOST` selects the LMCache HTTP bind
address; the default remains `127.0.0.1`. Use `0.0.0.0` for pod-IP metrics
scraping only behind a trusted-network policy or authenticated proxy: this
listener also exposes administrative APIs. Readiness probes use a concrete
address compatible with the selected bind, including IPv6.

Use this explicit setting for HTTP binding. The GLM wrappers do not interpret
extra arguments as shell commands or permit them to override checkpoint
transport, shared-memory ownership, or readiness addresses.

### LMCache checkpoint publication, RAM retention and server arguments

Status: **implemented**; GPU qualification must identify the tested image.

Concurrent request-boundary generations may share immutable attention pages.
If another producer is writing a shared page, checkpoint admission returns a
metadata-only `busy` response. The worker-owned background transfer retries
with bounded backoff, without blocking the model or the LMCache RPC handlers.
A committed page needs no additional copy; an aborted owner's page can be
reserved by the waiting producer. Capacity and cancellation failures remain
safe misses, and publication still requires every rank's acknowledgement.
This behavior is implemented by Derek Yates's
[LMCache #65](https://github.com/local-inference-lab/LMCache/pull/65).
Run workers and the sidecar from the same image so both understand the response.

`LMCACHE_L2_PREFETCH_POLICY` accepts `retain` (default) or `default`.
`retain` keeps objects loaded from the filesystem tier in the bounded L1 RAM
pool after readers finish. They remain eligible for ordinary eviction.
When filesystem storage is enabled, retained restores can synchronously evict
unlocked LRU objects to obtain staging space. This remains write-through:
restores do not switch the cache into synchronous writeback. Active read/write
owners remain protected, and insufficient reclaimable capacity remains a safe miss.
`default` releases temporary loaded objects when their final reader finishes.
This setting does not control the GPU hardware L2 prefetcher.

`LMCACHE_SERVER_EXTRA_ARGS` appends whitespace-separated server arguments,
for example `--max-cpu-workers 4`. Arguments are literal: shell expansion,
quote interpretation, and multiline values are unsupported. Quoted array
expansion prevents wildcard characters from becoming filenames. Additional
scalar options take precedence over earlier defaults when the server accepts
repetition. Launcher-managed transport, identity, geometry, prefetch policy,
and listener options must use their dedicated settings instead.

The interface derives from Tim Rice's
[launcher proposal](https://github.com/local-inference-lab/rtx6kpro/pull/100),
with literal argument handling and explicit protection for launcher-owned
configuration.

LMCache packaging records `lmcache.native.mode` in the source lock.
`reuse-all` requires identical native sources, Rust sources, build policy and
requirements before copying the compiled payload. `cpu-rebuild-cuda-reuse`
builds all three common C++ extensions with `NO_GPU_EXT=1`, then copies only the
CUDA extension from the immutable `lmcache.native.artifact.image`. Every native
input outside the two CPU implementation directories must match that reference,
including shared headers and build policy. The platform wheel's tags and RECORD
describe the combined payload. The build-stage image does not contribute its
layers to the serving image, which retains two filesystem layers.

The filesystem connector reconciles missing objects during idempotent deletion
and protects keys with pending native stores, following Derek Yates's
[LMCache #67](https://github.com/local-inference-lab/LMCache/pull/67). Bounded
object paths, readable legacy paths and real filesystem errors remain supported.
The native donor identified in `lmcache.native.artifact.image` contains this
filesystem correction. `reuse-all` retains that compiled implementation after
the source comparison. A donor without the correction requires
`cpu-rebuild-cuda-reuse`; copying only Python is insufficient.

The image sets `VLLM_DEFAULT_MOE_BACKEND=b12x`. Native `vllm serve` and Python
`KernelConfig()` therefore select B12X when no MoE backend is supplied. An
explicit `--moe-backend auto`, another backend, or a nested kernel configuration
overrides that default. The GLM wrapper also supports `MOE_BACKEND`; its default
is B12X. This policy does not replace attention or sampler backends, nor explicit
speculative-model backend choices. Outside the image, vLLM retains `auto` when
`VLLM_DEFAULT_MOE_BACKEND` is unset. Unsupported explicit B12X configurations
fail normally rather than silently selecting another implementation.

Persistent LMCache namespaces resolve the effective positional or `MODEL`
checkpoint and its immutable model/draft identity in both request-boundary and
aligned modes. Hub revisions are pinned; local weights and configuration are
content-hashed before serving. Aligned transfers use that identity only for
the filesystem namespace, without switching to the semantic connector.
Dry-run performs no model I/O and labels unavailable identities as unresolved.

### Publisher sampling defaults

Status: **implemented**, with CPU launcher-precedence tests. GPU quality and
performance measurements identify the sampling parameters actually supplied;
changing a default is not evidence of a model-quality repair.

The GLM launcher supplies `--override-generation-config
'{"temperature":1.0,"top_p":0.95}'` because a quantized checkpoint can omit
the [Z.ai generation defaults](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/main/generation_config.json).
The DS4 text and Vision launcher uses the same values for its agentic serving
profile, following the [DeepSeek text deployment recommendation](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731#how-to-run-locally)
and [Vision agent evaluation settings](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp).
For non-agentic DS4 use, DeepSeek recommends top-p 1.0 instead.

Explicit API request parameters override these server defaults. Supplying
`--generation-config`, `--override-generation-config` (including dotted fields)
or a native `--config` file selects operator policy instead of the launcher
preset. It is forwarded unchanged; the launcher does not add a competing JSON
object. DS4 also recognizes these options in `EXTRA_VLLM_ARGS`.

The separately documented Qwen native launch uses its checkpoint's generation
configuration: temperature 1.0, top-p 0.95 and top-k 20. These match the
[Qwen thinking-mode defaults](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/generation_config.json)
and are present in the qualified NVFP4 revision
`b797d2e1160b9596b2570e56c1d3590faa09d4ed`. Qwen's non-thinking profile has
different recommendations; disabling thinking alone does not automatically
change request sampling. A command that bypasses the model launcher must set
its own generation policy or retain the publisher's checkpoint metadata.

GLM's community chat defaults remain `reasoning_effort=high` and
`clear_thinking=false`. Sampling defaults do not clear historical reasoning,
truncate prompts, change the target KV dtype or quantize model weights.

### DeepSeek V4 model profiles

Status: **implemented; combined-image serving qualification is tracked separately**.
Use `--entrypoint /usr/local/bin/serve-ds4-jovian.sh` or
`ds4-jovian.compose.yml`. This selects the same installed packages as GLM,
with B12X attention/W4A8 MoE and DeepGEMM dense projections. It does not change
GLM or Qwen backend selection.

`DS4_MODEL_VARIANT=text` selects `deepseek-ai/DeepSeek-V4-Flash-0731`, fixed
probabilistic DSpark K5 and eight request slots. `DS4_MODEL_VARIANT=vision`
selects `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`, K3 and four request slots.
Both default to TP2/DCP1, FP8 compressed KV, buffered InstantTensor,
`FULL_AND_PIECEWISE`, and a 4096-token scheduler budget. `MODEL` can name a
local checkpoint or a Hugging Face repository. The native launcher reports
the resolved model revision and serving command.

The image has GLM-specific graph and CPU environment defaults. The DS4
entrypoint deliberately replaces those with `DS4_OMP_NUM_THREADS=2`,
`DS4_MAX_CUDAGRAPH_CAPTURE_SIZE=auto`, and
`DS4_CUDAGRAPH_CAPTURE_SIZES=default`. Automatic caps are 48 rows for text K5
and 16 for Vision K3. Use these DS4-prefixed settings to override that profile;
changing the GLM environment variables does not change the DS4 graph contract.

GPU-local caching is the default. `LMCACHE_MODE=ram|disk` enables worker-owned
asynchronous SHM transfer with a CPU-only sidecar, a 24 GiB L1 pool and
4096-token objects. `ram` does not enable filesystem storage. `disk` uses
`LMCACHE_L2_PATH` and `LMCACHE_L2_GB`. Select independent cache ports/namespaces
when serving multiple models. DS4 engine-driven TP2 with a model limit of at
least one million tokens defaults to utilization 0.970; GPU-only uses 0.975.
These are reference deployment profiles for 96 GiB GPUs, not universal memory
capacity or topology guarantees. Validate admission on the deployment hardware.

Use an empty external-cache namespace when adopting the paged-gather metadata
correction in LMCache #50. The source revision in the table keeps every batch's
pinned block-ID snapshot immutable until its asynchronous copy completes.
Payloads exported by a build without that correction can contain bytes from
different GPU pages while retaining valid storage keys. DS4 filesystem keys
do not automatically invalidate those payloads on a source update. Use a
separate Docker cache volume or L2 directory; do not delete another service's
cache or treat a successful read as proof that an existing payload is valid.

### Python scheduler overlay

Status: implemented; GPU qualification is recorded per image in the linked
model runbook. `Dockerfile.glm53-scheduler-overlay` applies scheduler Python
sources and launcher controls to the immutable R28 FP8 image without replacing
CUDA, PyTorch, B12X, FlashKDA, NCCL, or LMCache. It produces three filesystem
layers. The installed Git history remains complete; the incremental bundle
requires the parent image's authenticated source objects.

Extract the parent manifest and prepare the overlay from a clean vLLM checkout:

```bash
git -C vllm-source fetch https://github.com/voipmonitor/vllm.git \
  integration/glm-scheduler-hardening-20260908
git -C vllm-source checkout --detach 9ff42d83938e74018f9c255e8cfa7ca6df6921b0
docker run --rm --entrypoint /bin/cat \
  voipmonitor/vllm@sha256:f5f121e37fd2afbb6f8f036e7eb627435cfb736de0a4420306dc2a25b6631669 \
  /opt/glm53-flash/source.lock > r28.source.lock
uv run --no-project prepare_glm53_scheduler_overlay.py \
  --vllm ./vllm-source --parent-lock ./r28.source.lock \
  --uv "$(command -v uv)" --output ./scheduler-bundles \
  --release-name jovian-judgement-community-20260908-r28.1 \
  --release-version r28.1
bash build_glm53_scheduler_overlay.sh ./scheduler-bundles local/glm53:scheduler
```

The preparer rejects changes outside the scheduler and its tests. The installer
verifies the parent manifest, source trees, input hashes, and FlashKDA binary.
Generated vLLM version metadata identifies the installed Python source; native
component identities remain those in the manifest. A changed source lock
produces safe external-checkpoint misses across image versions. It does not
reinterpret persistent checkpoints written by another runtime identity.

### Scheduler controls

The image retains fixed prefill compute share `0.4`, schedule interval `1`, and
one prefill lane. Interleaving is opt-in. Explicit CLI values override the
corresponding environment variable and are emitted only once.

| Environment | vLLM option | Values |
|---|---|---|
| `PREFILL_COMPUTE_SHARE` | `--prefill-compute-share` | `auto` or finite float strictly between 0 and 1 |
| `PREFILL_COMPUTE_HALF_LIFE` | `--prefill-compute-half-life` | `smooth`, `responsive`, or positive finite seconds; requires auto share |
| `MAX_PARALLEL_PREFILLS` | `--max-parallel-prefills` | `auto` or positive integer |
| `PREFILL_POLICY` | `--prefill-policy` | `round-robin` or `decode-aware` |
| `DECODE_REFILL_TARGET` | `--decode-refill-target` | `auto` or positive integer |

Automatic lane count is `min(4, max_num_seqs)`, independently of attention,
recurrent, or LMCache object geometry. The scheduler token budget still limits
each forward pass. With automatic refill, the refill target follows the
effective lane count. A numeric fixed share rejects any half-life setting.
Compute-share scheduling requires interval `1`.

`FAIRNESS_ENGINE=none` suppresses the inherited environment share; an explicit
CLI share remains authoritative. `FAIRNESS_ENGINE=compute_share` is supported
for compatibility, while `micro_slicing` is rejected. `--help` lists the
controls without loading models. `CACHE_MODE=vram DRY_RUN=1` prints the full
command. Unit tests run with `uv run --no-project --with pytest pytest -q tests`.

### Models and cache

The installed launcher is `/usr/local/bin/serve-glm53-flash.sh`. Model defaults
are `local-inference-lab/GLM-5.3-Flash-NVFP4` and
`local-inference-lab/GLM-5.3-Flash-DFlash2`; revisions are resolved at startup.
The latter checkpoint contains offline MXFP8 draft weights. A changed model or
runtime identity cannot reuse an incompatible external recurrent checkpoint.

The GLM launcher supplies `--default-chat-template-kwargs
'{"reasoning_effort":"high","clear_thinking":false}'`.
A chat request can override the reasoning default with
`"reasoning_effort":"max"`, `"high"`, or `"low"`; this does not change
`max_tokens` or the parser. Without a server/request override, the checkpoint's
chat template selects `max`. Direct `vllm serve` commands bypass this launcher's
defaults and must provide the option themselves.

The `clear_thinking=false` agent profile follows the
[model author's preserved-thinking guidance](https://docs.z.ai/guides/capabilities/thinking-mode#preserved-thinking).
Clients must return the complete, ordered assistant reasoning with the message
history. The template retains that reasoning at its original assistant turn;
it does not insert historical reasoning into the generated turn. The setting
cannot recover reasoning omitted by the client.

Removing completed-turn reasoning changes the rendered token prefix. A cached
response containing that reasoning cannot be reused past the first changed
token; instruction and matching input-prefix checkpoints remain usable.
Chat applications intentionally omitting completed-turn reasoning can send
`"chat_template_kwargs":{"clear_thinking":true}`. An explicit server CLI
`--default-chat-template-kwargs` object replaces the launcher's object instead
of being duplicated. These defaults apply to the GLM entrypoint, not DS4 or
Qwen native serving commands.

The production scheduler budget is 4096 tokens, OMP threads1, NCCL channels16
with 2 MiB buffers, and `FULL_AND_PIECEWISE` CUDA graphs. GPU-local attention
pages default to2048 tokens. Target operations use B12X; FlashKDA supplies
checkpoint-producing recurrent prefill. The private MTP vocabulary projection
is NVFP4, while the target vocabulary projection remains BF16.

Modes use `SPECULATOR=mtp MTP_DEPTH=0`, `SPECULATOR=mtp MTP_DEPTH=3`, or
`SPECULATOR=dflash2 DFLASH_DEPTH=7`. External cache is opt-in with
`CACHE_MODE=lmcache`; its default transfer is `engine_driven`. Semantic
checkpoints use immutable all-rank storage and asynchronous shared-memory
copies performed by vLLM workers, without a sidecar CUDA context.
Checkpoint payload namespaces are scoped to rank and storage group so that
unlocked pages remain evictable under RAM pressure. Complete manifests and
transfer leases enforce atomic retrieval. Version-1 semantic payloads are
incompatible with these version-2 keys and produce safe cache misses.

Mode-specific deployment instructions and measured conditions belong in the
[GLM-5.3-Flash wiki](https://github.com/local-inference-lab/rtx6kpro/blob/master/models/glm-5.3-flash.md).
Qwen source may be included in the selected Git trees, but this launcher serves
GLM. A source build alone establishes neither Qwen runtime qualification nor
NVFP4 target-KV qualification.
