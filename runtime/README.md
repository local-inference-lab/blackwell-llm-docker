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

Restarts: when the engine fails (for example a CUDA out-of-memory error on one
GPU), the server stops within about a second and the container exits with status
1. The generated Compose files use `restart: on-failure`, and the `docker run -d`
examples below pass `--restart on-failure`, so Docker starts the server again.
A container you stop yourself, or one that fails within 10 seconds of starting
(a configuration error), is not restarted. Use `unless-stopped` to also start the
server after a host reboot. The generated Compose files also set
`stop_grace_period: 60s`, so an external cache has time to write its
checkpoints when the container stops; a container without one still stops in
seconds.

Profiles keep downloaded checkpoints and saved HF credentials in
`/root/.cache/huggingface` (`HF_HOME`), independently of the runtime-keyed JIT
cache under `/cache/jit`. Updating an image therefore does not move the model
cache. To use another location, mount that directory and set `-e HF_HOME=/path`.
`XDG_CACHE_HOME` controls the profile's JIT root; it does not relocate HF data.

## Named deployment settings

`PRESET` selects a data-only overlay from `presets.yaml`. Model defaults remain
separate from deployment-specific memory constraints. The default GLM recipe
for two 96-GB RTX PRO GPUs is `glm53-tp2`:

```bash
docker run -d --name glm-tp2 --init --gpus '"device=0,1"' \
  --restart on-failure --network host --ipc host --shm-size 32g --stop-timeout 30 \
  --ulimit memlock=-1 --ulimit stack=67108864:67108864 \
  -v model-cache:/root/.cache/huggingface -v glm-tp2-runtime:/cache \
  -e PRESET=glm53-tp2 -e PORT=8000 "$LIL_IMAGE"
```

It serves the QAD weights from the stored checkpoint
`local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD` at revision `dec48abd`:
MXFP8 attention and shared experts, and NVFP4 routed experts whose scales stay
losslessly compressed (FP4-CSF). The routed experts decode with BF16
activations and FP32 router weights (W4A16) and prefill with NVFP4 activations
(`PREFILL_ACTIVATIONS=a4`): long prompts run on FP4 tensor cores while
generation keeps the precise path. It runs TP2/DCP2, MTP3 with B12X drafter
experts, eight request slots, a 4,096-token prefill budget and a 7,296 MiB KV
cache per GPU (about 1.84M tokens). The input embedding table lives in host
RAM, the checkpoint stores the vision tower with MXFP8 attention and W4A16
NVFP4 MLPs, and the TP, DCP and EP groups share one NCCL communicator.

The prefix cache also lives in host RAM. LMCache keeps up to 64 GiB of
conversation KV and recurrent checkpoints, so a conversation that left the GPU
cache is restored instead of prefilled again. With eight agents whose contexts
grew past 450K tokens each, it served 435 turns in 15 minutes with 97% of the
prompt tokens from cache (90% of turns under 18 s), against 155 turns, 65% and
123 s with the GPU cache alone.

- Host RAM: engine-driven LMCache pins its whole RAM tier in `/dev/shm` at
  startup (with `--ipc host`, the host's). When less is free, the tier shrinks
  to 90% of the free `/dev/shm` and half of the RAM available to the container
  (a memory limit counts), at least 8 GiB, and the log names the size;
  `-e LMCACHE_L1_GB=<GiB>` sets it explicitly and must fit as given.
  The tier is freed when the container stops; `--stop-timeout 30` leaves time
  for that.
- Disk tier, so conversations also survive restarts: `-e LMCACHE_MODE=disk`.
  GPU cache only: `-e CACHE_MODE=vram`, with the same KV size.
- The 4,096-token prefill budget makes one LMCache object fill one 2,048-token
  page per DCP rank. A 3,072-token budget gives 1,536-token pages and about 9%
  fewer KV tokens; it also prefilled 32K prompts about 5% slower.
- Qualified worst case at eight slots: ten minutes of eight concurrent streams
  mixing 1K-168K-token prompts, one to ten images per request up to 6000x4000,
  and 64-1,024-token outputs, then fifteen minutes of eight agents growing past
  450K tokens each. 499 MiB per GPU stayed free at the peak. A larger explicit
  KV size or a desktop session on the same GPUs reduces that margin.
- It serves only its own checkpoint: `CHECKPOINT=original` is refused; use the
  GLM profile without a preset for other checkpoints.
- The image's vLLM must read `nvfp4_csf` checkpoints; otherwise the launcher
  refuses the preset.
- Sixteen request slots: `-e MAX_NUM_SEQS=16`. Each slot above eight takes
  64 MiB from the KV allocation for larger CUDA graphs and buffers. An explicit
  `KV_CACHE_MEMORY_BYTES` is used as given.
- `PREFILL_ACTIVATIONS=a16` keeps BF16 activations in prefill too. Five runs of
  estonia, hotel-lights, lavd-test, tool-eval and needle-checksum found no
  significant accuracy difference to a4, see
  [GLM-5.3-Flash expert precision](#glm-53-flash-expert-precision).
- `PRESET=glm53-spark-tp2` (the Spark checkpoint on two GPUs) was removed; it
  now stops with a pointer to `glm53-tp2`.
- Change serving choices with the same `SPECULATOR`, `MTP_DEPTH`, `TP`, `DCP`,
  `MAX_NUM_SEQS`, and `MAX_NUM_BATCHED_TOKENS` controls as other profiles.
  GLM cache object size follows a changed prefill budget unless explicitly set.
  Automatic capture capacity follows request slots and effective proposal width.
- Native form: `lil-serve --preset glm53-tp2 -- --port 8000`.
- Qwen TP2: `PRESET=qwen38-tp2`; CPU PLE placement and MTP defaults still come
  from the Qwen model profile. `PROFILE=qwen38-flash-next TP=2` is equivalent.
- Qwen as two TP1 replicas: `PRESET=qwen38-tp1x2` runs one complete server per
  GPU behind one endpoint, see [Replicas](#replicas-several-tp1-servers-behind-one-endpoint).
- GLM on three GPUs: `PRESET=glm53-tp3` (TP3 with expert parallelism, MTP3,
  eight request slots). MLA and KDA heads are padded to divide by three with
  zero weights, the vision tower runs data-parallel, and the large dense
  projections decode in FP8 (`-e VLLM_GLM53_FP8_DENSE=0` keeps them BF16).
  The external cache (`CACHE_MODE=lmcache`) is not available at TP3; the VRAM
  prefix cache is. The first start tunes FlashInfer MoE kernels and stores the
  result under `/cache`.
- GLM-5.3 (744B) on eight GPUs: `PRESET=glm53-csf-tp8` (profile `glm53`)
  serves `local-inference-lab/GLM-5.3-NVFP4-CSF` with TP8 and MTP3 from the
  model's built-in layer, 64 request slots and an 8,192-token prefill budget.
  The checkpoint's BF16 attention and shared experts are quantized to MXFP8
  while loading (`QUANTIZATION_CONFIG`), the MTP layer's routed experts run
  W4A16 NVFP4 on B12X, and the MTP drafts with an NVFP4 copy of the
  vocabulary head. B12X runs the DSA sparse attention with IndexCache, the
  W4A16 routed experts with two CTAs per SM for small batches
  (`B12X_W4A16_SMALL_M_OCCUPANCY=2`) and the fused PCIe all-reduce +
  RMSNorm; attention weights are prefetched into L2 during decode. The
  external cache is not available; the VRAM prefix cache is. `DCP=2` holds
  the full 1M-token context:

  | | `DCP=1` (default) | `DCP=2` |
  | --- | --- | --- |
  | KV cache tokens | 616,448 | 1,186,944 |
  | Longest context | 616,384 | 1,048,576 |
  | Decode steps/s C1 / C2 / C4 | 89.8 / 144.0 / 213.5 | 76.2 / 123.6 / 184.6 |
  | Prefill tok/s 8K / 128K | 8,316 / 6,928 | 7,771 / 7,420 |

  Measured on eight RTX PRO 6000 Workstation Edition GPUs with
  MTP-normalized decode steps; a needle in the middle of a prompt near the
  longest context is found at both sizes (578,836 tokens at DCP1 and
  961,990 at DCP2). DCP1 sizes the KV cache at 92% of GPU memory. With DCP
  the indexer and the DCP exchanges allocate context-sized buffers outside
  that budget, and at 92% a 1M-token prompt runs DCP2 out of memory, so the
  preset fixes the KV cache per GPU instead: 30.25 GiB at DCP2 and 27 GiB
  at DCP4 and DCP8. A
  1,048,448-token prompt, then 64 concurrent 8K prompts, then four minutes
  of 1K-96K prompts leave at least 500 MiB free per GPU at those sizes.
  `KV_CACHE_MEMORY_BYTES` overrides them. DCP8 holds 4.24M tokens. Batches
  that mix a prefill chunk with decode rows gather the cache for their
  prefill rows as well (`VLLM_B12X_MLA_CKV_GATHER_MIXED`, derived): a
  128K-token prompt arriving while 16 requests decode starts after 18.7 s at
  DCP2 and 22.9 s at DCP8 (385 s at DCP8 when such batches exchanged every
  head's queries instead).
- GLM-5.3 (744B) on six GPUs: `PRESET=glm53-csf-tp6` serves the same FP4-CSF
  checkpoint with TP6 and the same MXFP8 and NVFP4 drafting recipe as TP8.
  The attention heads are padded to 66 and the expert channels to 2112 (352
  per GPU) with zero weights. It runs 16 request slots and a 4,096-token
  prefill budget, with two NCCL channels for the DCP groups' exchanges.
  Decode context parallelism trades speed for context; `DCP=3` holds about
  1M tokens and `DCP=6` GLM-5.3's full 1M-token context twice:

  | | `DCP=1` (default) | `DCP=2` | `DCP=3` | `DCP=6` |
  | --- | --- | --- | --- | --- |
  | KV cache tokens | 353,152 | 667,008 | 1,000,512 | 2,000,779 |
  | Longest context | 353,088 | 666,880 | 1,000,320 | 1,048,576 |
  | Decode steps/s C1 / C2 / C4 | 72.7 / 103.3 / 173.6 | 56.8 / 83.9 / 138.3 | 51.1 / 73.8 / 123.0 | 43.8 / 62.7 / 112.2 |
  | Prefill tok/s 8K / 128K | 6,566 / 5,699 | 6,081 / 5,693 | 6,097 / 5,629 | 6,062 / 5,579 |

  Measured on six RTX PRO 6000 Workstation Edition GPUs with MTP-normalized
  decode steps; a needle in the middle of a prompt near each size's longest
  context is found at every DCP size (at DCP6 in a 1.03M-token prompt; a
  1M-token prompt prefills in about 5.6 minutes there). The KV cache is
  fixed per GPU (18 GiB at DCP1, 17 GiB with DCP): a prompt at the longest
  context, then 64 concurrent 8K prompts, then four minutes of 1K-96K
  prompts leave at least 677 MiB free per GPU at every DCP size.
  `KV_CACHE_MEMORY_BYTES` overrides it. Of these sizes the B12X PCIe
  all-to-all serves DCP2 only: DCP3 uses vLLM's generic all-to-all, and DCP6
  exchanges by all-gather plus reduce-scatter (`DCP_COMM_BACKEND=ag_rs`,
  derived). `glm53-csf-tp8` decodes C1 90, C2 144, C4 214 on eight. TP5 does
  not fit a useful context: even with MXFP8 its weights take about 83 GiB per
  GPU, which leaves room for well under 1M tokens at DCP5.
- MiMo-V2.6-Flash on two GPUs: `PRESET=mimo26-flash-tp2` serves
  `XiaomiMiMo/MiMo-V2.6-Flash-RL` (FP8 weights, text, images and audio) on
  b12x with the checkpoint's own DFlash drafter: seven draft tokens, adaptive
  verification, drafts sharded across both GPUs. The FP8 KV cache holds about
  1.31M tokens, enough for the full 1M context. Video input is off at TP2,
  because vLLM would reserve about 6 GiB per GPU for a maximum-size video.
  `PROFILE=mimo26-flash` is the four-GPU default and keeps video. The profile
  pins checkpoint revision `5711b268`, because older snapshots ship an invalid
  `dflash/config.json`. The drafter is read from the `dflash/` folder of that
  snapshot. Sampling defaults to temperature 1.0, top_p 0.95; top_p 1.0 makes
  the model ramble. `KV_CACHE_DTYPE=bfloat16` selects the exact BF16 cache at
  half the capacity. The first start autotunes b12x kernels for about 15
  minutes and stores the result under `/cache`.

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
  --restart on-failure --network host --ipc host --shm-size 32g \
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
  The original text/Vision checkpoints (`CHECKPOINT=original`) and their
  remote-code revisions follow the source launcher's pinned revisions; the
  default FP4-CSF checkpoints are described [below](#fp4-csf-checkpoints).
  An explicit model override does not inherit another repository's revision;
  `MODEL_REVISION` and `MODEL_CODE_REVISION` remain operator controls.
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

### GLM-5.3-Flash expert precision

GLM's routed FP4 experts run on B12X with BF16 activations (W4A16) and FP32
router weights by default. Three options choose the precision; each sets the
B12X or vLLM variables shown, and a variable you set yourself is kept and
decides its option:

| Option | Values (default first) | Variables |
| --- | --- | --- |
| `EXPERT_ACTIVATIONS` | `bf16` (W4A16), `fp4` (W4A4, faster, less exact) | `VLLM_B12X_MOE_FP4_FORCE_A16` 1/0 |
| `ROUTER_WEIGHTS` | `fp32`, `bf16` (only with `bf16` activations) | `B12X_W4A16_FP32_TOPK_WEIGHTS` 1/0 |
| `PREFILL_ACTIVATIONS` | `a4` (only with `bf16` activations and a QAD checkpoint), `a16` | `B12X_W4A16_A4_PREFILL` 1/0 (older images: `B12X_W4A16_A4_PREFILL_MIN_TOKENS` 1536/0) |

`a4` runs the prefill rows of every step with NVFP4 activations over the same
packed FP4 weights (images before vLLM #977 only did so for expert calls of
at least 1,536 tokens). On two RTX PRO 6000 Max-Q with the `glm53-tp2`
preset, 8K and 32K-token prefills ran 27% faster with `a4` than with `a16`
(9,528 and 9,784 against 7,490 and 7,728 tok/s), at the same decode speed; the
needle-checksum near-miss rate rose from 0.53% to 1.75%. A second NVFP4
activation plane for the residual (B12X's `B12X_W4A16_A4_PREFILL_TERMS=2`) gave
a smaller speedup without fewer near misses (1.85%), so it is not offered as an
option.
Decode always stays W4A16 (BF16 activations), in every mode and with no extra
variable: vLLM runs the decode and MTP verification rows of every step on
W4A16, and steps replayed from a CUDA graph keep W4A16 for all rows. Choosing
by row type also prefilled a 1,154-token prompt 11% faster than the call-size
threshold did (6,498 against 5,868 tok/s), with the same needle-checksum
results. `a4` needs an image whose B12X and vLLM include
that prefill path, and `EXPERT_ACTIVATIONS=bf16`; with `fp4` an explicit `a4`
is refused. The options apply to the B12X MoE backend, so the
GLM TP3 preset, which runs FlashInfer CUTLASS experts, has none of them.

### FP4-CSF checkpoints

Qwen3.8-Flash-Next, GLM-5.3-Flash, GLM-5.3 (744B), DeepSeek-V4.1-Flash,
DeepSeek-V4-Flash and DeepSeek-V4-Flash Vision serve their FP4-CSF checkpoints
by default
(`checkpoint: csf`). An FP4-CSF checkpoint holds the same weights as the
original with losslessly compressed routed-expert scales. B12X reads the
compressed scales while it runs the experts, and they stay compressed in GPU
memory, which leaves more room for the KV cache. The downloads are smaller too.

| Model | FP4-CSF (default) | Original |
| --- | --- | --- |
| Qwen3.8-Flash-Next | `local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD` | `local-inference-lab/Qwen3.8-Flash-Next-NVFP4` (QAD) |
| GLM-5.3-Flash | `local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD` (MXFP8 attention and shared experts) | `local-inference-lab/GLM-5.3-Flash-NVFP4` (QAD) |
| GLM-5.3 (744B) | `local-inference-lab/GLM-5.3-NVFP4-CSF` | `local-inference-lab/GLM-5.3-NVFP4` at `1f3bb90c` (keeps the BF16 MTP layer unquantized) |
| DeepSeek-V4.1-Flash | `local-inference-lab/DeepSeek-V4.1-Flash-lossless-CSF` | `deepseek-ai/DeepSeek-V4.1-Flash` |
| DeepSeek-V4-Flash | `local-inference-lab/DeepSeek-V4-Flash-0731-lossless-CSF` | `deepseek-ai/DeepSeek-V4-Flash-0731` at `9e165c30` |
| DeepSeek-V4-Flash Vision | `local-inference-lab/DeepSeek-V4-Flash-Vision-Exp-lossless-CSF` | `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` at `6821d6ad` |

Measured on RTX PRO 6000 Max-Q (325 W) with each profile's settings, FP4-CSF
against the original checkpoint (decode: aggregate output tok/s at the listed
concurrency, two runs; prefill: one 8K and one 32K prompt):

| Model (GPUs) | KV cache tokens | Decode | Prefill 8K / 32K |
| --- | --- | --- | --- |
| Qwen3.8-Flash-Next (1) | 750,354 vs 551,925 (+36%) | C1 and C8 within 0.5%; C16 1,079 vs 878 | -2.4% / -1.9% |
| GLM-5.3-Flash `glm53-tp2` (2) | 2,073,850 vs 871,556 | C1 171 vs 176, C8 536 vs 536 | -0.5% / -0.8% |
| GLM-5.3-Flash (4) | 6,871,032 vs 6,165,626 (+11%) | C1 +8.4%, C8 +9.1%, C16 +7.5%, C32 +6.1% | +2.6% / +3.3% |
| DeepSeek-V4.1-Flash (4) | 12,951,057 vs 9,566,426 (+35%) | C8 -3.1%, C16 -3.5%, C32 -1.2% | -1.3% / -1.7% |
| DeepSeek-V4-Flash (2) | 1,853,797 vs 1,308,256 (+42%) | C4 +0.9%, C8 -0.8% | +0.7% / +2.2% |
| DeepSeek-V4-Flash Vision (2) | 1,850,842 vs 1,306,708 (+42%) | C1 +4.2%, C2 -0.2%, C4 +3.1% | -1.4% / -2.9% |
| GLM-5.3 (744B) `glm53-csf-tp8` (8)* | 535,808 vs 489,920 (+9.4%) | steps/s C1 -2.0%, C2 -0.2%, C4 -0.3% | -2.2% / -1.1% |

The GLM-5.3-Flash TP4 row compares the QAD MXFP8 FP4-CSF checkpoint with the
BF16 original, both with a4 prefill and 96% of GPU memory, on RTX PRO 6000
Workstation Edition GPUs (600 W). It loads 4.3 GiB less weight per GPU (42.64
against 46.93 GiB), and its MXFP8 attention and shared experts carry most of
the decode gain.

The DeepSeek-V4.1-Flash decode cost comes from expanding the compressed scales
of every routed expert in each layer before the B12X W4A8 kernels run.

\* GLM-5.3 (744B) was measured on RTX PRO 6000 Workstation Edition GPUs
(600 W) with MTP-normalized decode steps per second, both checkpoints with the
`glm53` profile. At TP8 each GPU holds only 256 of an expert's 2,048
channels, so the single-request MoE calls have little work to hide the
rebuild of the compressed scales in each pipeline stage; the profile runs two
CTAs per SM for small batches (`B12X_W4A16_SMALL_M_OCCUPANCY=2`, which makes
the original 1-4% faster as well). `B12X_NVFP4_CSF_INLINE_WORDS=32` keeps the
first replacement scale words inline in the compressed records: C1 -1.3% for
about 1.1 GiB per GPU, 508,608 KV tokens. `PROFILE=glm53 CHECKPOINT=original`
trades the extra KV cache for the last percent of decode speed.

Qwen3.8-Flash-Next's C16 difference is capacity, not kernel speed. Each
running request holds about 44K tokens of KV cache even at context zero: one
attention page of 3,008 tokens plus the recurrent (GDN) states that MTP
verification keeps. With the original checkpoint only 12 of the 16 request
slots fit (4 wait); the FP4-CSF checkpoint's larger cache runs all 16. The
time per step is the same.

- `-e CHECKPOINT=original` serves the original checkpoint with the profile's
  ModelOpt or DeepSeek settings, at the revision shown.
- `MODEL` naming either checkpoint selects it. A local FP4-CSF copy selects
  the FP4-CSF settings from its own files (the `manifest.json` schema, or the
  CSF recipes in a Hugging Face-layout `config.json`). `CHECKPOINT=original`
  refuses such a copy: the original settings would read its compressed scales
  as plain ones and serve wrong output without an error. Any other `MODEL` or
  `MODEL_REVISION`, such as an older revision of the original repository,
  keeps the original settings.
- The launcher downloads an FP4-CSF repository itself. It then gives vLLM a
  directory under `/tmp/lil-csf` with the repository's metadata files and a
  `config.json` whose quantization (`nvfp4_csf` or `mxfp4_csf`) points at the
  downloaded weights. vLLM gets neither a Hub revision nor, for the DeepSeek
  profiles that trust remote code, a code revision for that directory. MTP
  and DSpark drafters stored in the checkpoint are read from it too.
- Offline (`HF_HUB_OFFLINE=1`) or without network, a pinned revision that is
  not in the cache is served from a cached revision with the same
  `manifest.json`, which names every weight and metadata file by SHA-256.
  Hub revisions that change only the model card or license files keep the
  manifest, so a newer image does not need a new download for them.
- An image whose vLLM cannot read a model's FP4-CSF checkpoint serves the
  original with a warning. The launcher checks the installed vLLM's FP4-CSF
  load formats and, for DeepSeek-V4-Flash and its vision variant, the
  `deepseek_v4_flash` family of its MXFP4-CSF loader. An explicit choice
  (`CHECKPOINT=csf`, or `MODEL` naming the FP4-CSF repository, as the
  generated Compose files do) fails instead; use `CHECKPOINT=original` with
  the original `MODEL`.
- FP4-CSF needs B12X MoE without expert parallelism, so the GLM TP3 preset
  serves the original checkpoint. A preset that names a checkpoint of its
  own, such as `glm53-tp2` (FP4-CSF), serves that checkpoint and refuses a
  `CHECKPOINT` of the other kind.

GLM and DeepSeek retain the source launchers' temperature 1/top-p 0.95
server defaults. Explicit generation configuration replaces these defaults;
request sampling remains authoritative. GLM retains `reasoning_effort=high`
and `clear_thinking=false`. No profile changes target weight or KV precision
to manufacture a speedup.

GLM serves `runtime/templates/glm53-flash.jinja`: the checkpoint's chat
template with assistant answers rendered exactly as the model generated them.
The checkpoint's template trims whitespace from a previous answer, so an answer
that ended in a newline no longer matched the tokens the model produced, and
the next turn recomputed the whole previous response instead of reusing its
cached state (in one measured two-turn chat, 77% instead of 99.9% of the prompt
was reused). Answers without surrounding whitespace render exactly as before.
`-e CHAT_TEMPLATE=checkpoint` restores the checkpoint's template, and any other
value is passed to vLLM as `--chat-template`.

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

## Tokenizer backend

The image includes [fastokens](https://github.com/crusoecloud/fastokens) 0.3.2,
a Rust implementation of the Hugging Face `tokenizers` backend, and every
profile enables it (`VLLM_USE_FASTOKENS=1`): vLLM loads its Hugging Face
tokenizers through fastokens, so prompts are tokenized and output is detokenized
by fastokens, while tokenizer classes, chat templates and the reasoning and tool
parsers stay the same. Set `-e VLLM_USE_FASTOKENS=0` to go back to the standard
backend. With fastokens enabled the log shows `[fastokens] patch_transformers:
successfully patched transformers`.

vLLM tokenizes the whole prompt on every request before prefill starts, also
when the prefix cache already holds it, so this time adds directly to time to
first token in long agent conversations. On an EPYC 9575F a 300K-token agent
conversation took 450 ms with the standard backend and 27 ms with fastokens
(1M tokens: 1.5 s and 85 ms). fastokens spreads one long prompt over CPU
threads; restricted to a single CPU it still took only 58 ms (1M tokens:
180 ms). A slower CPU takes proportionally longer with either backend.

Both backends gave identical token IDs, decoded text and streamed output for
every profile's tokenizer on multilingual text, code, chat conversations with
tools and reasoning, and prompts of up to 1M tokens. They differ in these rare
cases; set `VLLM_USE_FASTOKENS=0` if one of them matters for your clients:

- fastokens reuses the regular-expression splits of the previous prompt on the
  same thread when two prompts without special tokens share at least 4 KiB. When
  the new prompt continues differently right after whitespace, it can be split
  differently (GLM, Qwen, MiMo). vLLM's rotating pool of tokenizer copies
  prevented this in every serving-path test, and chat prompts contain special
  tokens, which disable the reuse.
- Qwen3.8 and MiMo normalize text to NFC. fastokens uses Unicode 17 data for
  this, the standard backend older data, so combining marks added in Unicode 10
  or later (128 code points, for example U+0898-U+089F) next to other combining
  marks can be tokenized differently.
- DeepSeek: U+180E MONGOLIAN VOWEL SEPARATOR next to a space is tokenized
  differently. fastokens also counts added tokens in the tokenizer's vocabulary
  size; vLLM (vllm #934) derives the largest valid prompt token ID so that ID
  129280, one past the DeepSeek vocabulary, is still rejected.
- Character offsets are not available, so `return_token_offsets` on the render
  endpoints enabled by `--enable-scale-out` fails.

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

## Replicas: several TP1 servers behind one endpoint

`REPLICAS=N` (or `replicas: N` in a settings file, or the `qwen38-tp1x2`
preset) runs N complete single-GPU servers in the container, one per visible
GPU, instead of one server split across GPUs:

```bash
docker run -d --name qwen38-tp1x2 --restart on-failure --init \
  --gpus '"device=0,1"' --network host --ipc host \
  -v /root/.cache/huggingface:/root/.cache/huggingface -v lil-cache:/cache \
  -e PRESET=qwen38-tp1x2 -e PORT=8000 "$LIL_IMAGE"
```

- Each replica gets one GPU, in `CUDA_VISIBLE_DEVICES` order, and a loopback
  port. Every option applies per replica, so `MAX_NUM_SEQS` and the KV size are
  per GPU. `TP`, `DCP` and pipeline parallelism must stay 1.
- The public port is a proxy. It records which replica served each request's
  whole history (chat, messages or Responses input, or a completion prompt).
  The next turn of a conversation extends that history, so it goes to the same
  replica, and its prefix cache and recurrent checkpoints stay on one GPU.
  First turns, and identical prompts sent side by side, go to the replica with
  the fewest requests in flight. Responses follow-ups with
  `previous_response_id` go to the replica that produced the previous response,
  and requests with the same `X-LIL-Affinity` header value share a replica.
  Other requests, such as `/v1/models`, go to replica 0.
- `/health` returns 200 once every replica is ready, and `/metrics` carries
  each replica's metrics with a `replica` label.
- With `CACHE_MODE=lmcache`, all replicas use one LMCache service.
- Qwen3.8-Flash-Next keeps one host-RAM copy of its 26.8 GiB PLE table for
  all replicas (`VLLM_PLE_TABLE_MEMORY=shared` under `/dev/shm/lil-ple`, keyed
  by the checkpoint content). The first replica loads it while the others
  wait, then they map it. With `--ipc host` the table outlives the container,
  so the next start maps it instead of loading it; it uses host RAM until
  removed. `docker exec <container> python -m
  vllm.models.qwen4_exp.nvidia.ple_shared_table list` shows the tables and
  `prune --keep <key>` removes the others.
- When a replica or the proxy exits, the others stop and the container exits,
  so `--restart on-failure` restarts all of them.
- Benchmark through the proxy: `lil-bench --url http://127.0.0.1:<PORT>`.

Two TP1 replicas avoid the cross-GPU communication of every TP2 layer and keep
separate KV caches; a single request still gets only one GPU. How much that
pays depends on what TP2 communication costs on the host. On two PCIe RTX PRO
6000 Max-Q GPUs, awsdl measured 12-20% more decode throughput than TP2 at 8-16
concurrent requests, with the same single-request speed. On two PCIe RTX PRO
6000 Server Edition GPUs (MTP, no context, two runs each), two replicas matched
TP2 at 8 requests (1016 vs 1012 tok/s), were 2% ahead at 16 (1531 vs 1497) and
22% slower for a single request (187 vs 240 tok/s).

## Benchmark a running container (`lil-bench`)

The runtime image ships llm-inference-bench at the commit pinned in
`tools/jovian_wheel_runtime/lil-bench.lock.json` (archive checksum locked),
installed by `install_lil_bench.py` under `/opt/lil/bench` with p2pmark built
for sm_100/sm_120/sm_121. With the model loaded and no other traffic:

```bash
docker exec -it -e LIL_BENCH_TOKEN=lilb_... <container> lil-bench
```

It reads the serving command from the vLLM process, records hardware and PCIe
topology, runs p2pmark and the standard prefill/decode matrix while sampling
GPU clocks and throttle reasons, saves the result to `/cache/lil-bench`, and
uploads it to docker.local-inference-lab.ai. The identifier comes from
<https://docker.local-inference-lab.ai/bench/token>; without it the command
stops with instructions. `lil-bench --no-upload` measures locally.

### Tool-calling quality (`--tool-eval`)

The bench also runs [tool-eval-bench](https://github.com/SeraphimSerapis/tool-eval-bench)
with the community leaderboard settings (`--hardmode --seed 42 --temperature 0
--parallel 4 --max-turns 30 --timeout 600`) against the running server:

```bash
docker exec -it -e LLM_BENCH_CACHE_DIR=/cache/llm-bench <container> \
  python3 /opt/lil/bench/llm_decode_bench.py --tool-eval --port <PORT>
```

The first run installs the pinned tool-eval-bench commit from GitHub into its
own virtual environment (a few seconds, network required); the serving venv is
never modified. `LLM_BENCH_CACHE_DIR=/cache/llm-bench` keeps that environment
on the runtime volume, so a recreated container reuses it. A run takes about
1.5 minutes on GLM-5.3-Flash; at four-way parallelism single runs vary by a few
points, so compare means of several runs.

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
docker run -d --name qwen38-native --gpus '"device=0,1"' --restart on-failure \
  --network host --ipc host \
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
| GLM atomic recurrent cache | Target/recurrent/draft all-rank bundles, request/SYSTEM boundaries, identity checks and restart restore | Implemented for text and image/video-bearing requests (placeholders keyed by content hash); TP2 requires engine-driven request-boundary transfer. TP4/TP8 also accept aligned transfer. |
| DS4 engine-driven cache | Worker-owned pinned SHM, CPU-only service, RAM/disk storage and coordinated shutdown | Implemented for text and authenticated image-bearing prefixes. |
| Qwen atomic recurrent cache | Complete target/GDN/draft checkpoint bundles | Implemented for text with engine-driven transfer; image-bearing requests use the same content-hash placeholder keys (GPU-validated on GLM-5.3-Flash only). |
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

`PRESET=glm53-tp2` starts LMCache with the RAM tier by default. To add the
disk tier there, or to size the tiers yourself, set for example:

```bash
-e LMCACHE_MODE=disk -e LMCACHE_L1_GB=32 -e LMCACHE_L2_GB=256
```

Use `LMCACHE_MODE=ram` to omit disk storage. Keep `/cache` on a persistent
Docker volume for disk restore. At startup the disk tier is capped to 90% of
what its filesystem can give it (free space plus what the tier already holds),
with a warning, so it cannot fill the disk. A default RAM tier shrinks the same
way to what `/dev/shm` and the available RAM can hold; an explicit
`LMCACHE_L1_GB` must fit as given. The RAM arena is freed when the container
stops. One left in `/dev/shm` by a container that was killed or crashed is
removed at the next start; an arena whose container is still running is
refused.

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

- `always` writes every checkpoint to disk. It is the default for models
  without request-boundary checkpoints.
- `on-evict` (default for GLM-5.3-Flash and Qwen3.8) writes the current
  checkpoint of a conversation once, when it
  leaves RAM, and again for everything still only in RAM when the container
  stops. Checkpoints that a newer turn of the same conversation has
  superseded are never written. Disk writes fall from every request to about
  one checkpoint per conversation that goes idle, so the disk holds the
  newest state of many more conversations. Give the container time to stop:
  `docker run --stop-timeout 60` or Compose `stop_grace_period: 60s` (the
  generated Compose files set it). At SIGTERM LMCache first keeps serving
  the checkpoint stores the model is still copying (a few seconds at most),
  then writes the checkpoints that are only in RAM, current ones first,
  within 30 s. With Docker's default 10 s the current checkpoints are
  written first; the log names what was left, and after the restart lookups
  skip those checkpoints instead of failing a restore. A crash or
  `docker kill` loses checkpoints that were only in RAM.
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
for TP2. Image-bearing requests restore external recurrent checkpoints too:
each image's placeholder positions are keyed by its content hash (and the
vision precision), so a checkpoint is never restored for a different image
and an image at the start of a conversation no longer blocks reuse of the
text after it. The auto-fit context
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
