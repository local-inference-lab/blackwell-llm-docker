# GLM Spark tensor-parallel serving on two GPUs

Status: **implemented**. The launcher-only diagnostic image
`sha256:5f3b9f8e3ef6953c79d6f8822a6bb6509a5316c7bb0b932a4c72fbdcfed04bf8`
qualifies the FP8 TP2/DCP2 MTP3 external-checkpoint path described below on
two RTX PRO 6000 Workstation GPUs at stock clock offsets. It is not a
published release or a qualification of other GPU models.

`serve-glm-spark-tp2.sh` serves
`local-inference-lab/GLM-5.3-Flash-NVFP4-Spark` with tensor parallelism 2,
decode-context parallelism 2, MTP depth 3, a 3072-token scheduler budget,
four request slots, native vision and full-and-piecewise CUDA graphs.
Attention, MoE and linear backends are B12X. Target KV is FP8; the target
LM head remains BF16 with a private NVFP4 proposal head. NVFP4 MTP experts
use B12X W4A16; this profile does not imply W4A4 for every operator.

## GPU memory and context capacity

`KV_CACHE_MEMORY_BYTES=auto` permits vLLM to profile model execution,
persistent allocations and CUDA graphs before sizing KV. The default
`GPU_MEMORY_UTILIZATION=0.985` applies to that automatic calculation.
`MAX_MODEL_LEN=-1` fits the context to the resulting pool, bounded by the
checkpoint's context limit. A supplied positive context limit must fit.

An explicit `KV_CACHE_MEMORY_BYTES` or `--kv-cache-memory-bytes` bypasses
automatic KV sizing. Reducing GPU utilization does not shrink that fixed
allocation. Budgeting almost all physically free memory for KV can leave
insufficient space for a later projection or vision allocation. Internal
cuBLAS errors can follow allocator OOMs; inspect preceding log messages
instead of identifying every cuBLAS error as the same defect.

On the qualified Workstation pair, automatic profiling selected 3.01 GiB
KV per rank and a 733184-token context. A cold 732928-token prompt containing
one image completed correctly, with at least 1.006 GiB free per GPU in
sampled telemetry. These are measured conditions, not a universal minimum
reserve or a one-million-token capacity guarantee. Fixed 4 GiB KV can fit
a larger context, but the same fixed budget is not portable across hosts.

## Engine-driven host and filesystem cache

Set `CACHE_MODE=lmcache`. The TP2 profile delegates to the cache-complete
launcher and requires `LMCACHE_TRANSFER_MODE=engine_driven`. GPU workers own
the asynchronous copies; the LMCache sidecar has an empty
`CUDA_VISIBLE_DEVICES` and does not create a GPU context. The target cache
remains FP8. Do not bypass this entrypoint with a different LMCache wrapper.

Attention pages contain 2048 tokens per DCP rank. LMCache objects cover
4096 global tokens. Request-boundary bundles contain their own recurrent
endpoint, so the 3072-token model budget need not equal the storage-object
size. Aligned transfers retain the equal-budget and page-alignment checks.

`LMCACHE_L1_SIZE_GB` controls a preallocated pinned shared-memory arena,
defaulting to 64 GiB. It is not a lazy-growth limit. Worker mappings refer
to the same physical pool; they do not each allocate another 64 GiB.
Select a size that leaves host shared-memory capacity for transfer buffers
and other services. `LMCACHE_L2_ENABLED=0` disables filesystem storage;
otherwise `LMCACHE_L2_ROOT` selects its root directory.

The diagnostic image completed these conditions with a 64 GiB L1 pool:

| Condition | Result |
|---|---|
| 262144-token cold prompt | 30.273 s |
| RAM restore of that prompt | 0.276 s; all prompt tokens restored, zero recompute |
| Restore after both services restart | 0.347 s; all prompt tokens restored, zero recompute |
| 54643-token literal document | Exact answers from cold execution, GPU-local cache, RAM and filesystem |
| Shared leading instructions with a different user turn | 11340 tokens restored; only the 11-token suffix computed |

Filesystem timings include the OS page cache; they do not measure cold
physical-disk bandwidth. The public API binds `0.0.0.0`; administrative
LMCache listeners remain loopback-only unless explicitly configured.

## Image admission

The default `--limit-mm-per-prompt '{"image":1,"video":0}'` counts images
across one complete API request, including images repeated in conversation
history. It is not a lifetime limit across independent requests.

The native override `--limit-mm-per-prompt '{"image":2,"video":0}'` passed
three independent ordered-transcription requests with two 400×80 PNG images
on the diagnostic image. The one-image default correctly rejected the same
two-image input with HTTP 400 and remained healthy. Both boot profiles fitted
a 733184-token context on the tested pair. This qualifies basic admission and
encoding, not simultaneous maximum-resolution images at maximum context.
Requalify that combined workload before relying on its memory capacity.

## Compatibility and validation

Explicit native CLI memory and scheduling options remain authoritative.
Cache geometry and transfer options owned by the LMCache wrapper use its
documented environment controls. No clock settings, checkpoint weights,
sampling policy or native kernels are changed by this launcher.

Run launcher contracts with:

```bash
uv run --no-project --with pytest python -m pytest -q \
  tests/test_glm_spark_tp2_launcher.py tests/test_lmcache_http_launcher.py
```

The build must install these source launchers and regenerate the source lock.
A diagnostic overlay's component lock does not authenticate modified launcher
files and must not be published as a source-locked release.
