# GLM Spark tensor-parallel serving on two GPUs

Status: **implemented**. The diagnostic image
`sha256:314f3adb479ea6fba80c5953dcec00c9196a25c3dd0dc6c96abb4bcde839dfa6`,
with the explicit memory settings below, qualifies FP8 TP2/DCP2 MTP3 serving on
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

The default requests `MAX_MODEL_LEN=1048576` with a fixed
`KV_CACHE_MEMORY_BYTES=4294967296` budget per GPU. It does not silently reduce
the context to fit the host. The KV pool is shared across requests; this is
not four independent one-million-token allocations.

`CUBLAS_WORKSPACE_CONFIG=:4096:1` limits cuBLAS workspace to 4 MiB per handle.
The setting does not quantize weights, activations, KV or recurrent state.
Operators can override it, but a larger workspace consumes memory outside
the fixed KV budget and requires requalification on a tightly packed GPU.

`KV_CACHE_MEMORY_BYTES=auto` explicitly enables vLLM's memory profiling.
`GPU_MEMORY_UTILIZATION=0.985` applies only to that calculation. To also
permit a shorter context, explicitly set `MAX_MODEL_LEN=-1`. Automatic
profiling with a positive context limit must still fit that complete limit.

An explicit `KV_CACHE_MEMORY_BYTES` or `--kv-cache-memory-bytes` bypasses
automatic KV sizing. Reducing GPU utilization does not shrink that fixed
allocation. Budgeting almost all physically free memory for KV can leave
insufficient space for a later projection or vision allocation. Internal
cuBLAS errors can follow allocator OOMs; inspect preceding log messages
instead of identifying every cuBLAS error as the same defect.

Status: **qualified** for GPU-local capacity on the two stock-clock Workstation
GPUs using diagnostic image
`sha256:314f3adb479ea6fba80c5953dcec00c9196a25c3dd0dc6c96abb4bcde839dfa6`
with the explicit memory settings above. vLLM reported 1,051,958 usable KV
tokens after hybrid-state accounting. A cold 1,048,320-token input including
native image features returned a correct visual answer, with a 256-token
output allowance and at least 230.31 MiB physical free memory in sampled
telemetry. This is not a qualification of other GPU models or unrestricted
image counts.

In a matched stock-clock comparison, the 4 MiB cuBLAS workspace reduced
startup live allocation by 140 MiB per rank. Cold 32K prefill was 10,642.33
versus 10,643.94 input tokens/s. Three warmed C1 cells had median verifier
throughput 70.176 versus 70.224 steps/s, with overlapping ranges and no
errors. The three cuBLAS-limited cells measured 70.2031, 70.1757 and 70.1521
steps/s; the control measured 70.3548, 70.2239 and 70.1606. Each cell uses
15 seconds of warmup and 30 seconds of measurement at temperature 1.0 and
top-p 0.95. These results do not establish a speedup.

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

Status: **qualified** for the text checkpoint cases below, with fixed 4 GiB
KV per GPU, a 1,048,576-token context and a 64 GiB L1 pool:

| Condition | Result |
|---|---|
| 1,048,320-token cold prompt | 136.270 s |
| RAM restore of that prompt | 0.812 s; all prompt tokens restored, zero recompute |
| Restore after both services restart | 1.329 s; all prompt tokens restored, zero recompute |
| 54643-token literal document | Exact answers from cold execution, GPU-local cache, RAM and filesystem |
| Shared leading instructions with a different user turn | 11340 tokens restored; only the 11-token suffix computed |
| 1,048,320-token input with native image features | Correct visual answer in 152.518 s; minimum sampled free GPU memory 106.31 MiB |
| Three decoders plus a cold 32K image after the long image request | Correct image answer, progress from all decoders, zero errors |
| 9751/9752/9753-token cache and 536-token continuation checks | 27 checks passed after the long image request |

Filesystem timings include the OS page cache; they do not measure cold
physical-disk bandwidth. The public API binds `0.0.0.0`; administrative
LMCache listeners remain loopback-only unless explicitly configured.
The long probe uses deterministic token IDs and one output token to check
transfer attribution and output equality; it is not a language-quality test.
The separate literal-document and shared-instruction cases check exact answers.

## Image admission

The default `--limit-mm-per-prompt '{"image":1,"video":0}'` counts images
across one complete API request, including images repeated in conversation
history. It is not a lifetime limit across independent requests.

The native override `--limit-mm-per-prompt '{"image":2,"video":0}'` passed
three independent ordered-transcription requests with two 400×80 PNG images
on launcher diagnostic image `sha256:5f3b9f8e3ef6953c79d6f8822a6bb6509a5316c7bb0b932a4c72fbdcfed04bf8`
with automatic KV sizing and a 733184-token context. The one-image default
correctly rejected the same two-image input with HTTP 400 and remained healthy.
This qualifies basic admission and
encoding, not simultaneous maximum-resolution images at maximum context.
Requalify that combined workload before relying on its memory capacity.

Exact multimodal request-endpoint restore is **unsupported** by the semantic
boundary adapter. Aligned fallback may restore most of an image prompt while
recomputing its tail. A correct visual answer and a successful capacity test
must not be described as a zero-recompute multimodal checkpoint restore.

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
