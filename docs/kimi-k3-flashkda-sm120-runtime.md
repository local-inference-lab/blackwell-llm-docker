# Kimi-K3 FlashKDA SM120 runtime

## Scope

This release composes `vllm-project/FlashKDA` dev commit
`ee0be888cd0e972f9409bf53756f8c38c6652173` with an SM120 workspace-transport
fix. It does not narrow FlashKDA's generic state, dtype, fixed-length, varlen,
or checkpoint ABI variants.

The patch is applied as an independently locked source component when the
CUDA 13.3 / PyTorch 2.13 Kimi-K3 image builds its vLLM native extensions.

## Failure

The first FlashKDA prefill on an RTX PRO 6000 Blackwell reserved about 3.8 GiB
outside PyTorch's allocator. A checkpoint-aware build then failed with
`cudaErrorMemoryAllocation`, even after reducing the served context from
372,000 to 320,000 tokens and the explicit KV allocation from 2.0 to 1.75 GiB.

The checkpoint ABI was not the source of the allocation:

- the old and checkpoint-aware generic binaries reserved the same amount;
- K1 alone reserved about 122 MiB;
- K2 alone reproduced the complete allocation on both V-split and default
  recurrence paths; and
- only K2 referenced `__cuda_syscall_cp_async_bulk_unicast` in its SM120
  cubin.

CUDA 13.3 lowers K2's six `SM90_BULK_COPY_G2S` workspace restores to that
driver-provided syscall on SM120. Loading the syscall runtime caused the
multi-GiB context reservation.

## Fix

On SM120, the six separated K1 workspace arrays are described as raw `uint32`
TensorMap tiles. K2 loads those tiles through the existing three-stage TMA
transaction barrier. This preserves the private, swizzled K1-to-K2 byte-image
ABI while removing the syscall dependency.

SM90 and SM100 cubins retain the original native bulk-copy path. The selection
is compile-time on `__CUDA_ARCH__`, so the SM120 cubin cannot contain the
fallback call even when a fat binary also targets earlier architectures.

The focused stage-reuse test uses five 16-token chunks. Four chunks exercise
only the first circular-stage reuse and did not expose an intermediate
cooperative-copy publication race; the fifth chunk did. The final TensorMap
implementation passes Compute Sanitizer racecheck with zero hazards.

## Reproduction

The source lock is
`patches/releases/flashkda-sm120-tensormap-r1/flashkda/integration.lock.json`.
The build verifies the manifest, patch digest, resulting Git tree, image
labels, and absence of the syscall symbol in the packaged extension.

```bash
pytest -q tests/test_flashkda_sm120_release.py
bash -n build-kimi-k3-infernal-invocation-cu133-torch213.sh
docker build --check \
  -f Dockerfile.kimi-k3-infernal-invocation-cu133-torch213 .
```

The GPU qualification build used CUDA 13.3.73, PyTorch 2.13, and `sm_120a`.

## Qualification

- Generic ABI binary SHA-256:
  `5d2466074b29005bc149e9da942e8e33f342af54f031e8055daa9d1daeecbde3`.
- `cuobjdump`: no `__cuda_syscall`; no stack or local-memory spill.
- Compute Sanitizer racecheck: 0 errors and 0 warnings.
- FP32 reference comparisons passed for V-split and default K2 paths,
  checkpoint off/on, and a 3,080-token production chunk with a 3,072-token
  checkpoint. Relative RMSE was 0.0044 to 0.0056.
- Reusable output, final-state, checkpoint-state, and workspace buffers were
  bitwise stable across repeated calls.
- Warm 3,080-token checkpoint call: 0.1283 ms for the generic patched build;
  the old bulk-copy build measured 0.1354 ms.
- First-call multi-GiB driver residency was eliminated.

The production-specialized artifact has SHA-256
`c0b440ab87e6cfec16b23138f95a67d8fe9142836c040a56c41a574dc7d27ed1`.
The qualified image is
`sha256:087276f1ec791909e71055d655eeadb3b35c246e46f2be21af9d210aa970a60b`.

Live TP8/DCP8 validation covered exact normal output, JSON schema output,
four concurrent requests, an 18K client disconnect followed by immediate
retry, a 16,104-token cold prefill and checkpoint replay, and a 32,092-token
cold prefill at approximately 884.6 prompt tokens/s. No request error,
corruption, worker restart, container OOM, or CUDA error was observed.

## Memory envelope

The 372K / 2.0 GiB profile completed its first FlashKDA request, but a later
264,241,152-byte allocation had only 263,585,792 bytes physically free on rank
0. That profile was rejected despite the engine recovering.

The active production envelope is therefore:

- maximum context: 320,000 tokens;
- explicit KV allocation: 1.75 GiB per rank;
- KV capacity: 625,777 tokens; and
- scheduler target/draft budgets: 3,080 / 3,080 tokens.

The validated Triton launcher remains the primary rollback. Reducing
`max_model_len` alone does not create working memory while the KV allocation is
fixed explicitly; both values must be reviewed together.
