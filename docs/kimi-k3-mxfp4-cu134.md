# Kimi-K3 MXFP4 on the CUDA 13.4 wheel runtime

Status: implemented; installed-image serving qualification is required before
publication. This recipe targets 16 RTX PRO 6000 Blackwell 96 GB GPUs, tensor
parallelism 16, and decode context parallelism 16. It does not support weight
offload or the QSRT checkpoint.

`runtime/serve-kimi-k3.sh` launches installed vLLM and B12X packages. It does
not mount or import a source checkout. The image uses the component bundles
and NVIDIA foundation described in [the wheel runtime specification](jovian-cu134-wheel-runtime.md).
The assembler records each component's immutable commit and wheel hashes.
FlashInfer can be reused without recompilation when its source and ABI match.

The launcher selects the official `moonshotai/Kimi-K3` checkpoint at revision
`2496450e92e425c886db095102a52a6682ca3970`. Routed experts retain MXFP4 W4A16;
the configured KDA and vision projections are converted to MXFP8 at load time.
InstantTensor uses AIO, a 512 MiB tensor ring, and I/O queue depth 16. The ring
size alone does not bound the separate I/O staging allocation. Depth 16 leaves
repack headroom after the speculative runner's buffers are constructed.
Vision weights load before
the routed experts. Target KV uses FP8 with native 32 GiB CPU offload. The
default context limit is 950,000 tokens and scheduled prefill chunks contain
4,096 tokens. This context limit is not a measured physical-cache receipt.

```bash
docker run -d --name kimi-k3 --gpus all --network host --ipc host \
  --shm-size 64g --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /root/.cache/huggingface:/root/.cache/huggingface \
  -v /mnt/kimi-k3-cache:/cache/kimi-k3 \
  -e KIMI_SPECULATOR=dspark -e KIMI_PORT=8000 \
  IMAGE_AT_IMMUTABLE_DIGEST bash /opt/lil/runtime/serve-kimi-k3.sh
```

The image entrypoint preserves the packaged NCCL bootstrap. Supply credentials
with an environment file containing `VLLM_API_KEY`; never include the key in
a launcher, published command, or printed configuration.

| `KIMI_SPECULATOR` | Draft checkpoint | GPU KV bytes per rank | Captured rows for one sequence |
| --- | --- | --- | --- |
| `none` | None | 910,000,000 | 1 |
| `dspark` | `Inferact/Kimi-K3-DSpark@cf6b8244620e7ea4b0651d214f28e89eac75bed6` | 1,325,000,000 | 8 |
| `dflash` | `modal-labs/Kimi-K3-DFlash@c192d15a43407bf758b5ae0880d5c72052fef1de` | 1,325,000,000 | 8 |

DSpark uses seven proposals, a 32,768-token draft tail, compact rotary tables,
and a sharded Markov head. DFlash uses seven proposals and block-major KV
storage for its replicated draft attention. Both use online MXFP8 draft
projections with the fused input projections excluded as specified by the
launcher's quantization configuration.

`KIMI_MAX_SEQS` defaults to one. Increasing it expands captured batch sizes and
increases graph, recurrent-state, and transient memory requirements. Admission
at larger concurrency needs separate qualification; the launcher does not
claim that twelve sequences fit with the default cache allocation. Override
`KIMI_KV_BYTES` and `KIMI_MAX_MODEL_LEN` together when changing physical cache
capacity. `KIMI_NATIVE_KV_GIB` changes host offload capacity.

Image input is enabled without an image-count cap. Per-image processing is
limited to 40,960 patches and 512 patches along one side. Kimi-K3 reasoning
and tool parsers are enabled. `KIMI_PRINT_COMMAND=1` prints the resolved command
without loading weights; do not use it with secret CLI arguments.

The `/cache/kimi-k3` volume retains Triton, CuTeDSL, and B12X compilation
artifacts. B12X uses its `b12x-compile` subdirectory for compiled programs and
source-validated preparation decisions. Keep this volume across container
replacements to avoid recompiling unchanged kernels.
