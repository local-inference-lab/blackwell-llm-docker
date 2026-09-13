# Jovian CUDA 13.3 wheel runtime

Status: **research-only**

The Jovian CUDA 13.3 wheel runtime defines one source-locked Python environment
for container and direct-host vLLM serving. The environment uses Python 3.12,
CUDA 13.3, PyTorch 2.13.0, the C++11 ABI, and NVIDIA compute capability 12.0.
Every package artifact has a SHA-256 digest and a semantic source identity.

The runtime has three independently published parts:

- the Python foundation published by `blackwell-llm-docker`;
- the native NCCL runtime published by `nccl-canonical`;
- application wheels for vLLM, B12X, LMCache, and FlashInfer.

A deployment manifest selects one immutable release of each part. Docker and
direct-host installation consume the same manifest and wheel bytes. The NVIDIA
driver and CUDA userspace remain native host or container-base requirements.

## Python foundation artifact

The workflow in `.github/workflows/jovian-wheel-runtime-release.yml` publishes
six wheels declared by `tools/jovian_wheel_runtime/foundation.lock`:

- PyTorch 2.13.0;
- TorchVision 0.28.0;
- XGrammar 0.2.5;
- NVIDIA Triton 3.7.1;
- Triton Kernels 1.0.0;
- FlashAttention 2.7.4.

PyTorch, TorchVision, and XGrammar are recovered as their exact original wheel
payloads from the immutable OCI source image. Their paths and original SHA-256
digests must both match the lock.

The qualified OCI build created and deleted the NVIDIA Triton, Triton Kernels,
and FlashAttention wheel files inside one filesystem layer. Docker layer
history therefore cannot recover those three original archives. The publisher
verifies every hashed installed file from each package's `RECORD`, excludes
installation-generated scripts, metadata, and bytecode, and creates a
deterministic wheel from the qualified installed payload. The release manifest
labels these artifacts `hash-verified-installed-distribution-repack`; it does
not claim byte identity with the unavailable source wheels.

The repacked wheels are installed into an isolated target directory and their
package versions and imports are verified before publication. A repeated build
must produce byte-identical wheels and an identical provenance manifest.

## NCCL runtime artifact

The package `local-inference-nccl-cu133` contains the LIL NCCL 2.31.2 shared
library and complete generated header tree. `nccl4py` remains a separate Python
binding and does not select the native NCCL library.

The launcher must resolve the installed library path and set both variables
before importing PyTorch:

```bash
NCCL_SO=$(local-inference-nccl-path)
export LD_PRELOAD="${NCCL_SO}"
export VLLM_NCCL_SO_PATH="${NCCL_SO}"
exec /path/to/venv/bin/python -m vllm.entrypoints.cli.main serve MODEL
```

This ordering is an invariant. Setting the variables after importing PyTorch
can leave a different NCCL library resident in the process.

## Installation contract

GitHub Releases provides immutable direct-download URLs rather than a PEP 503
package index. Each release contains a `requirements-github.txt` file with
direct URLs and hashes. An offline archive contains the same wheels and a local
requirements file.

The foundation installer creates an isolated environment without
`--system-site-packages`:

```bash
./install_foundation.sh /opt/local-inference/venvs/jovian
```

Application and NCCL installers add packages to that environment only after
the CUDA/Python ABI fields in all manifests agree.

Direct-host execution additionally requires:

- an NVIDIA driver compatible with CUDA 13.3;
- CUDA 13.3 userspace and compiler;
- cuDNN 9.24.0.43;
- cuSPARSELt 0.9.1.1;
- the Open MPI `libmpi.so.40` ABI required by the locked PyTorch wheel.

The native dependency preflight and end-to-end Qwen launch are not yet
qualified. A foundation-only installation is therefore not a supported serving
environment.

## Build isolation and caching

The `lil-wheel-builder` self-hosted runner uses a dedicated rootless Docker
daemon and BuildKit worker. The worker is limited to 64 logical CPUs, a 240 GiB
memory-high threshold, a 256 GiB hard memory limit, no swap, and 8,192 tasks.
Its persistent package-download and compiler-object caches are isolated from
the production Docker daemon.

The organization runner group admits only the declared build workflows in the
`flashinfer`, `vllm`, `b12x`, `LMCache`, `nccl-canonical`, and
`blackwell-llm-docker` repositories. Organization-scoped runner registration
requires a token with GitHub organization runner-administration permission.
The repository-scoped frank2 runner cannot execute jobs from the other five
repositories until that registration is completed.
