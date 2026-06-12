import importlib.util
import torch, flashinfer, transformers, sglang, b12x
print(f'PyTorch:      {torch.__version__}')
print(f'CUDA:         {torch.version.cuda}')
print(f'torch NCCL:   {torch.cuda.nccl.version()}')
print(f'FlashInfer:   {flashinfer.__version__}')
print(f'Transformers: {transformers.__version__}')
print(f'SGLang:       {getattr(sglang, "__version__", "editable-install")}')
print(f'b12x:         {getattr(b12x, "__version__", "?")}')
assert torch.__version__.startswith('2.12.0+cu132'), torch.__version__
# Verify the patched NCCL via ctypes on the actual .so files (deterministic;
# torch.cuda.nccl.version() can report the compiled-against version instead)
import ctypes, glob
nccl_libs = glob.glob('/opt/venv/lib/python3.12/site-packages/**/libnccl.so.2', recursive=True)
assert nccl_libs, 'no libnccl.so.2 in venv'
for _p in nccl_libs:
    _lib = ctypes.CDLL(_p)
    _v = ctypes.c_int()
    _lib.ncclGetVersion(ctypes.byref(_v))
    print(f'NCCL at {_p}: {_v.value}')
    # NCCL encodes 2.30.4 as 2*10000 + 30*100 + 4 = 23004
    assert _v.value >= 23004, f'patched NCCL not active at {_p}: {_v.value}'
assert importlib.util.find_spec('sglang.srt.layers.attention.b12x_backend'), 'b12x backend missing'
import sgl_kernel
from sglang.srt.server_args import ServerArgs
print('OK: smoke test passed')
