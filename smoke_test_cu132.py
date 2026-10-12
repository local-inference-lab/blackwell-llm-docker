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

# --- ServerArgs.from_cli_args gate (CodeRabbit: moved out of inline Dockerfile
# python). An orphaned dataclass field (added without its CLI arg) raises
# AttributeError here — the exact failure mode that once crashed a deploy at
# server start. Network/config errors in __post_init__ are expected offline.
import argparse
from sglang.srt.server_args import ServerArgs

_parser = argparse.ArgumentParser()
ServerArgs.add_cli_args(_parser)
# --enable-strict-thinking exists only once the cherry-pick layer is applied;
# this script also gates the base image build, so probe before passing it.
_argv = ['--model-path', 'smoke-test-dummy']
if '--enable-strict-thinking' in _parser._option_string_actions:
    _argv.append('--enable-strict-thinking')
_args = _parser.parse_args(_argv)
try:
    ServerArgs.from_cli_args(_args)
except AttributeError as e:
    print(f'FATAL orphaned dataclass field: {e}')
    sys.exit(1)
except Exception as e:
    print(f'OK: field mapping passed (post-init stopped at expected '
          f'{type(e).__name__})')
else:
    print('OK: from_cli_args full round-trip')
