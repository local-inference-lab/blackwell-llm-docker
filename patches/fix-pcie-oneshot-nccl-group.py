#!/usr/bin/env python3
"""Pass the NCCL device group to the b12x PCIe oneshot IPC exchange.

b12x >= 0.16 pcie_oneshot does its IPC handle exchange via
dist.broadcast_object_list(device=cuda) and asserts the exchange group is
NCCL-backed. sglang's GroupCoordinator constructs CustomAllreduce with the
gloo cpu_group (vLLM heritage), so setup fails with:
    "PCIe oneshot IPC exchange requires an NCCL process group, got gloo"
and the whole custom allreduce silently falls back to NCCL.

Fix: thread the coordinator's device_group (NCCL) into CustomAllreduce and
use it as the b12x exchange_group. The gloo group keeps serving everything
else (vLLM-style IPC buffer setup for the non-b12x path).
"""
import sys
from pathlib import Path

SRT = Path("/opt/sglang/python/sglang/srt")

# --- 1. custom_all_reduce.py: accept + use device_group ----------------------
ca = SRT / "distributed/device_communicators/custom_all_reduce.py"
src = ca.read_text()

if "device_group" in src:
    print(f"Already patched: {ca}")
else:
    old_init = """    def __init__(
        self,
        group: ProcessGroup,
        device: Union[int, str, torch.device],
        max_size=_MAX_CAR_SIZE,
    ) -> None:"""
    new_init = """    def __init__(
        self,
        group: ProcessGroup,
        device: Union[int, str, torch.device],
        max_size=_MAX_CAR_SIZE,
        device_group: ProcessGroup = None,
    ) -> None:"""
    if old_init not in src:
        print(f"ERROR: __init__ anchor not found in {ca}", file=sys.stderr)
        sys.exit(1)
    src = src.replace(old_init, new_init, 1)

    old_store = """        self.device = device
        self.group = group"""
    new_store = """        self.device = device
        self.group = group
        # NCCL-backed group for b12x pcie_oneshot IPC exchange (b12x >= 0.16
        # rejects gloo); falls back to `group` when not provided.
        self.device_group = device_group"""
    if old_store not in src:
        print(f"ERROR: attribute-store anchor not found in {ca}", file=sys.stderr)
        sys.exit(1)
    src = src.replace(old_store, new_store, 1)

    old_call = """            self._pcie_runtime = runtime_cls.from_exchange_group(
                exchange_group=group,"""
    new_call = """            self._pcie_runtime = runtime_cls.from_exchange_group(
                exchange_group=(
                    self.device_group if self.device_group is not None else group
                ),"""
    if old_call not in src:
        print(f"ERROR: from_exchange_group anchor not found in {ca}", file=sys.stderr)
        sys.exit(1)
    src = src.replace(old_call, new_call, 1)
    ca.write_text(src)
    print(f"OK: patched {ca}")

# --- 1b. custom_all_reduce.py: disable b12x stream-affinity enforcement ------
# b12x >= 0.16 binds each channel to one CUDA stream and raises during sglang's
# graph capture (capture stream != default stream). The single-channel usage
# here matches b12x's own PCIeOneshotAllReducePool(single_channel=True), which
# sets _stream_affine=False — pass the constructor's explicit opt-out.
src = ca.read_text()
if "stream_affine=False" in src:
    print(f"Already patched (stream_affine): {ca}")
else:
    old_args = """                eager_buffer_bytes=self.max_size,
                max_size=self.max_size,
            )"""
    new_args = """                eager_buffer_bytes=self.max_size,
                max_size=self.max_size,
                stream_affine=False,
            )"""
    if old_args not in src:
        print(f"ERROR: from_exchange_group args anchor not found in {ca}", file=sys.stderr)
        sys.exit(1)
    src = src.replace(old_args, new_args, 1)
    ca.write_text(src)
    print(f"OK: patched stream_affine in {ca}")

# --- 2. parallel_state.py: pass device_group at the call site ----------------
ps = SRT / "distributed/parallel_state.py"
src = ps.read_text()

if "device_group=self.device_group," in src:
    print(f"Already patched: {ps}")
else:
    old = """                CAClass = dispatch_custom_allreduce()
                self.ca_comm = CAClass(
                    group=self.cpu_group,
                    device=self.device,
                )"""
    new = """                CAClass = dispatch_custom_allreduce()
                try:
                    self.ca_comm = CAClass(
                        group=self.cpu_group,
                        device=self.device,
                        device_group=self.device_group,
                    )
                except TypeError:
                    # CA variants (HIP quick/mscclpp) without device_group param
                    self.ca_comm = CAClass(
                        group=self.cpu_group,
                        device=self.device,
                    )"""
    if old not in src:
        print(f"ERROR: call-site anchor not found in {ps}", file=sys.stderr)
        sys.exit(1)
    src = src.replace(old, new, 1)
    ps.write_text(src)
    print(f"OK: patched {ps}")

# --- 3. parallel_state.py: re-wire the crossover autotune ---------------------
# The Apr-13 fork rebuild lost the block (present at f7a239ac) that calls
# ca_comm.find_crossover_size() after construction when max-size='auto'.
# Without it the b12x runtime keeps max_size=1MB, including the 128KB-1MB
# range where NCCL beats the oneshot kernel on this PCIe topology.
src = ps.read_text()
if "find_crossover_size" in src:
    print(f"Already patched (crossover): {ps}")
else:
    anchor = """                except TypeError:
                    # CA variants (HIP quick/mscclpp) without device_group param
                    self.ca_comm = CAClass(
                        group=self.cpu_group,
                        device=self.device,
                    )
            except Exception as e:
                logger.warning(
                    f"Setup Custom allreduce failed with {e}. To silence this "
                    "warning, specify --disable-custom-all-reduce explicitly."
                )
"""
    addition = anchor + """
            if (
                self.ca_comm is not None
                and not self.ca_comm.disabled
                and getattr(self.ca_comm, "_needs_crossover_bench", False)
            ):
                self.ca_comm.find_crossover_size(self.device_group)
"""
    if anchor not in src:
        print(f"ERROR: crossover anchor not found in {ps}", file=sys.stderr)
        sys.exit(1)
    src = src.replace(anchor, addition, 1)
    ps.write_text(src)
    print(f"OK: re-wired crossover autotune in {ps}")

# --- syntax check -------------------------------------------------------------
import ast
for f in (ca, ps):
    ast.parse(f.read_text(), str(f))
print("OK: syntax verified")
