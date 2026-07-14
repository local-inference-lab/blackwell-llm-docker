#!/usr/bin/env python3
"""Run a real NCCL collective and report the mapped NCCL runtime."""

from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _mapped_nccl_libraries() -> list[str]:
    paths: set[str] = set()
    with open("/proc/self/maps", encoding="utf-8") as maps:
        for line in maps:
            if "libnccl" in line:
                path = line.rsplit(maxsplit=1)[-1]
                if path.startswith("/"):
                    paths.add(path)
    return sorted(paths)


def _worker(rank: int, world_size: int, port: int, expected_library: str) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    value = torch.tensor(float(rank + 1), device=f"cuda:{rank}")
    dist.all_reduce(value)
    torch.cuda.synchronize()

    expected_sum = world_size * (world_size + 1) / 2
    if value.item() != expected_sum:
        raise RuntimeError(
            f"rank {rank}: all_reduce returned {value.item()}, expected {expected_sum}"
        )

    libraries = _mapped_nccl_libraries()
    print(f"rank={rank} value={value.item():g} nccl={libraries}", flush=True)
    if not any(
        os.path.exists(library) and os.path.samefile(expected_library, library)
        for library in libraries
    ):
        raise RuntimeError(
            f"rank {rank}: expected NCCL library {expected_library!r} is not mapped"
        )

    dist.barrier()
    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--port", type=int, default=29614)
    parser.add_argument(
        "--expected-library",
        default="/opt/libnccl-local-inference.so.2.30.4",
    )
    args = parser.parse_args()
    mp.spawn(
        _worker,
        args=(args.world_size, args.port, args.expected_library),
        nprocs=args.world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
