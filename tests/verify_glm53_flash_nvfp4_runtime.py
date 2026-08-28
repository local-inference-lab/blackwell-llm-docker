#!/usr/bin/env python3
"""Verify the installed GLM-5.3-Flash NVFP4 runtime contract."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import pathlib
import subprocess


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm-version", required=True)
    parser.add_argument("--vllm-tree", required=True)
    parser.add_argument("--b12x-tree", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    assert importlib.metadata.version("vllm") == args.vllm_version
    assert importlib.metadata.version("b12x") == "1.3.0"
    assert importlib.metadata.version("flashinfer-python") == "0.6.18+cu133"
    assert importlib.metadata.version("instanttensor") == "0.1.9"
    assert importlib.metadata.version("nvidia-cutlass-dsl") == "4.6.2"

    import b12x
    import torch
    import vllm
    from vllm.model_executor.models.registry import ModelRegistry

    assert torch.__version__.startswith("2.13.0")
    supported_archs = ModelRegistry.get_supported_archs()
    assert "Glm5NextForCausalLM" in supported_archs
    assert "Glm5NextForConditionalGeneration" in supported_archs
    assert "DFlash2DraftModel" in supported_archs

    vllm_path = pathlib.Path(vllm.__file__).resolve()
    b12x_path = pathlib.Path(b12x.__file__).resolve()
    assert vllm_path.is_relative_to("/opt/glm53-flash/vllm")
    assert b12x_path.is_relative_to("/opt/glm53-flash/b12x")
    assert pathlib.Path("/opt/glm53-flash/vllm/vllm/models/glm5next").is_dir()

    stable_ops_spec = importlib.util.find_spec("vllm._C_stable_libtorch")
    assert stable_ops_spec is not None and stable_ops_spec.origin is not None
    assert pathlib.Path(stable_ops_spec.origin).resolve(strict=True).is_file()
    if torch.cuda.is_available():
        importlib.import_module("vllm._C_stable_libtorch")
    importlib.import_module("vllm.vllm_flash_attn.layers.rotary")
    importlib.import_module("vllm.models.glm5next.nvidia.model")
    importlib.import_module("vllm.model_executor.models.qwen3_dflash2")
    importlib.import_module("instanttensor")

    from vllm.models.deepseek_v4.nvidia import b12x_indexer
    from vllm.models.deepseek_v4.nvidia.b12x_indexer import B12xC4SparseIndexer
    from vllm.models.glm5next.nvidia.pooled_indexer import Glm5NextPooledIndexer

    assert callable(B12xC4SparseIndexer.run_paged_topk)
    assert callable(b12x_indexer._run_deepgemm_prefill_topk)
    assert callable(B12xC4SparseIndexer.run_deepgemm_prefill_topk)
    assert not hasattr(Glm5NextPooledIndexer, "run_deepgemm_prefill_topk")

    assert len(args.vllm_tree) == 40
    assert len(args.b12x_tree) == 40
    for source_dir, expected_tree in (
        ("/opt/glm53-flash/vllm", args.vllm_tree),
        ("/opt/glm53-flash/b12x", args.b12x_tree),
    ):
        actual_tree = subprocess.check_output(
            ["git", "-C", source_dir, "rev-parse", "HEAD^{tree}"], text=True
        ).strip()
        assert actual_tree == expected_tree
        subprocess.run(
            ["git", "-C", source_dir, "diff", "--quiet", "HEAD", "--"],
            check=True,
        )


if __name__ == "__main__":
    main()
