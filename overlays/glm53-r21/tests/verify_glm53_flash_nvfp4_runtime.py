#!/usr/bin/env python3
# Extended verifier: R21 NVFP4 runtime + strict TP3 policy checks, --policy-only mode.
"""Verify the installed GLM-5.3-Flash NVFP4 runtime contract."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

TP3_DISPATCHER = pathlib.Path("/usr/local/bin/serve-glm53-flash.sh")
TP3_LAUNCHER = pathlib.Path("/usr/local/bin/serve-glm53-flash-tp3-r21.sh")
BASE_DELEGATE = "/usr/local/bin/serve-glm53-flash-nvfp4-dflash2.sh"
CAPTURE_WRAPPER = pathlib.Path("/usr/local/bin/serve-glm53-flash-nvfp4-dflash2.sh")
CAPTURE_BASE_DELEGATE = "/usr/local/libexec/serve-glm53-flash-nvfp4-dflash2.sh"
LMCACHE_WRAPPER = pathlib.Path("/usr/local/bin/serve-glm53-flash-lmcache.sh")
GENERIC_CACHE_DELEGATE = (
    "/usr/local/libexec/serve-glm53-flash-lmcache-cache-complete.sh"
)
LOCKED_FINGERPRINT = (
    "cu133-torch213-glm53-r21-tp3-vllme96b18db-b12x6d47b10e-dense-ctx1m-seq8-bt8192"
)
GENERIC_FINGERPRINT = (
    "cu133-torch213-glm53-r21-vllme96b18db-b12x6d47b10e-cache-complete"
)
EXPECTED_VLLM_COMMIT = "e96b18dbb8c19230591e79e0ed056b12947b2ea1"
EXPECTED_B12X_COMMIT = "6d47b10eddf408799796650baf3e802bd56bf844"
EXPECTED_VLLM_TREE = "31e73a43eb8a03e932f03c51341df2c73c60f3d4"
EXPECTED_B12X_TREE = "afdd4b4cc589fddb079f1661d91e932f9d99b8c5"
SOURCE_LOCK = pathlib.Path("/opt/glm53-flash/source.lock")
EXPECTED_SOURCE_LOCK_SHA256 = (
    "21fd2d6ffa3e842ee656f780a8530cce0ffb6601dfa47a5138409247ec4df0d4"
)
EXPECTED_SOURCE_LOCK_FIELDS = {
    "format": "local-inference-source-lock/v1",
    "status": "research-only",
    "release.parent.digest": (
        "sha256:f096012c508f9bc12e8c4e617b8ed19da3a2cecb525e9479904e848730f0c8ac"
    ),
    "vllm.ref": "refs/pull/547/head",
    "vllm.integration.commit": EXPECTED_VLLM_COMMIT,
    "vllm.integration.tree": EXPECTED_VLLM_TREE,
    "b12x.ref": "refs/pull/265/head",
    "b12x.integration.commit": EXPECTED_B12X_COMMIT,
    "b12x.integration.tree": EXPECTED_B12X_TREE,
    "lmcache.commit": "aefe3ab701ab7a835532e701be89f5055b13ec0f",
    "lmcache.tree": "683ab2c165a9aa0e2d1a1ab757af4a8b193688c5",
    "lmcache.package.tree": "976a97f22c0497f34db089dc5f02a713dd0b5888",
    "cache.generic.fingerprint": GENERIC_FINGERPRINT,
    "cache.tp3.fingerprint": LOCKED_FINGERPRINT,
    "target.revision": "378ca54585c46542bad1f3cb3ed0d73ae51cdb62",
    "draft.revision": "aea0ac8a05624512ca9e106c09c16087da998426",
    "qualification.status": "research-only",
}
SOURCE_LOCK_LAUNCHERS = (
    "launcher.tp3",
    "launcher.dflash",
    "launcher.cache-mode",
    "launcher.lmcache",
    "launcher.base-dflash",
    "launcher.base-cache",
)
EXPECTED_DFLASH_DELEGATE = pathlib.Path(
    "/usr/local/libexec/serve-glm53-flash-nvfp4-dflash2.sh"
)
EXPECTED_GENERIC_DELEGATE = pathlib.Path(
    "/usr/local/libexec/serve-glm53-flash-lmcache-cache-complete.sh"
)
EXPECTED_DELEGATE_SHA256 = {
    EXPECTED_DFLASH_DELEGATE: (
        "053145c9586d610c25aba787453f208ee3e4ab3618c35563d09d3f4867897604"
    ),
    EXPECTED_GENERIC_DELEGATE: (
        "60aebc7e9be9cc4dd182c8fcdf123d21ecf97c9696d83ccca8e18271714a501b"
    ),
}
LOCKED_ENV = (
    ("TP", "3"),
    ("DCP", "1"),
    ("MM_ENCODER_TP_MODE", "weights"),
    ("CP_KV_CACHE_INTERLEAVE_SIZE", "4"),
    ("DCP_CKV_GATHER", "0"),
    ("MODEL", "local-inference-lab/GLM-5.3-Flash-NVFP4"),
    ("MODEL_REVISION", "378ca54585c46542bad1f3cb3ed0d73ae51cdb62"),
    ("DFLASH_MODEL_REVISION", "aea0ac8a05624512ca9e106c09c16087da998426"),
    ("DFLASH_MODEL", "local-inference-lab/GLM-5.3-Flash-DFlash2"),
    ("LMCACHE_ENABLED", "0"),
    ("MAX_MODEL_LEN", "1048576"),
    ("MAX_NUM_SEQS", "8"),
    ("MAX_NUM_BATCHED_TOKENS", "8192"),
    ("PREFILL_SCHEDULE_INTERVAL", "8"),
    ("MAX_NUM_PREFILL_TOKENS_PER_STEP", "0"),
    ("MAX_NUM_PARTIAL_PREFILLS", "0"),
    ("DECODE_PREFILL_MIN_DECODE_STEPS", "0"),
    ("DECODE_PREFILL_MAX_WAIT_MS", "0"),
    ("GPU_MEMORY_UTILIZATION", "0.91"),
    ("KV_CACHE_DTYPE", "fp8"),
    ("LOAD_FORMAT", "instanttensor"),
    ("ATTENTION_BACKEND", "B12X"),
    ("VLLM_DISABLE_PYNCCL", "0"),
    ("VLLM_ALLREDUCE_USE_SYMM_MEM", "1"),
    ("NCCL_IB_DISABLE", "1"),
    ("NCCL_P2P_LEVEL", "SYS"),
    ("NCCL_P2P_DISABLE", "0"),
    ("NCCL_CUMEM_ENABLE", "0"),
    ("NCCL_PROTO", "LL,LL128,Simple"),
    ("NCCL_MIN_NCHANNELS", "16"),
    ("NCCL_MAX_NCHANNELS", "16"),
    ("NCCL_BUFFSIZE", "2097152"),
    ("NCCL_NET_PLUGIN", "none"),
    ("NCCL_TUNER_PLUGIN", "none"),
    ("MOE_BACKEND", "auto"),
    ("LINEAR_BACKEND", "b12x"),
    ("MTP_ATTENTION_BACKEND", "B12X"),
    ("MTP_MOE_BACKEND", "humming"),
    ("DFLASH_ATTENTION_BACKEND", "FLASH_ATTN"),
    ("DFLASH_KV_CACHE_DTYPE", "auto"),
    ("B12X_POLICY_MODE", "auto"),
    ("VLLM_B12X_MOE_FP4_FORCE_A16", "0"),
    ("VLLM_PCIE_ALLREDUCE_BACKEND", "b12x"),
    ("VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE", "84KB"),
    ("VLLM_PCIE_ONESHOT_FUSED_ADD_RMS_NORM_MAX_SIZE", "84KB"),
    ("VLLM_PCIE_TWOSHOT_ALLREDUCE_MAX_SIZE", "768KB"),
    ("VLLM_PCIE_DMA_MIN_BYTES", "6MB"),
    ("B12X_PCIE_ONESHOT_THREADS", "512"),
    ("B12X_PCIE_ONESHOT_BLOCK_LIMIT", "4"),
    ("B12X_PCIE_ONESHOT_PUSH", "0"),
    ("B12X_PCIE_FUSED_THREADS", "256"),
    ("B12X_PCIE_FUSED_CTAS_PER_ROW", "0"),
    ("B12X_PCIE_DMA_PIECES", "0"),
    ("B12X_PCIE_ONESHOT_PDL", "1"),
    ("B12X_MHC_PDL", "1"),
    ("VLLM_CPP_AR_1STAGE_NCCL_CUTOFF", "56KB"),
    ("VLLM_CPP_AR_IGNORE_CUTOFF_MAX_ROWS", "0"),
    ("B12X_PCIE_ALLREDUCE", "1"),
    ("B12X_PCIE_ALLREDUCE_ALGORITHM", "auto"),
    ("GLM53_KDA_DECODE_BACKEND", "b12x"),
    ("GLM53_KDA_PREFILL_BACKEND", "flashkda"),
    ("CUDAGRAPH_MODE", "FULL"),
    ("MAX_CUDAGRAPH_CAPTURE_SIZE", "16"),
    ("CUDAGRAPH_CAPTURE_SIZES", "1 2 4 8 16"),
    ("FAIRNESS_ENGINE", "none"),
    ("PREFILL_COMPUTE_SHARE", "none"),
    ("VLLM_ENABLE_PCIE_ALLREDUCE", "1"),
    ("GLM53_CACHE_LAYOUT", "dense"),
    ("ENABLE_PREFIX_CACHING", "1"),
    ("GLM53_R17_REQUIRE_RUNTIME_PROOF", "1"),
)
REQUIRED_UNSET_ENV = (
    "KV_CACHE_QUANT",
    "VLLM_KV_CACHE_LAYOUT",
    "VLLM_SSM_CONV_STATE_LAYOUT",
    "VLLM_DP_SIZE",
    "VLLM_DP_RANK",
    "VLLM_DP_RANK_LOCAL",
    "VLLM_DP_MASTER_IP",
    "VLLM_DP_MASTER_PORT",
    "GLM53_TARGET_BLOCK_SIZE",
    "GLM53_MAMBA_BLOCK_SIZE",
    "NCCL_ALGO",
    "NCCL_COLLNET_ENABLE",
    "NCCL_NVLS_ENABLE",
    "NCCL_SHM_DISABLE",
    "NCCL_PXN_DISABLE",
    "NCCL_P2P_DIRECT_DISABLE",
    "VLLM_PCIE_DMA_FP8",
    "B12X_PCIE_DMA_FP8",
)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy-only",
        action="store_true",
        help="run only the TP3 policy checks (no imports/GPU state required)",
    )
    parser.add_argument("--vllm-version", required=not _is_policy_only(argv))
    parser.add_argument("--vllm-tree", required=not _is_policy_only(argv))
    parser.add_argument("--b12x-tree", required=not _is_policy_only(argv))
    return parser.parse_args(argv)


def _is_policy_only(argv: list[str]) -> bool:
    return "--policy-only" in argv


def substitute(text: str, old: str, new: str) -> str:
    """Replace a sandbox target, failing before a production path can survive."""
    if old not in text:
        raise AssertionError(f"missing sandbox substitution target: {old!r}")
    return text.replace(old, new)


def verify_source_lock() -> None:
    """Verify the child lock and every launcher identity it declares."""
    raw = SOURCE_LOCK.read_bytes()
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    if actual_sha256 != EXPECTED_SOURCE_LOCK_SHA256:
        raise AssertionError(
            f"source lock sha256 {actual_sha256} != {EXPECTED_SOURCE_LOCK_SHA256}"
        )

    fields: dict[str, str] = {}
    for line in raw.decode().splitlines():
        if not line:
            continue
        key, separator, value = line.partition("=")
        if not separator or not key or key in fields:
            raise AssertionError(f"invalid source lock line: {line!r}")
        fields[key] = value

    for key, expected in EXPECTED_SOURCE_LOCK_FIELDS.items():
        actual = fields.get(key)
        if actual != expected:
            raise AssertionError(f"source lock {key} {actual!r} != {expected!r}")

    for prefix in SOURCE_LOCK_LAUNCHERS:
        launcher = pathlib.Path(fields[f"{prefix}.path"])
        expected_sha256 = fields[f"{prefix}.sha256"]
        if not launcher.is_file():
            raise AssertionError(f"missing locked launcher: {launcher}")
        with launcher.open("rb") as source:
            actual_sha256 = hashlib.file_digest(source, "sha256").hexdigest()
        if actual_sha256 != expected_sha256:
            raise AssertionError(
                f"launcher sha256 {actual_sha256} != {expected_sha256}: {launcher}"
            )


def main(args: argparse.Namespace) -> None:
    verify_source_lock()
    assert args.vllm_tree == EXPECTED_VLLM_TREE
    assert args.b12x_tree == EXPECTED_B12X_TREE
    assert os.environ.get("LOCAL_INFERENCE_CACHE_FINGERPRINT") == (GENERIC_FINGERPRINT)
    for delegate, expected_sha256 in EXPECTED_DELEGATE_SHA256.items():
        assert delegate.is_file(), delegate
        assert os.access(delegate, os.X_OK), delegate
        with delegate.open("rb") as source:
            assert hashlib.file_digest(source, "sha256").hexdigest() == (
                expected_sha256
            )
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
    assert "Glm5NextForCausalLM" in ModelRegistry.get_supported_archs()
    assert "Glm5NextForConditionalGeneration" in ModelRegistry.get_supported_archs()

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
    importlib.import_module("instanttensor")

    from vllm.models.deepseek_v4.nvidia import b12x_indexer
    from vllm.models.deepseek_v4.nvidia.b12x_indexer import B12xC4SparseIndexer
    from vllm.models.glm5next.nvidia.pooled_indexer import Glm5NextPooledIndexer

    assert callable(B12xC4SparseIndexer.run_paged_topk)
    assert not hasattr(b12x_indexer, "_run_deepgemm_prefill_topk")
    assert not hasattr(Glm5NextPooledIndexer, "run_deepgemm_prefill_topk")

    assert len(args.vllm_tree) == 40
    assert len(args.b12x_tree) == 40
    for source_dir, expected_commit, expected_tree in (
        (
            "/opt/glm53-flash/vllm",
            EXPECTED_VLLM_COMMIT,
            args.vllm_tree,
        ),
        (
            "/opt/glm53-flash/b12x",
            EXPECTED_B12X_COMMIT,
            args.b12x_tree,
        ),
    ):
        actual_commit = subprocess.check_output(
            ["git", "-C", source_dir, "rev-parse", "HEAD"], text=True
        ).strip()
        actual_tree = subprocess.check_output(
            ["git", "-C", source_dir, "rev-parse", "HEAD^{tree}"], text=True
        ).strip()
        assert actual_commit == expected_commit
        assert actual_tree == expected_tree
        subprocess.run(
            ["git", "-C", source_dir, "diff", "--quiet", "HEAD", "--"],
            check=True,
        )


def verify_tp3_launcher() -> None:
    """Check the strict dispatcher, CLI denial, and complete policy env."""
    assert TP3_DISPATCHER.is_file()
    assert TP3_LAUNCHER.is_file()

    dispatcher_text = TP3_DISPATCHER.read_text()
    tp3_text = TP3_LAUNCHER.read_text()

    assert "if [[ ${TP:-4} == 3 ]]" in dispatcher_text
    assert "normalize_speculator" in dispatcher_text
    assert 'exec /usr/local/bin/serve-glm53-flash-tp3-r21.sh "$@"' in dispatcher_text
    assert "CACHE_MODE must be vram, native, or lmcache" in dispatcher_text

    from_parent_names = {
        "MAX_NUM_SEQS",
        "MAX_NUM_BATCHED_TOKENS",
        "PREFILL_SCHEDULE_INTERVAL",
        "MAX_CUDAGRAPH_CAPTURE_SIZE",
        "CUDAGRAPH_CAPTURE_SIZES",
        "FAIRNESS_ENGINE",
        "PREFILL_COMPUTE_SHARE",
    }
    for name, value in LOCKED_ENV:
        if name in {
            "MODEL",
            "MODEL_REVISION",
            "DFLASH_MODEL",
            "DFLASH_MODEL_REVISION",
        }:
            pattern = rf'lock_env {re.escape(name)} "\$\{{locked_\w+\}}"'
        elif name in from_parent_names:
            pattern = (
                rf"lock_env_from_parent {re.escape(name)} '?{re.escape(value)}'?\b"
            )
        else:
            pattern = rf"lock_env {re.escape(name)} {re.escape(value)}\b"
        assert re.search(pattern, tp3_text), f"missing lock: {name}"
    assert "local-inference-lab/GLM-5.3-Flash-NVFP4" in tp3_text
    assert "local-inference-lab/GLM-5.3-Flash-DFlash2" in tp3_text
    for name in REQUIRED_UNSET_ENV:
        assert f"require_unset_env {name}" in tp3_text
    assert f"readonly fingerprint={LOCKED_FINGERPRINT}" in tp3_text
    for cache_export in (
        "MM_SPARSE_ATTN_AOT_CACHE=${cache_root}/minfer/mm-sparse-attn",
        "MINFER_FMHA_CACHE_DIR=${cache_root}/minfer/fmha-sm120",
        "NUMBA_CACHE_DIR=${cache_root}/numba",
    ):
        assert f"export {cache_export}" in tp3_text

    assert "if (($# > 0)); then" in tp3_text
    assert "rejects caller option ${1%%=*}" in tp3_text
    assert "split-page block-size variables" in tp3_text


def run_tp3_policy_cases() -> None:
    """Exercise the fail-closed TP3 policy and generic delegate boundaries."""
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="glm53-tp3-policy-"))
    try:
        base_stub = tmp / "base-dflash2"
        base_stub.write_text(
            "#!/usr/bin/env bash\n"
            'printf "ENV: MAX_NUM_SEQS=%s MAX_NUM_BATCHED_TOKENS=%s '
            "PREFILL_SCHEDULE_INTERVAL=%s MAX_CUDAGRAPH_CAPTURE_SIZE=%s "
            "CUDAGRAPH_CAPTURE_SIZES=%q FAIRNESS_ENGINE=%s "
            'PREFILL_COMPUTE_SHARE=%s\\n" '
            '"${MAX_NUM_SEQS-}" "${MAX_NUM_BATCHED_TOKENS-}" '
            '"${PREFILL_SCHEDULE_INTERVAL-}" "${MAX_CUDAGRAPH_CAPTURE_SIZE-}" '
            '"${CUDAGRAPH_CAPTURE_SIZES-}" "${FAIRNESS_ENGINE-}" '
            '"${PREFILL_COMPUTE_SHARE-}"\n'
            'printf "LOCKS: MODEL=%s DFLASH_MODEL=%s '
            "CP_KV_CACHE_INTERLEAVE_SIZE=%s DCP_CKV_GATHER=%s "
            'VLLM_DISABLE_PYNCCL=%s FINGERPRINT=%s\\n" '
            '"${MODEL-}" "${DFLASH_MODEL-}" '
            '"${CP_KV_CACHE_INTERLEAVE_SIZE-}" "${DCP_CKV_GATHER-}" '
            '"${VLLM_DISABLE_PYNCCL-}" '
            '"${LOCAL_INFERENCE_CACHE_FINGERPRINT-}"\n'
            'printf "MODE: SPECULATOR=%s NUM_SPECULATIVE_TOKENS=%s '
            'DRY_RUN=%s\\n" "${SPECULATOR-}" '
            '"${NUM_SPECULATIVE_TOKENS-}" "${DRY_RUN-}"\n'
            'printf "STUB-REACHED:"\nprintf " %q" "$@"\nprintf "\\n"\n'
        )
        base_stub.chmod(0o755)

        generic_stub = tmp / "generic-base"
        generic_stub.write_text(
            "#!/usr/bin/env bash\n"
            'printf "GENERIC-STUB: TP=%s CACHE_MODE=%s FINGERPRINT=%s ARGV:" '
            '"${TP-}" "${CACHE_MODE-unset}" '
            '"${LOCAL_INFERENCE_CACHE_FINGERPRINT-}"\n'
            'printf " %q" "$@"\nprintf "\\n"\n'
        )
        generic_stub.chmod(0o755)

        sandbox_tp3 = tmp / "tp3.sh"
        sandbox_tp3.write_text(
            substitute(TP3_LAUNCHER.read_text(), BASE_DELEGATE, str(base_stub))
        )
        sandbox_tp3.chmod(0o755)
        sandbox_dispatcher = tmp / "dispatcher.sh"
        dispatcher_text = substitute(
            TP3_DISPATCHER.read_text(),
            "/usr/local/bin/serve-glm53-flash-tp3-r21.sh",
            str(sandbox_tp3),
        )
        dispatcher_text = substitute(
            dispatcher_text, GENERIC_CACHE_DELEGATE, str(generic_stub)
        )
        sandbox_dispatcher.write_text(dispatcher_text)
        sandbox_dispatcher.chmod(0o755)

        def output_env(**over: str) -> dict[str, str]:
            env = dict(os.environ)
            clean_keys = {
                "CACHE_MODE",
                *(name for name, _ in LOCKED_ENV),
                "VLLM_GLM53_SPLIT_TARGET_BLOCK_SIZE",
                "VLLM_GLM53_SPLIT_MAMBA_BLOCK_SIZE",
                "SPECULATOR",
                "MTP_DEPTH",
                "MTP",
                "DFLASH_DEPTH",
                *REQUIRED_UNSET_ENV,
                "NUM_SPECULATIVE_TOKENS",
                "DRY_RUN",
                "CACHE_CONFIG_DRY_RUN",
                "LOCAL_INFERENCE_CACHE_FINGERPRINT",
            }
            for key in clean_keys:
                env.pop(key, None)
            env.update(over)
            env.setdefault("LOCAL_INFERENCE_CACHE_FINGERPRINT", GENERIC_FINGERPRINT)
            env.setdefault("TP", "3")
            return env

        def reject(
            desc: str, env: dict[str, str], args: list[str]
        ) -> subprocess.CompletedProcess:
            result = subprocess.run(
                ["bash", str(sandbox_tp3), *args],
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            assert result.returncode == 2, (
                f"TP3 policy accepted disallowed input: {desc}; "
                f"rc={result.returncode} stderr={result.stderr}"
            )
            return result

        def accept(
            desc: str, env: dict[str, str], args: list[str]
        ) -> subprocess.CompletedProcess:
            result = subprocess.run(
                ["bash", str(sandbox_tp3), *args],
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            assert result.returncode == 0, (
                f"TP3 policy rejected compliant input: {desc}; "
                f"rc={result.returncode} stderr={result.stderr}"
            )
            return result

        # TP3 accepts no caller arguments: even an end-of-options delimiter can
        # hide mandatory options appended by the wrapper layers.
        for caller_args in (
            ["--api-key", "secret"],
            ["--attention-conf.backend=FLASH_ATTN"],
            ["--ec-transfer-config={}"],
            ["-O3"],
            ["some-model"],
            ["--"],
        ):
            reject(f"caller arguments {caller_args}", output_env(), caller_args)
        secret = "must-not-reach-stderr"
        result = reject(
            "equals-form secret caller option",
            output_env(),
            [f"--api-key={secret}"],
        )
        assert secret not in result.stderr, result.stderr
        assert "--api-key" in result.stderr, result.stderr
        result = reject("secret positional model", output_env(), [secret])
        assert secret not in result.stderr, result.stderr
        for tp_args in (
            ["--tensor-parallel-size", "3"],
            ["--tensor-parallel-s=3"],
            ["-tp3"],
        ):
            result = subprocess.run(
                ["bash", str(sandbox_dispatcher), *tp_args],
                env=output_env(TP="4"),
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            assert result.returncode == 2, (
                f"dispatcher accepted TP override {tp_args}: {result.stderr}"
            )
            assert "GENERIC-STUB" not in result.stdout

        # Environment overrides are rejective at policy level. CACHE_MODE is
        # proven separately against the dispatcher below.
        reject("LMCACHE_ENABLED=1", output_env(LMCACHE_ENABLED="1"), [])
        reject("DCP=2", output_env(DCP="2"), [])
        for key, locked_value in (f for f in LOCKED_ENV if f[0] != "TP"):
            reject(
                f"env override {key}",
                output_env(**{key: ("999999" if locked_value.isdigit() else "bogus")}),
                [],
            )
        for key in REQUIRED_UNSET_ENV:
            reject(
                f"env must remain unset: {key}",
                output_env(**{key: "1"}),
                [],
            )
        reject(
            "split target block size",
            output_env(VLLM_GLM53_SPLIT_TARGET_BLOCK_SIZE="2048"),
            [],
        )
        reject(
            "split mamba block size",
            output_env(VLLM_GLM53_SPLIT_MAMBA_BLOCK_SIZE="2048"),
            [],
        )
        reject("TP=4 fails closed in TP3 script", output_env(TP="4"), [])
        reject("TP=8 fails closed in TP3 script", output_env(TP="8"), [])
        reject("TP unset fails closed", output_env(TP=""), [])
        reject(
            "unqualified MTP depth",
            output_env(SPECULATOR="mtp", NUM_SPECULATIVE_TOKENS="1"),
            [],
        )
        reject(
            "unqualified DFlash2 depth",
            output_env(
                SPECULATOR="dflash2",
                NUM_SPECULATIVE_TOKENS="6",
            ),
            [],
        )
        reject("unknown speculator", output_env(SPECULATOR="ngram"), [])

        # Exact policy values and clean defaults reach the base launcher.
        full_env = output_env(**dict(LOCKED_ENV))
        result = accept("full policy env value set", full_env, [])
        assert "STUB-REACHED" in result.stdout, result.stdout
        for flag in (
            "--enable-expert-parallel",
            "--mm-encoder-tp-mode",
            "weights",
        ):
            assert flag in result.stdout
        assert "--disable-custom-all-reduce" not in result.stdout
        for field in (
            "MODEL=local-inference-lab/GLM-5.3-Flash-NVFP4",
            "DFLASH_MODEL=local-inference-lab/GLM-5.3-Flash-DFlash2",
            "CP_KV_CACHE_INTERLEAVE_SIZE=4",
            "DCP_CKV_GATHER=0",
            "VLLM_DISABLE_PYNCCL=0",
            f"FINGERPRINT={LOCKED_FINGERPRINT}",
        ):
            assert field in result.stdout, (field, result.stdout)

        result = accept("clean policy default", output_env(), [])
        assert "STUB-REACHED" in result.stdout

        # Inherited R21 defaults are translated to TP3 values rather than
        # rejected as arbitrary operator overrides.
        parent_defaults = {
            "MAX_NUM_SEQS": "32",
            "MAX_NUM_BATCHED_TOKENS": "4096",
            "PREFILL_SCHEDULE_INTERVAL": "1",
            "MAX_CUDAGRAPH_CAPTURE_SIZE": "256",
            "CUDAGRAPH_CAPTURE_SIZES": ("1 2 4 8 16 32 40 48 64 96 128 192 256"),
            "FAIRNESS_ENGINE": "compute_share",
            "PREFILL_COMPUTE_SHARE": "0.4",
        }
        result = accept(
            "inherited R21 defaults",
            output_env(**parent_defaults),
            [],
        )
        for expected in (
            "MAX_NUM_SEQS=8",
            "MAX_NUM_BATCHED_TOKENS=8192",
            "PREFILL_SCHEDULE_INTERVAL=8",
            "MAX_CUDAGRAPH_CAPTURE_SIZE=16",
            "CUDAGRAPH_CAPTURE_SIZES=1\\ 2\\ 4\\ 8\\ 16",
            "FAIRNESS_ENGINE=none",
            "PREFILL_COMPUTE_SHARE=none",
        ):
            assert expected in result.stdout, (expected, result.stdout)
        assert "--disable-custom-all-reduce" not in result.stdout

        # The public dispatcher normalizes generic aliases before entering the
        # TP3 policy and maps cache dry-run to the base launcher's dry-run.
        for cache_mode in (None, "vram"):
            env = output_env()
            if cache_mode is not None:
                env["CACHE_MODE"] = cache_mode
            result = subprocess.run(
                ["bash", str(sandbox_dispatcher)],
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            assert result.returncode == 0, (
                f"CACHE_MODE={cache_mode or 'unset'} rejected: {result.stderr}"
            )
            assert "STUB-REACHED" in result.stdout, result.stderr

        for mode_env, expected in (
            (
                {"MTP_DEPTH": "3", "CACHE_CONFIG_DRY_RUN": "1"},
                (
                    "SPECULATOR=mtp",
                    "NUM_SPECULATIVE_TOKENS=3",
                    "DRY_RUN=1",
                ),
            ),
            (
                {
                    "SPECULATOR": "dflash",
                    "DFLASH_DEPTH": "7",
                    "CACHE_CONFIG_DRY_RUN": "1",
                },
                (
                    "SPECULATOR=dflash2",
                    "NUM_SPECULATIVE_TOKENS=7",
                    "DRY_RUN=1",
                ),
            ),
        ):
            result = subprocess.run(
                ["bash", str(sandbox_dispatcher)],
                env=output_env(**mode_env),
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            assert result.returncode == 0, result.stderr
            for field in expected:
                assert field in result.stdout, (field, result.stdout)

        for cache_mode in ("native", "lmcache"):
            result = subprocess.run(
                ["bash", str(sandbox_dispatcher)],
                env=output_env(TP="3", CACHE_MODE=cache_mode),
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            assert result.returncode == 2, (
                f"CACHE_MODE={cache_mode} returned {result.returncode}: {result.stderr}"
            )
            assert "STUB-REACHED" not in result.stdout

        # TP4/TP8 must preserve cache selection and argv while positively
        # reaching the unmodified generic delegate.
        for tp in ("4", "8"):
            for cache_mode in (None, "vram"):
                env = output_env(TP=tp)
                expected_cache_mode = "unset"
                if cache_mode is not None:
                    env["CACHE_MODE"] = cache_mode
                    expected_cache_mode = cache_mode
                sentinel = f"sentinel-{tp}-{expected_cache_mode}"
                generic_run = subprocess.run(
                    [
                        "bash",
                        str(sandbox_dispatcher),
                        "--served-model-name",
                        sentinel,
                        "--",
                    ],
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )
                assert generic_run.returncode == 0, (
                    f"TP={tp} CACHE_MODE={expected_cache_mode} generic "
                    f"delegate failed: {generic_run.stderr}"
                )
                expected_route = (
                    f"GENERIC-STUB: TP={tp} "
                    f"CACHE_MODE={expected_cache_mode} "
                    f"FINGERPRINT={GENERIC_FINGERPRINT} ARGV:"
                )
                assert expected_route in generic_run.stdout, generic_run.stdout
                assert f" --served-model-name {sentinel}" in generic_run.stdout
                assert generic_run.stdout.rstrip().endswith(" --")
                assert "STUB-REACHED" not in generic_run.stdout, (
                    f"TP={tp} entered TP3 policy"
                )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run_wrapper_regression_cases() -> None:
    """Exercise input validation and cache-dtype translation without a GPU."""
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="glm53-wrapper-policy-"))
    try:
        base_stub = tmp / "base-launcher"
        base_stub.write_text(
            "#!/usr/bin/env bash\n"
            'printf "BASE: KV_CACHE_DTYPE=%s\\n" "${KV_CACHE_DTYPE-}"\n'
            'printf "ARGS:"\nprintf " %q" "$@"\nprintf "\\n"\n'
        )
        base_stub.chmod(0o755)

        capture_wrapper = tmp / "capture-wrapper"
        capture_wrapper.write_text(
            substitute(
                CAPTURE_WRAPPER.read_text(), CAPTURE_BASE_DELEGATE, str(base_stub)
            )
        )
        capture_wrapper.chmod(0o755)

        isolated_env = dict(os.environ)
        for name in (
            "FAIRNESS_ENGINE",
            "PREFILL_COMPUTE_SHARE",
            "MAX_NUM_PREFILL_TOKENS_PER_STEP",
            "MAX_NUM_PARTIAL_PREFILLS",
            "DECODE_PREFILL_MIN_DECODE_STEPS",
            "DECODE_PREFILL_MAX_WAIT_MS",
            "LMCACHE_MP_PORT",
            "LMCACHE_HTTP_PORT",
            "LMCACHE_PROMETHEUS_PORT",
        ):
            isolated_env.pop(name, None)

        marker = tmp / "arithmetic-injection"
        env = dict(isolated_env)
        env.update(
            {
                "CUDAGRAPH_CAPTURE_SIZES": "1",
                "MAX_CUDAGRAPH_CAPTURE_SIZE": f'probe[$(touch "{marker}")]',
            }
        )
        result = subprocess.run(
            ["bash", str(capture_wrapper)],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert result.returncode == 2, result
        assert not marker.exists(), "capture-size validation executed shell input"
        assert "BASE:" not in result.stdout, result.stdout

        for maximum, sizes in (
            ("18446744073709551618", "1"),
            ("256", "18446744073709551618"),
            ("9223372036854775808", "1"),
            ("256", "9223372036854775808"),
        ):
            overflow_env = dict(isolated_env)
            overflow_env.update(
                {
                    "CUDAGRAPH_CAPTURE_SIZES": sizes,
                    "MAX_CUDAGRAPH_CAPTURE_SIZE": maximum,
                }
            )
            result = subprocess.run(
                ["bash", str(capture_wrapper)],
                env=overflow_env,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            assert result.returncode == 2, result
            assert "exceeds Bash signed integer range" in result.stderr, result.stderr
            assert "BASE:" not in result.stdout, result.stdout

        env["MAX_CUDAGRAPH_CAPTURE_SIZE"] = "00000000009223372036854775807"
        env["CUDAGRAPH_CAPTURE_SIZES"] = "1 8 16"
        result = subprocess.run(
            ["bash", str(capture_wrapper)],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "BASE:" in result.stdout, result.stdout
        common_env = dict(isolated_env)
        common_env.update(
            {
                "CUDAGRAPH_CAPTURE_SIZES": "none",
                "MAX_CUDAGRAPH_CAPTURE_SIZE": "16",
                "FAIRNESS_ENGINE": "none",
                "PREFILL_COMPUTE_SHARE": "none",
                "MAX_NUM_PREFILL_TOKENS_PER_STEP": "0",
                "MAX_NUM_PARTIAL_PREFILLS": "0",
                "DECODE_PREFILL_MIN_DECODE_STEPS": "0",
                "DECODE_PREFILL_MAX_WAIT_MS": "0",
            }
        )
        invalid_share_env = dict(common_env)
        invalid_share_env.update(
            {
                "FAIRNESS_ENGINE": "compute_share",
                "PREFILL_COMPUTE_SHARE": "0.5invalid",
            }
        )
        result = subprocess.run(
            ["bash", str(capture_wrapper)],
            env=invalid_share_env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert result.returncode == 2, result
        assert "BASE:" not in result.stdout, result.stdout

        for env_name, option_name in (
            (
                "MAX_NUM_PREFILL_TOKENS_PER_STEP",
                "--max-num-prefill-tokens-per-step",
            ),
            ("MAX_NUM_PARTIAL_PREFILLS", "--max-num-partial-prefills"),
            (
                "DECODE_PREFILL_MIN_DECODE_STEPS",
                "--decode-prefill-min-decode-steps",
            ),
            ("DECODE_PREFILL_MAX_WAIT_MS", "--decode-prefill-max-wait-ms"),
        ):
            leading_zero_env = dict(common_env)
            leading_zero_env[env_name] = "08"
            result = subprocess.run(
                ["bash", str(capture_wrapper)],
                env=leading_zero_env,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            assert result.returncode == 0, (env_name, result.stderr)
            assert f" {option_name} 8" in result.stdout, (
                env_name,
                result.stdout,
            )

        df_stub = tmp / "df"
        df_stub.write_text(
            "#!/usr/bin/env bash\nprintf '1B-blocks\\n1099511627776\\n'\n"
        )
        df_stub.chmod(0o755)
        lmcache_stub = tmp / "lmcache"
        lmcache_stub.write_text("#!/usr/bin/env bash\nexec sleep 30\n")
        lmcache_stub.chmod(0o755)
        health = tmp / "health"
        health.write_text("ok\n")

        lmcache_wrapper = tmp / "lmcache-wrapper"
        lmcache_text = LMCACHE_WRAPPER.read_text()
        lmcache_text = substitute(lmcache_text, BASE_DELEGATE, str(base_stub))
        lmcache_text = substitute(
            lmcache_text, "/opt/venv/bin/lmcache", str(lmcache_stub)
        )
        lmcache_text = substitute(
            lmcache_text,
            'readonly health_url="http://127.0.0.1:${http_port}/healthcheck"',
            f'readonly health_url="file://{health}"',
        )
        lmcache_wrapper.write_text(lmcache_text)
        lmcache_wrapper.chmod(0o755)

        env = dict(isolated_env)
        env.update(
            {
                "PATH": f"{tmp}:{env['PATH']}",
                "LMCACHE_ENABLED": "1",
                "LMCACHE_KV_CACHE_DTYPE": "fp8_ds_mla",
                "LMCACHE_L2_ENABLED": "0",
                "LMCACHE_TRANSFER_MODE": "engine_driven",
            }
        )
        for port_name in (
            "LMCACHE_MP_PORT",
            "LMCACHE_HTTP_PORT",
            "LMCACHE_PROMETHEUS_PORT",
        ):
            invalid_port_env = dict(env)
            invalid_port_env[port_name] = "65536"
            result = subprocess.run(
                ["bash", str(lmcache_wrapper)],
                env=invalid_port_env,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            assert result.returncode == 2, (port_name, result.stderr)
            assert "BASE:" not in result.stdout, result.stdout

        result = subprocess.run(
            ["bash", str(lmcache_wrapper)],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert "BASE: KV_CACHE_DTYPE=fp8" in result.stdout, result.stdout
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    argv = sys.argv[1:]
    if _is_policy_only(argv):
        args = parse_args(
            argv + ["--vllm-version", "x", "--vllm-tree", "0", "--b12x-tree", "0"]
        )
        verify_tp3_launcher()
        run_tp3_policy_cases()
        run_wrapper_regression_cases()
        print("TP3 policy: recorded (structural), fail-closed matrix: PASS")
    else:
        args = parse_args(argv)
        main(args)
        verify_tp3_launcher()
        run_tp3_policy_cases()
        run_wrapper_regression_cases()
