from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPOSITORY_ROOT / "launchers" / "serve-kimi-k3-qsrt-dspark"
TP8_LAUNCHER = REPOSITORY_ROOT / "launchers" / "serve-kimi-k3-qsrt-dspark-tp8"


def _run_launcher(
    path: Path = LAUNCHER, **overrides: str
) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "DRY_RUN": "1",
        "MODEL": "/nonexistent/target",
        "DRAFT_MODEL": "/nonexistent/draft",
        **overrides,
    }
    return subprocess.run(
        [str(path)],
        env=env,
        check=False,
        text=True,
        capture_output=True,
    )


def test_qsrt_dspark_dry_run_preserves_default_depth() -> None:
    result = _run_launcher()

    assert result.returncode == 0, result.stderr
    assert '"num_speculative_tokens":7' in result.stdout
    assert "--max-num-scheduled-tokens" not in result.stdout


def test_qsrt_dspark_accepts_aligned_explicit_scheduler_budget() -> None:
    result = _run_launcher(
        DSPARK_SPECULATIVE_TOKENS="3",
        MAX_NUM_BATCHED_TOKENS="1540",
        MAX_NUM_SCHEDULED_TOKENS="1536",
        MAX_NUM_SEQS="2",
    )

    assert result.returncode == 0, result.stderr
    assert '"num_speculative_tokens":3' in result.stdout
    assert "--max-num-batched-tokens 1540" in result.stdout
    assert "--max-num-scheduled-tokens 1536" in result.stdout
    assert "--max-num-seqs 2" in result.stdout
    assert "DSpark reserve: 4" in result.stderr


def test_qsrt_dspark_rejects_budget_without_draft_reserve() -> None:
    result = _run_launcher(
        DSPARK_SPECULATIVE_TOKENS="3",
        MAX_NUM_BATCHED_TOKENS="1536",
        MAX_NUM_SCHEDULED_TOKENS="1536",
        MAX_NUM_SEQS="2",
    )

    assert result.returncode == 2
    assert "requires at least 1540 total batch slots" in result.stderr


def test_tp8_profile_preserves_recurrent_boundary() -> None:
    result = _run_launcher(TP8_LAUNCHER)

    assert result.returncode == 0, result.stderr
    assert "TP_SIZE=8" in result.stdout
    assert "DCP_SIZE=8" in result.stdout
    assert '"num_speculative_tokens":3' in result.stdout
    assert "--max-num-batched-tokens 1540" in result.stdout
    assert "--max-num-scheduled-tokens 1536" in result.stdout
    assert "--max-num-seqs 2" in result.stdout
