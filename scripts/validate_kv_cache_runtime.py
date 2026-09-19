#!/usr/bin/env python3
"""Exercise a serving endpoint's external KV-cache path.

The probe stores one long prompt, submits unrelated prompts to pressure the
GPU prefix cache, and then repeats the original prompt. It records response
usage, latency, and selected Prometheus metrics without assuming a particular
cache implementation.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def _json_request(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.load(response)
    return {"elapsed_seconds": time.perf_counter() - started, "body": body}


def _parse_metrics(text: str) -> dict[str, float]:
    selected: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name_and_labels, separator, value = line.rpartition(" ")
        if not separator:
            continue
        metric_name = name_and_labels.split("{", 1)[0]
        lowered = metric_name.lower()
        if "kv_offload" not in lowered and "lmcache" not in lowered:
            continue
        try:
            numeric_value = float(value)
        except ValueError:
            continue
        selected[name_and_labels] = numeric_value
    return selected


def _read_metrics(url: str, timeout: float) -> dict[str, float]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return _parse_metrics(response.read().decode())


def _reset_local_prefix_cache(
    base_url: str,
    timeout: float,
    attempts: int = 30,
) -> dict[str, Any]:
    """Clear GPU prefix blocks while preserving connector-managed storage."""
    url = f"{base_url.rstrip('/')}/reset_prefix_cache?reset_external=false"
    for attempt in range(1, attempts + 1):
        try:
            result = _json_request(url, {}, timeout)
        except urllib.error.HTTPError as error:
            if error.code == 404:
                raise RuntimeError(
                    "local prefix reset requires VLLM_SERVER_DEV_MODE=1 "
                    "on the model server"
                ) from error
            raise
        if result["body"].get("success") is True:
            return {
                "attempts": attempt,
                "elapsed_seconds": result["elapsed_seconds"],
                "success": True,
            }
        if attempt != attempts:
            time.sleep(1)
    raise RuntimeError(
        f"local prefix cache remained busy after {attempts} reset attempts"
    )


def _clear_external_l1(url: str, timeout: float) -> dict[str, Any]:
    """Clear connector RAM after asynchronous stores have settled."""
    result = _json_request(url, {"tier": "l1", "force": True}, timeout)
    body = result["body"]
    if body.get("status") != "ok":
        raise RuntimeError(f"external L1 clear failed: {body}")
    return {
        "elapsed_seconds": result["elapsed_seconds"],
        "response": body,
    }


def _completion(
    base_url: str,
    model: str,
    prompt: str,
    timeout: float,
    max_tokens: int = 1,
) -> dict[str, Any]:
    result = _json_request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": False,
        },
        timeout,
    )
    body = result["body"]
    choice = (body.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    usage = body.get("usage") or {}
    prompt_details = usage.get("prompt_tokens_details") or {}
    return {
        "elapsed_seconds": result["elapsed_seconds"],
        "id": body.get("id"),
        "finish_reason": choice.get("finish_reason"),
        "content": message.get("content"),
        "reasoning": message.get("reasoning") or message.get("reasoning_content"),
        "completion_tokens": usage.get("completion_tokens"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "cached_tokens": prompt_details.get("cached_tokens"),
    }


def _native_restored_bytes(before: dict, after: dict) -> float:
    """Count completed native CPU-to-GPU bytes, excluding metric timestamps."""
    return sum(
        value - before.get(url, {}).get(name, 0)
        for url, metrics in after.items()
        for name, value in metrics.items()
        if name.startswith("vllm:kv_offload_total_bytes_total{")
        and 'transfer_type="CPU_to_GPU"' in name
    )


def _prompt(label: str, characters: int) -> str:
    sentence = (
        f"Cache stream {label} contains a deterministic qualification record "
        "for distributed inference, request ownership, and reproducible "
        "measurements. "
    )
    body = (sentence * (characters // len(sentence) + 1))[:characters]
    return f"Cache record {label}.\n{body}\nReturn the integer 17."


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--metrics-url")
    parser.add_argument("--secondary-metrics-url")
    parser.add_argument(
        "--external-l1-clear-url",
        help=(
            "connector endpoint that clears only its RAM tier before replay; "
            "LMCache MP exposes POST /cache/clear"
        ),
    )
    parser.add_argument("--prompt-characters", type=int, default=32_000)
    parser.add_argument("--churn-requests", type=int, default=6)
    parser.add_argument("--reset-local-before-replay", action="store_true")
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=0,
        help=(
            "wait for asynchronous cache writes before replay and for "
            "asynchronous reads before collecting final metrics"
        ),
    )
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--require-native-restore", action="store_true")
    parser.add_argument("--require-identical-output", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.max_tokens < 1:
        parser.error("--max-tokens must be positive")
    if args.require_native_restore and not args.metrics_url:
        parser.error("--require-native-restore requires --metrics-url")

    metric_urls = [
        url for url in (args.metrics_url, args.secondary_metrics_url) if url is not None
    ]

    def metrics() -> dict[str, dict[str, float]]:
        return {url: _read_metrics(url, args.timeout) for url in metric_urls}

    seed_prompt = _prompt("seed", args.prompt_characters)
    report: dict[str, Any] = {
        "schema_version": 1,
        "base_url": args.base_url,
        "model": args.model,
        "prompt_characters": len(seed_prompt),
        "churn_requests": args.churn_requests,
        "settle_seconds": args.settle_seconds,
        "max_tokens": args.max_tokens,
        "metrics_before": metrics(),
        "seed": _completion(
            args.base_url, args.model, seed_prompt, args.timeout, args.max_tokens
        ),
        "churn": [],
    }
    for index in range(args.churn_requests):
        report["churn"].append(
            _completion(
                args.base_url,
                args.model,
                _prompt(f"churn-{index}", args.prompt_characters),
                args.timeout,
                args.max_tokens,
            )
        )
    if args.settle_seconds > 0:
        time.sleep(args.settle_seconds)
    report["metrics_after_churn"] = metrics()
    if args.external_l1_clear_url:
        report["external_l1_clear"] = _clear_external_l1(
            args.external_l1_clear_url, args.timeout
        )
    if args.reset_local_before_replay:
        report["local_prefix_reset"] = _reset_local_prefix_cache(
            args.base_url, args.timeout
        )
    report["replay"] = _completion(
        args.base_url, args.model, seed_prompt, args.timeout, args.max_tokens
    )
    if args.settle_seconds > 0:
        time.sleep(args.settle_seconds)
    report["metrics_after_replay"] = metrics()
    report["native_restored_bytes"] = _native_restored_bytes(
        report["metrics_after_churn"], report["metrics_after_replay"]
    )
    output_fields = ("content", "reasoning", "completion_tokens", "finish_reason")
    report["identical_output"] = all(
        report["seed"].get(key) == report["replay"].get(key) for key in output_fields
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.require_native_restore and report["native_restored_bytes"] <= 0:
        raise RuntimeError("replay completed without a native CPU-to-GPU cache restore")
    if args.require_identical_output and not report["identical_output"]:
        raise RuntimeError("cold and restored requests generated different output")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
