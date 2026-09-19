import importlib.util
import io
import urllib.error
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "validate_kv_cache_runtime.py"
SPEC = importlib.util.spec_from_file_location("validate_kv_cache_runtime", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_parse_metrics_selects_cache_metrics_and_preserves_labels() -> None:
    metrics = MODULE._parse_metrics(
        """
# HELP vllm:kv_offload_store_bytes_total Stored bytes
vllm:kv_offload_store_bytes_total{engine=\"0\",tier=\"cpu\"} 4096
lmcache_retrieve_hit_rate 0.75
vllm:num_requests_running 3
"""
    )

    assert metrics == {
        'vllm:kv_offload_store_bytes_total{engine="0",tier="cpu"}': 4096.0,
        "lmcache_retrieve_hit_rate": 0.75,
    }


def test_prompt_identity_is_stable_and_labels_are_distinct() -> None:
    seed = MODULE._prompt("seed", 4096)
    churn = MODULE._prompt("churn-0", 4096)

    assert seed == MODULE._prompt("seed", 4096)
    assert seed != churn
    assert seed[1024:3072] != churn[1024:3072]
    assert seed.endswith("Return the integer 17.")


def test_native_restore_counter_excludes_stores_and_timestamps() -> None:
    load = 'vllm:kv_offload_total_bytes_total{transfer_type="CPU_to_GPU"}'
    store = 'vllm:kv_offload_total_bytes_total{transfer_type="GPU_to_CPU"}'
    timestamp = 'vllm:kv_offload_total_bytes_created{transfer_type="CPU_to_GPU"}'
    before = {"endpoint": {load: 1024, store: 2000, timestamp: 1000}}
    after = {"endpoint": {load: 8192, store: 4000, timestamp: 1000}}
    assert MODULE._native_restored_bytes(before, after) == 7168
    assert MODULE._native_restored_bytes(after, after) == 0


def test_completion_preserves_requested_decode_length_and_reasoning(
    monkeypatch,
) -> None:
    def request(url, payload, timeout):
        assert payload["max_tokens"] == 64
        return {
            "elapsed_seconds": 1,
            "body": {
                "choices": [
                    {
                        "message": {"reasoning_content": "reason", "content": "17"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"completion_tokens": 12},
            },
        }

    monkeypatch.setattr(MODULE, "_json_request", request)
    result = MODULE._completion("http://localhost:8000", "model", "prompt", 60, 64)
    assert result["reasoning"] == "reason"
    assert result["completion_tokens"] == 12


def test_local_prefix_reset_retries_without_resetting_external_storage(
    monkeypatch,
) -> None:
    calls: list[str] = []
    responses = iter(
        [
            {"elapsed_seconds": 0.1, "body": {"success": False}},
            {"elapsed_seconds": 0.2, "body": {"success": True}},
        ]
    )

    def fake_json_request(url, payload, timeout):
        calls.append(url)
        assert payload == {}
        assert timeout == 10
        return next(responses)

    monkeypatch.setattr(MODULE, "_json_request", fake_json_request)
    monkeypatch.setattr(MODULE.time, "sleep", lambda _: None)

    result = MODULE._reset_local_prefix_cache(
        "http://127.0.0.1:8000", timeout=10, attempts=2
    )

    assert result == {"attempts": 2, "elapsed_seconds": 0.2, "success": True}
    assert calls == [
        "http://127.0.0.1:8000/reset_prefix_cache?reset_external=false",
        "http://127.0.0.1:8000/reset_prefix_cache?reset_external=false",
    ]


def test_local_prefix_reset_explains_hidden_admin_endpoint(monkeypatch) -> None:
    def hidden_endpoint(url, payload, timeout):
        raise urllib.error.HTTPError(url, 404, "Not Found", {}, io.BytesIO())

    monkeypatch.setattr(MODULE, "_json_request", hidden_endpoint)

    try:
        MODULE._reset_local_prefix_cache("http://127.0.0.1:8000", timeout=10)
    except RuntimeError as error:
        assert str(error) == (
            "local prefix reset requires VLLM_SERVER_DEV_MODE=1 on the model server"
        )
    else:
        raise AssertionError("hidden reset endpoint unexpectedly succeeded")


def test_external_l1_clear_preserves_lower_tiers(monkeypatch) -> None:
    calls = []

    def fake_json_request(url, payload, timeout):
        calls.append((url, payload, timeout))
        return {
            "elapsed_seconds": 0.25,
            "body": {"status": "ok", "cleared": {"tier": "l1"}},
        }

    monkeypatch.setattr(MODULE, "_json_request", fake_json_request)

    result = MODULE._clear_external_l1("http://127.0.0.1:8089/cache/clear", timeout=20)

    assert calls == [
        (
            "http://127.0.0.1:8089/cache/clear",
            {"tier": "l1", "force": True},
            20,
        )
    ]
    assert result == {
        "elapsed_seconds": 0.25,
        "response": {"status": "ok", "cleared": {"tier": "l1"}},
    }
