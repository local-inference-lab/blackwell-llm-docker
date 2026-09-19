#!/usr/bin/env python3
"""Qualify Kimi-K3 reasoning, tool-call, and image API behavior."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import time
import urllib.request
from pathlib import Path
from typing import Any


def _request_json(
    url: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    timeout: float = 600,
) -> tuple[dict[str, Any], float]:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        method=method,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.load(response)
    return body, time.perf_counter() - started


def _write_receipt(
    output_dir: Path,
    name: str,
    *,
    request: dict[str, Any],
    response: dict[str, Any],
    seconds: float,
) -> None:
    receipt = {
        "request": request,
        "response": response,
        "response_seconds": seconds,
    }
    path = output_dir / f"{name}.json"
    path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _message(response: dict[str, Any]) -> dict[str, Any]:
    choices = response.get("choices")
    assert isinstance(choices, list) and len(choices) == 1, response
    message = choices[0].get("message")
    assert isinstance(message, dict), response
    return message


def _finish_reason(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    assert isinstance(choices, list) and len(choices) == 1, response
    finish_reason = choices[0].get("finish_reason")
    assert isinstance(finish_reason, str), response
    return finish_reason


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    models, models_seconds = _request_json(f"{args.base_url}/v1/models")
    advertised = {
        entry.get("id")
        for entry in models.get("data", [])
        if isinstance(entry, dict)
    }
    assert args.model in advertised, (args.model, advertised)

    reasoning_request = {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Compute 17 multiplied by 23. Explain the calculation briefly, "
                    "then give the numeric result."
                ),
            }
        ],
        "temperature": 0,
        "max_tokens": 384,
    }
    reasoning_response, reasoning_seconds = _request_json(
        f"{args.base_url}/v1/chat/completions",
        method="POST",
        payload=reasoning_request,
    )
    reasoning_message = _message(reasoning_response)
    assert _finish_reason(reasoning_response) == "stop", reasoning_response
    reasoning = reasoning_message.get("reasoning") or reasoning_message.get(
        "reasoning_content"
    )
    assert isinstance(reasoning, str) and reasoning.strip(), reasoning_response
    content = reasoning_message.get("content")
    assert isinstance(content, str) and "391" in content, reasoning_response
    _write_receipt(
        args.output_dir,
        "reasoning",
        request=reasoning_request,
        response=reasoning_response,
        seconds=reasoning_seconds,
    )

    tool_request = {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Use the calculator tool to multiply 37 by 19. "
                    "Do not answer without the tool."
                ),
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "calculator",
                    "description": "Evaluate a multiplication.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "integer"},
                            "b": {"type": "integer"},
                        },
                        "required": ["a", "b"],
                    },
                },
            }
        ],
        "tool_choice": "required",
        "temperature": 0,
        "max_tokens": 512,
    }
    tool_response, tool_seconds = _request_json(
        f"{args.base_url}/v1/chat/completions",
        method="POST",
        payload=tool_request,
    )
    tool_message = _message(tool_response)
    assert _finish_reason(tool_response) == "tool_calls", tool_response
    tool_calls = tool_message.get("tool_calls")
    assert isinstance(tool_calls, list) and len(tool_calls) == 1, tool_response
    function = tool_calls[0].get("function")
    assert isinstance(function, dict) and function.get("name") == "calculator"
    arguments = json.loads(function["arguments"])
    assert arguments == {"a": 37, "b": 19}, arguments
    _write_receipt(
        args.output_dir,
        "tool-call",
        request=tool_request,
        response=tool_response,
        seconds=tool_seconds,
    )

    # Raw completions bypass the chat parser's request adjustments. Close the
    # reasoning prefix and disable inserted spaces at special-token boundaries.
    choice_request = {
        "model": args.model,
        "prompt": (
            "Write HELLO exactly, with no explanation.\n"
            "<|open|>think<|sep|>The response is HELLO."
            "<|close|>think<|sep|><|open|>response<|sep|>"
        ),
        "structured_outputs": {"choice": ["HELLO"]},
        "spaces_between_special_tokens": False,
        "temperature": 0,
        "max_tokens": 16,
    }
    choice_response, choice_seconds = _request_json(
        f"{args.base_url}/v1/completions",
        method="POST",
        payload=choice_request,
    )
    assert choice_response["choices"][0]["text"] == "HELLO", choice_response
    assert _finish_reason(choice_response) == "stop", choice_response
    _write_receipt(
        args.output_dir,
        "structured-choice",
        request=choice_request,
        response=choice_response,
        seconds=choice_seconds,
    )

    image_bytes = args.image.read_bytes()
    image_mime = mimetypes.guess_type(args.image.name)[0] or "image/png"
    image_uri = (
        f"data:{image_mime};base64," + base64.b64encode(image_bytes).decode()
    )
    vision_request = {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_uri}},
                    {
                        "type": "text",
                        "text": (
                            "Identify the fruit depicted in this image and state "
                            "its usual ripe color in one sentence."
                        ),
                    },
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": 4096,
    }
    vision_response, vision_seconds = _request_json(
        f"{args.base_url}/v1/chat/completions",
        method="POST",
        payload=vision_request,
    )
    vision_message = _message(vision_response)
    assert _finish_reason(vision_response) == "stop", vision_response
    vision_text = " ".join(
        str(vision_message.get(key) or "")
        for key in ("reasoning", "reasoning_content", "content")
    ).lower()
    assert "strawberr" in vision_text and "red" in vision_text, vision_response
    # The receipt records the image identity without embedding a base64 copy.
    vision_receipt_request = dict(vision_request)
    vision_receipt_request["messages"] = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "path": str(args.image.resolve()),
                        "sha256": hashlib.sha256(image_bytes).hexdigest(),
                        "media_type": image_mime,
                    },
                },
                vision_request["messages"][0]["content"][1],
            ],
        }
    ]
    _write_receipt(
        args.output_dir,
        "vision",
        request=vision_receipt_request,
        response=vision_response,
        seconds=vision_seconds,
    )

    summary = {
        "artifact_kind": "Kimi-K3 OpenAI API qualification",
        "base_url": args.base_url,
        "model": args.model,
        "models_response_seconds": models_seconds,
        "reasoning": "qualified",
        "tool_calls": "qualified",
        "structured_choice": "qualified",
        "vision": "qualified",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
