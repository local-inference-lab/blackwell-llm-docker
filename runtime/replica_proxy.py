"""Conversation-affinity HTTP proxy in front of independent replica servers.

In replicas mode every replica is a complete vLLM server on its own GPU and
this proxy is the public endpoint. Each request records the key of its whole
history (chat or Responses messages, or a completion prompt) with the replica
that serves it. A later request whose history extends a recorded one, the next
turn of the same conversation, goes to that replica, so its prefix cache and
recurrent checkpoints stay on one GPU. Every other request, including first
turns and identical prompts sent side by side, goes to the replica with the
fewest requests in flight. Responses API follow-ups that name a
``previous_response_id`` go to the replica that produced it, and every request
with the same ``X-LIL-Affinity`` header goes to one replica.

``/health`` succeeds only while every replica is healthy, and ``/metrics``
merges the replicas' Prometheus metrics with a ``replica`` label.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import sys
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

AFFINITY_HEADER = "x-lil-affinity"
_BLOCK_CHARS = 4096
_BLOCK_TOKENS = 1024
_HOP_BY_HOP = frozenset(
    {
        "connection",
        "content-length",
        "host",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)
_RESPONSE_ID = re.compile(rb'"id"\s*:\s*"(resp_[A-Za-z0-9_-]+)"')


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str
    ).encode()


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()[:32]


def _chain(seed: Any, items: list[Any]) -> list[str]:
    """Keys of the first 1, 2, ... items, each extending the previous key."""
    state = hashlib.sha256(_canonical(seed))
    keys = []
    for item in items:
        state.update(_canonical(item) + b"\x1e")
        keys.append(state.hexdigest()[:32])
    return keys


@dataclass(frozen=True)
class Prompt:
    """A completion prompt indexed by its whole blocks and a hashed tail."""

    anchors: tuple[str, ...]
    length: int
    block: int
    tail: str
    value: Any

    def tail_matches(self, start: int, length: int, digest: str) -> bool:
        """Whether this prompt continues ``length`` items after ``start``
        with the given digest and then goes on."""
        end = start + length
        return end < self.length and _digest(self.value[start:end]) == digest


def _prompt(seed: Any, value: Any) -> Prompt:
    block = _BLOCK_CHARS if isinstance(value, str) else _BLOCK_TOKENS
    whole = len(value) // block
    anchors = tuple(
        _chain(
            seed, [value[index * block : (index + 1) * block] for index in range(whole)]
        )
    )
    return Prompt(
        anchors=(_digest(seed), *anchors),
        length=len(value),
        block=block,
        tail=_digest(value[whole * block :]),
        value=value,
    )


@dataclass(frozen=True)
class Affinity:
    """What one request records, and what it may follow."""

    key: str | None = None
    prefixes: tuple[str, ...] = ()
    prompt: Prompt | None = None
    previous_response_id: str | None = None


def request_affinity(path: str, body: bytes, headers: dict[str, str]) -> Affinity:
    """Describe how a request relates to earlier ones."""
    lowered = {name.lower(): value for name, value in headers.items()}
    explicit = lowered.get(AFFINITY_HEADER)
    if explicit:
        key = _digest(["header", explicit])
        return Affinity(key=key, prefixes=(key,))
    try:
        request = json.loads(body) if body else None
    except (UnicodeDecodeError, ValueError):
        return Affinity()
    if not isinstance(request, dict):
        return Affinity()
    model = request.get("model")
    tools = request.get("tools")
    if path.endswith(("/chat/completions", "/messages")):
        messages = request.get("messages")
        if isinstance(messages, list) and messages:
            keys = _chain(["chat", model, request.get("system"), tools], messages)
            return Affinity(key=keys[-1], prefixes=tuple(reversed(keys[:-1])))
    elif path.endswith("/completions"):
        prompt = request.get("prompt")
        if isinstance(prompt, list) and prompt and not isinstance(prompt[0], int):
            prompt = prompt[0] if len(prompt) == 1 else None
        if isinstance(prompt, (str, list)) and prompt:
            return Affinity(prompt=_prompt(["completion", model], prompt))
    elif path.endswith("/responses"):
        items = request.get("input")
        items = items if isinstance(items, list) else [items]
        keys = _chain(["responses", model, request.get("instructions"), tools], items)
        previous = request.get("previous_response_id")
        return Affinity(
            key=keys[-1],
            prefixes=tuple(reversed(keys[:-1])),
            previous_response_id=previous if isinstance(previous, str) else None,
        )
    return Affinity()


class StickyRouter:
    """Send each conversation's next turn to its replica; balance the rest."""

    def __init__(self, replicas: int, capacity: int = 65536) -> None:
        if replicas < 1:
            raise ValueError("StickyRouter needs at least one replica")
        self.replicas = replicas
        self.capacity = capacity
        self.in_flight = [0] * replicas
        self._next = 0
        self._keys: OrderedDict[str, int] = OrderedDict()
        self._prompts: OrderedDict[str, list[tuple[int, str, int]]] = OrderedDict()
        self._responses: OrderedDict[str, int] = OrderedDict()

    def _remember(self, table: OrderedDict, key: str, value: Any) -> None:
        table[key] = value
        table.move_to_end(key)
        while len(table) > self.capacity:
            table.popitem(last=False)

    def _follow(self, affinity: Affinity) -> int | None:
        if affinity.previous_response_id is not None:
            replica = self._responses.get(affinity.previous_response_id)
            if replica is not None:
                return replica
        for key in affinity.prefixes:
            replica = self._keys.get(key)
            if replica is not None:
                return replica
        prompt = affinity.prompt
        if prompt is not None:
            for index in range(len(prompt.anchors) - 1, -1, -1):
                for length, digest, replica in reversed(
                    self._prompts.get(prompt.anchors[index], ())
                ):
                    if prompt.tail_matches(index * prompt.block, length, digest):
                        return replica
        return None

    def _least_busy(self) -> int:
        # Ties rotate so sequential conversations spread across replicas.
        least = min(self.in_flight)
        for offset in range(self.replicas):
            replica = (self._next + offset) % self.replicas
            if self.in_flight[replica] == least:
                break
        self._next = (replica + 1) % self.replicas
        return replica

    def choose(self, affinity: Affinity) -> int:
        replica = self._follow(affinity)
        if replica is None:
            replica = self._least_busy()
        if affinity.key is not None:
            self._remember(self._keys, affinity.key, replica)
        prompt = affinity.prompt
        if prompt is not None:
            anchor = prompt.anchors[-1]
            whole = (len(prompt.anchors) - 1) * prompt.block
            entries = [
                entry
                for entry in self._prompts.get(anchor, [])
                if entry[:2] != (prompt.length - whole, prompt.tail)
            ][-7:]
            entries.append((prompt.length - whole, prompt.tail, replica))
            self._remember(self._prompts, anchor, entries)
        return replica

    def remember_response(self, response_id: str, replica: int) -> None:
        self._remember(self._responses, response_id, replica)


_SAMPLE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{.*\})?(\s.*)$")


def _family(name: str, families: dict[str, list[str]]) -> str:
    if name in families:
        return name
    for suffix in ("_bucket", "_count", "_sum", "_total", "_created"):
        if name.endswith(suffix) and name[: -len(suffix)] in families:
            return name[: -len(suffix)]
    return name


def merge_metrics(texts: list[str]) -> str:
    """Merge Prometheus text expositions, labelling each sample with its replica."""
    headers: dict[str, list[str]] = {}
    samples: dict[str, list[str]] = {}
    order: list[str] = []
    for replica, text in enumerate(texts):
        current = None
        for line in text.splitlines():
            if not line.strip():
                continue
            if line.startswith("#"):
                parts = line.split(None, 3)
                if len(parts) >= 3 and parts[1] in ("HELP", "TYPE"):
                    current = parts[2]
                    if current not in headers:
                        headers[current] = []
                        samples[current] = []
                        order.append(current)
                    if line not in headers[current]:
                        headers[current].append(line)
                continue
            match = _SAMPLE.match(line)
            if match is None:
                continue
            name, labels, rest = match.groups()
            family = current if current is not None else _family(name, samples)
            if family not in samples:
                headers[family] = []
                samples[family] = []
                order.append(family)
            label = f'replica="{replica}"'
            labels = (
                f"{{{label}}}"
                if not labels or labels == "{}"
                else f"{{{label},{labels[1:]}"
            )
            samples[family].append(f"{name}{labels}{rest}")
    lines = []
    for family in order:
        lines.extend(headers[family])
        lines.extend(samples[family])
    return "\n".join(lines) + "\n"


def _forward_headers(headers: Any) -> dict[str, str]:
    return {
        name: value
        for name, value in headers.items()
        if name.lower() not in _HOP_BY_HOP
    }


def build_app(upstreams: list[str], *, router: StickyRouter | None = None):
    """Create the aiohttp application proxying to ``host:port`` upstreams."""
    import aiohttp
    from aiohttp import web

    router = router or StickyRouter(len(upstreams))
    if router.replicas != len(upstreams):
        raise ValueError("router replica count must match the upstreams")
    bases = [f"http://{upstream}" for upstream in upstreams]

    async def on_startup(app: web.Application) -> None:
        app["client"] = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=None, connect=30, sock_connect=30),
            auto_decompress=False,
        )

    async def on_cleanup(app: web.Application) -> None:
        await app["client"].close()

    async def fetch(app: web.Application, base: str, path: str) -> tuple[int, str]:
        try:
            async with app["client"].get(
                base + path, timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                return response.status, await response.text()
        except (TimeoutError, aiohttp.ClientError) as error:
            return 0, str(error)

    async def health(request: web.Request) -> web.Response:
        results = await asyncio.gather(
            *(fetch(request.app, base, "/health") for base in bases)
        )
        statuses = [status for status, _ in results]
        if all(status == 200 for status in statuses):
            return web.Response(status=200)
        return web.json_response(
            {"replicas": [{"replica": i, "status": s} for i, s in enumerate(statuses)]},
            status=503,
        )

    async def metrics(request: web.Request) -> web.Response:
        results = await asyncio.gather(
            *(fetch(request.app, base, "/metrics") for base in bases)
        )
        texts = [text if status == 200 else "" for status, text in results]
        return web.Response(
            text=merge_metrics(texts), content_type="text/plain", charset="utf-8"
        )

    async def proxy(request: web.Request) -> web.StreamResponse:
        body = await request.read()
        affinity = (
            request_affinity(request.path, body, dict(request.headers))
            if request.method == "POST"
            else Affinity()
        )
        match = re.search(r"/responses/(resp_[A-Za-z0-9_-]+)", request.path)
        if match:
            affinity = Affinity(previous_response_id=match.group(1))
        routed = (
            affinity.key is not None
            or affinity.prompt is not None
            or affinity.previous_response_id is not None
        )
        replica = router.choose(affinity) if routed else 0
        router.in_flight[replica] += 1
        response = None
        try:
            async with request.app["client"].request(
                request.method,
                bases[replica] + request.rel_url.path_qs,
                headers=_forward_headers(request.headers),
                data=body or None,
            ) as upstream:
                response = web.StreamResponse(
                    status=upstream.status,
                    reason=upstream.reason,
                    headers=_forward_headers(upstream.headers),
                )
                await response.prepare(request)
                watch = request.path.endswith("/responses")
                seen = b""
                async for chunk in upstream.content.iter_any():
                    if watch:
                        seen += chunk
                        found = _RESPONSE_ID.search(seen)
                        if found:
                            router.remember_response(found.group(1).decode(), replica)
                            watch = False
                        elif len(seen) > 65536:
                            watch = False
                    await response.write(chunk)
                await response.write_eof()
                return response
        except (aiohttp.ClientError, ConnectionResetError) as error:
            if response is not None:
                # The client left or the replica broke mid-stream; closing the
                # upstream connection lets the replica abort the request.
                return response
            return web.json_response(
                {"error": {"message": f"replica {replica} unavailable: {error}"}},
                status=502,
            )
        finally:
            router.in_flight[replica] -= 1

    app = web.Application(client_max_size=1 << 30)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    app.router.add_get("/health", health)
    app.router.add_get("/metrics", metrics)
    app.router.add_route("*", "/{tail:.*}", proxy)
    return app


def main(argv: list[str] | None = None) -> int:
    from aiohttp import web

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument(
        "--replica", action="append", required=True, metavar="HOST:PORT"
    )
    args = parser.parse_args(argv)
    print(
        f"[lil-serve] Proxying {args.host}:{args.port} to replicas "
        + ", ".join(args.replica),
        file=sys.stderr,
        flush=True,
    )
    web.run_app(
        build_app(args.replica),
        host=args.host,
        port=args.port,
        access_log=None,
        print=None,
        shutdown_timeout=5,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
