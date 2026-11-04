"""LLM client: every model call is logged, and outputs are cached for replay.

Backends implement complete(messages, params) -> Completion:

    StubLLM          deterministic function of the messages; no network
    OpenAICompatLLM  any OpenAI-compatible server (llama.cpp, vLLM, ...)
    OnyxLLM          TritonGPT's Onyx chat API (used as the TAMAS judge)

LoggedLLM wraps a backend, logs each call as an llm_call event and caches the
output under a hash of (model, messages, params), so re-running an unchanged
call returns the logged output instead of a new sample.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from .log import Kind, Log, canonical, sha256

Messages = list[dict[str, str]]


class LLMError(Exception):
    pass


@dataclass(frozen=True)
class Completion:
    text: str
    prompt_tokens: int
    completion_tokens: int


class StubLLM:
    """Scripted model for tests and dry runs. Token counts are word counts."""

    def __init__(self, fn: Callable[[Messages], str], model: str = "stub"):
        self.fn = fn
        self.model = model

    def complete(self, messages: Messages, params: dict) -> Completion:
        text = self.fn(messages)
        return Completion(text, sum(len(m["content"].split()) for m in messages),
                          len(text.split()))


class OpenAICompatLLM:
    def __init__(self, base_url: str, model: str, api_key: str = "none",
                 timeout: float = 120.0, http_client: Any = None):
        from openai import OpenAI
        self.model = model
        # No silent retries: a retry would hide failures and distort cost.
        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout,
                              max_retries=0, http_client=http_client)

    def complete(self, messages: Messages, params: dict) -> Completion:
        r = self._client.chat.completions.create(model=self.model, messages=messages, **params)
        u = r.usage
        return Completion(r.choices[0].message.content or "",
                          u.prompt_tokens if u else 0, u.completion_tokens if u else 0)


def parse_onyx_stream(lines: Iterable[str]) -> str:
    """Answer text from an Onyx send-chat-message stream (one JSON object per line)."""
    text = []
    for line in lines:
        if not line.strip():
            continue
        obj = json.loads(line)
        if "error" in obj:
            raise LLMError(obj["error"][:300])
        o = obj.get("obj", {})
        if o.get("type") in ("message_start", "message_delta"):
            text.append(o.get("content") or "")
    return "".join(text)


class OnyxLLM:
    """TritonGPT (an Onyx server). Its chat API takes one message, so the system and
    user turns are sent together; each call uses a fresh chat that is deleted after.
    Onyx does not report token usage, so counts are 0."""

    def __init__(self, base_url: str, model: str, token: str, provider: str = "LiteLLM Prod",
                 timeout: float = 300.0):
        import httpx
        self.model, self.provider = model, provider
        self._http = httpx.Client(base_url=base_url, timeout=timeout,
                                  headers={"Authorization": f"Bearer {token}"})

    def complete(self, messages: Messages, params: dict) -> Completion:
        message = "\n\n".join(m["content"] for m in messages)
        sid = self._http.post("/api/chat/create-chat-session",
                              json={"persona_id": 0, "description": "cone-judge"}
                              ).raise_for_status().json()["chat_session_id"]
        try:
            body = {"chat_session_id": sid, "parent_message_id": None, "message": message,
                    "stream": True, "llm_override": {"model_provider": self.provider,
                                                     "model_version": self.model,
                                                     "temperature": params.get("temperature", 0)}}
            with self._http.stream("POST", "/api/chat/send-chat-message", json=body) as r:
                r.raise_for_status()
                text = parse_onyx_stream(r.iter_lines())
        finally:
            self._http.delete(f"/api/chat/delete-chat-session/{sid}")
        return Completion(text, 0, 0)


class LoggedLLM:
    def __init__(self, backend, log: Log, params: dict | None = None,
                 cache: dict[str, dict] | None = None):
        self.backend = backend
        self.log = log
        self.params = {"temperature": 0, **(params or {})}
        self.cache = cache if cache is not None else {}
        self.stats = {"calls": 0, "cache_hits": 0, "errors": 0,
                      "prompt_tokens": 0, "completion_tokens": 0, "seconds": 0.0}

    def key(self, messages: Messages, params: dict) -> str:
        return sha256(canonical({"model": self.backend.model, "messages": messages,
                                 "params": params}))

    def call(self, agent: str, messages: Messages, deps: Iterable[int] = (),
             use_cache: bool = True, **params) -> tuple[str, int]:
        """Return (output text, seq of the llm_call event)."""
        params = {**self.params, **params}
        key = self.key(messages, params)
        record = {"model": self.backend.model, "messages": messages, "params": params,
                  "key": key}
        hit = use_cache and key in self.cache
        start = time.perf_counter()
        try:
            c = Completion(**self.cache[key]) if hit else self.backend.complete(messages, params)
        except Exception as exc:
            seconds = time.perf_counter() - start
            self.stats["calls"] += 1
            self.stats["errors"] += 1
            self.stats["seconds"] += seconds
            self.log.append(agent, Kind.LLM_CALL,
                            {**record, "error": repr(exc), "seconds": round(seconds, 4)}, deps)
            raise LLMError(f"{agent}: model call failed: {exc!r}") from exc
        seconds = time.perf_counter() - start
        if not hit:
            self.cache[key] = asdict(c)
        self.stats["calls"] += 1
        if hit:
            self.stats["cache_hits"] += 1
        else:
            self.stats["prompt_tokens"] += c.prompt_tokens
            self.stats["completion_tokens"] += c.completion_tokens
            self.stats["seconds"] += seconds
        ev = self.log.append(agent, Kind.LLM_CALL, {
            **record, "output": c.text, "cached": hit, "seconds": round(seconds, 4),
            "usage": {"prompt_tokens": c.prompt_tokens,
                      "completion_tokens": c.completion_tokens}}, deps)
        return c.text, ev.seq


def save_cache(cache: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(cache, sort_keys=True, ensure_ascii=False),
                          encoding="utf-8")


def load_cache(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
