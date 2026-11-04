import json

import httpx
import pytest

from cone.llm import (LLMError, LoggedLLM, OpenAICompatLLM, StubLLM, load_cache,
                      save_cache)
from cone.log import Kind, Log


def echo(messages):
    return "echo: " + messages[-1]["content"]


def msgs(text):
    return [{"role": "user", "content": text}]


def test_call_is_logged_with_usage():
    log = Log()
    llm = LoggedLLM(StubLLM(echo), log)
    text, seq = llm.call("researcher", msgs("find facts"))
    assert text == "echo: find facts"
    p = log.payload(seq)
    assert log.events[seq].kind is Kind.LLM_CALL
    assert p["output"] == text and p["cached"] is False
    assert p["params"] == {"temperature": 0}
    assert p["usage"] == {"prompt_tokens": 2, "completion_tokens": 3}
    log.verify()


def test_cache_returns_logged_output_exactly():
    calls = []

    def counting(messages):
        calls.append(1)
        return f"sample {len(calls)}"

    log = Log()
    llm = LoggedLLM(StubLLM(counting), log)
    first, s1 = llm.call("a", msgs("q"))
    again, s2 = llm.call("a", msgs("q"))
    fresh, _ = llm.call("a", msgs("q"), use_cache=False)
    assert again == first == log.payload(s1)["output"]
    assert log.payload(s2)["cached"] is True
    assert fresh == "sample 2"
    assert len(calls) == 2
    assert llm.stats["calls"] == 3 and llm.stats["cache_hits"] == 1


def test_params_and_model_are_part_of_the_key():
    llm = LoggedLLM(StubLLM(echo), Log())
    k = llm.key(msgs("q"), {"temperature": 0})
    assert k != llm.key(msgs("q"), {"temperature": 0.7})
    assert k != LoggedLLM(StubLLM(echo, model="other"), Log()).key(msgs("q"), {"temperature": 0})


def test_failed_call_is_logged_and_raised():
    def broken(messages):
        raise TimeoutError("server timed out")

    log = Log()
    llm = LoggedLLM(StubLLM(broken), log)
    with pytest.raises(LLMError, match="timed out"):
        llm.call("analyst", msgs("q"), deps=())
    assert "TimeoutError" in log.payload(0)["error"]
    assert llm.stats["errors"] == 1


def test_cache_persists(tmp_path):
    llm = LoggedLLM(StubLLM(echo), Log())
    llm.call("a", msgs("q"))
    save_cache(llm.cache, tmp_path / "cache.json")
    again = LoggedLLM(StubLLM(lambda m: "different"), Log(),
                      cache=load_cache(tmp_path / "cache.json"))
    assert again.call("a", msgs("q"))[0] == "echo: q"


def test_openai_compatible_backend_without_network():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "id": "x", "object": "chat.completion", "created": 0, "model": "m",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "hi"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}})

    backend = OpenAICompatLLM("http://model.test/v1", "m",
                              http_client=httpx.Client(transport=httpx.MockTransport(handler)))
    log = Log()
    text, seq = LoggedLLM(backend, log, params={"seed": 7}).call("a", msgs("q"))
    assert text == "hi"
    assert seen["url"] == "http://model.test/v1/chat/completions"
    assert seen["body"]["temperature"] == 0 and seen["body"]["seed"] == 7
    assert log.payload(seq)["usage"] == {"prompt_tokens": 5, "completion_tokens": 1}


def test_onyx_stream_parsing_and_errors():
    from cone.llm import parse_onyx_stream
    lines = ['{"user_message_id": 1}', '{"obj": {"type": "chat_heartbeat"}}',
             '{"obj": {"type": "message_start", "content": "<answer>"}}',
             '{"obj": {"type": "message_delta", "content": "0</answer>"}}', '']
    assert parse_onyx_stream(lines) == "<answer>0</answer>"
    with pytest.raises(LLMError):
        parse_onyx_stream(['{"error": "model group failed"}'])


def test_judge_label_parsing():
    from cone.judge_tamas import parse_label
    assert parse_label("<answer>\n    0\n    </answer>") == 0
    assert parse_label("I think 3") is None
