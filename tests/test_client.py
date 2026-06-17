"""Tests for AgentWatch client and callback handler.

HTTP is mocked; we assert that the right payloads get enqueued and POSTed.
"""

import time
from unittest import mock

import pytest

import agentwatch
from agentwatch.callbacks import AgentWatchCallbackHandler, WrappedRunnable
from agentwatch.pricing import calculate_cost


def _wait_for_queue(client, timeout=2.0):
    """Wait until the background worker drains the queue."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if client._queue.unfinished_tasks == 0:
            return
        time.sleep(0.01)


@pytest.fixture
def client():
    with mock.patch("agentwatch.client.requests.Session") as SessionCls:
        instance = SessionCls.return_value
        instance.headers = {}
        instance.post = mock.MagicMock()
        c = agentwatch.init(api_url="http://test.local")
        c._mock_post = instance.post  # expose for assertions
        yield c


def test_init_generates_session_id():
    c = agentwatch.init()
    assert c.session_id
    assert c.api_url == "https://agentwatch-api.up.railway.app"


def test_init_strips_trailing_slash():
    c = agentwatch.init(api_url="http://test.local/")
    assert c.api_url == "http://test.local"


def test_start_session_posts_once(client):
    client._start_session(agent_version="v1", workspace_id="ws", model_version="gpt-4o")
    client._start_session()  # second call is a no-op
    _wait_for_queue(client)

    session_posts = [
        call for call in client._mock_post.call_args_list
        if call.args and call.args[0].endswith("/sessions")
    ]
    assert len(session_posts) == 1
    payload = session_posts[0].kwargs["json"]
    assert payload["session_id"] == client.session_id
    assert payload["agent_version"] == "v1"
    assert payload["workspace_id"] == "ws"
    assert payload["model_version"] == "gpt-4o"


def test_send_event_enqueues_and_posts(client):
    event = {"session_id": client.session_id, "event_type": "llm_call"}
    client.send_event(event)
    _wait_for_queue(client)

    event_posts = [
        call for call in client._mock_post.call_args_list
        if call.args and call.args[0] == "http://test.local/events"
    ]
    assert len(event_posts) == 1
    assert event_posts[0].kwargs["json"] == event


def test_post_failure_is_silent(client):
    client._mock_post.side_effect = RuntimeError("network down")
    client.send_event({"session_id": client.session_id})
    _wait_for_queue(client)
    # Worker thread should still be alive despite the exception.
    assert client._worker.is_alive()


# -- callback handler ------------------------------------------------------


class _Recorder:
    """Minimal client stand-in that records events instead of sending them."""

    def __init__(self):
        self.session_id = "test-session"
        self.events = []

    def send_event(self, event):
        self.events.append(event)


def test_tool_call_event_captured():
    rec = _Recorder()
    handler = AgentWatchCallbackHandler(rec)
    run_id = "run-1"

    handler.on_tool_start({"name": "search"}, "what is the weather", run_id=run_id)
    handler.on_tool_end("sunny", run_id=run_id)

    assert len(rec.events) == 1
    ev = rec.events[0]
    assert ev["event_type"] == "tool_call"
    assert ev["event_name"] == "search"
    assert ev["status"] == "success"
    assert ev["payload"]["input"] == "what is the weather"
    assert ev["payload"]["output"] == "sunny"
    assert ev["latency_ms"] is not None


def test_tool_error_event_captured():
    rec = _Recorder()
    handler = AgentWatchCallbackHandler(rec)
    handler.on_tool_start({"name": "search"}, "boom", run_id="r")
    handler.on_tool_error(ValueError("bad"), run_id="r")

    ev = rec.events[0]
    assert ev["status"] == "error"
    assert "bad" in ev["payload"]["error"]


def test_llm_event_with_tokens_and_cost():
    rec = _Recorder()
    handler = AgentWatchCallbackHandler(rec)

    response = mock.MagicMock()
    response.llm_output = {
        "model_name": "gpt-4o-mini",
        "token_usage": {
            "prompt_tokens": 1000,
            "completion_tokens": 1000,
            "total_tokens": 2000,
        },
    }
    response.generations = []

    handler.on_llm_start({"name": "gpt-4o-mini"}, ["hi"], run_id="llm-1")
    handler.on_llm_end(response, run_id="llm-1")

    ev = rec.events[0]
    assert ev["event_type"] == "llm_call"
    assert ev["event_name"] == "gpt-4o-mini"
    assert ev["payload"]["input_tokens"] == 1000
    assert ev["payload"]["output_tokens"] == 1000
    # 1k input * 0.00015 + 1k output * 0.0006
    assert ev["cost_usd"] == pytest.approx(0.00075)


def test_chain_end_emits_session_outcome():
    rec = _Recorder()
    handler = AgentWatchCallbackHandler(rec)
    handler.on_chain_start({}, {}, run_id="c1", parent_run_id=None)
    handler.on_chain_end({"output": "done"}, run_id="c1", parent_run_id=None)

    assert len(rec.events) == 1
    assert rec.events[0]["event_type"] == "session_outcome"
    assert rec.events[0]["status"] == "success"


def test_nested_chain_does_not_emit_outcome():
    rec = _Recorder()
    handler = AgentWatchCallbackHandler(rec)
    handler.on_chain_end({}, run_id="child", parent_run_id="parent")
    assert rec.events == []


def test_payload_truncation():
    rec = _Recorder()
    handler = AgentWatchCallbackHandler(rec)
    big = "x" * 1000
    handler.on_tool_start({"name": "t"}, big, run_id="r")
    handler.on_tool_end(big, run_id="r")
    assert len(rec.events[0]["payload"]["output"]) < 600
    assert rec.events[0]["payload"]["output"].endswith("...[truncated]")


# -- wrapper ---------------------------------------------------------------


def test_wrapped_runnable_injects_handler_and_delegates():
    inner = mock.MagicMock()
    inner.invoke.return_value = {"output": "ok"}
    inner.some_attr = 42

    handler = AgentWatchCallbackHandler(_Recorder())
    wrapped = WrappedRunnable(inner, handler)

    result = wrapped.invoke({"input": "q"})
    assert result == {"output": "ok"}

    # handler must be passed through config callbacks
    _, kwargs = inner.invoke.call_args[0], inner.invoke.call_args
    config = inner.invoke.call_args[0][1]
    assert handler in config["callbacks"]

    # arbitrary attributes delegate to the wrapped object
    assert wrapped.some_attr == 42


def test_wrap_merges_existing_callbacks():
    inner = mock.MagicMock()
    handler = AgentWatchCallbackHandler(_Recorder())
    wrapped = WrappedRunnable(inner, handler)

    existing = mock.MagicMock()
    wrapped.invoke({"input": "q"}, config={"callbacks": [existing]})
    config = inner.invoke.call_args[0][1]
    assert existing in config["callbacks"]
    assert handler in config["callbacks"]


# -- pricing ---------------------------------------------------------------


def test_pricing_known_model():
    assert calculate_cost("gpt-4o", 1000, 1000) == pytest.approx(0.0125)


def test_pricing_alias_versioned_name():
    # versioned name should resolve via alias substring matching
    assert calculate_cost("claude-3-5-sonnet-20241022", 1000, 0) == pytest.approx(0.003)


def test_pricing_unknown_model_returns_none():
    assert calculate_cost("mystery-model", 100, 100) is None


def test_pricing_missing_tokens_returns_none():
    assert calculate_cost("gpt-4o", None, None) is None
