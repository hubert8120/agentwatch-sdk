# AgentWatch

**The reliability layer for AI agents.**

Drop-in observability for [LangChain](https://www.langchain.com/) agents built for agencies and software houses that deploy AI agents for B2B clients.

Wrap your agent once — AgentWatch automatically tracks sessions, LLM calls, tool calls, costs, and outcomes. No code changes to your agent logic.

```bash
pip install useagentwatch
```

---

## Why AgentWatch

Most production AI agents are flying blind. You get a monthly bill from OpenAI but no answer to:

- Which client's agent caused that $800 spike?
- Why did 30% of sessions return garbage last Tuesday?
- What did the agent actually do — step by step?

AgentWatch answers all three.

---

## Quick start

```python
import agentwatch

aw = agentwatch.init(
    api_url="https://agentwatch-api.up.railway.app",
    api_key="your-api-key"
)

wrapped_agent = aw.wrap(your_langchain_agent)
wrapped_agent.invoke({"input": "..."})  # use exactly as before
```

3 lines. That's it. Every LLM call, tool call, and session outcome is now tracked automatically.

---

## What gets tracked

| Event | When | Captured |
|---|---|---|
| **Session start** | `wrap()` is called | `session_id`, `agent_version`, `workspace_id`, `model_version` |
| **LLM call** | each model invocation | model name, input/output tokens, latency, estimated cost |
| **Tool call** | each tool invocation | tool name, input, output (truncated to 500 chars), latency, success/error |
| **Session outcome** | agent finishes | success or error |

### Event schema

```json
{
  "session_id": "…",
  "event_type": "llm_call | tool_call | session_outcome",
  "event_name": "gpt-4o-mini",
  "timestamp": "2026-06-12T09:20:00+00:00",
  "latency_ms": 842,
  "cost_usd": 0.00075,
  "status": "success | error",
  "payload": { "model": "…", "input_tokens": 1000, "output_tokens": 1000 }
}
```

### Cost estimation

Built-in pricing table covers `gpt-4o`, `gpt-4o-mini`, `gpt-3.5-turbo`, `claude-3-5-sonnet`, and `claude-3-5-haiku`. Versioned names (e.g. `claude-3-5-sonnet-20241022`) resolve automatically. Unknown models return `cost_usd: null`.

---

## Configuration

```python
aw = agentwatch.init(
    api_url="https://agentwatch-api.up.railway.app",
    api_key="your-api-key",
)

wrapped = aw.wrap(
    agent,
    agent_version="v1",
    workspace_id="client-acme"  # per-client attribution
)
```

The wrapped agent is a transparent proxy — `.invoke()`, `.ainvoke()`, `.stream()`, `.astream()`, and `.batch()` all work exactly as before. Every other attribute is delegated through unchanged.

---

## Non-blocking by design

All HTTP calls run on a background worker thread fed by an in-memory queue. AgentWatch never blocks your agent and never raises into it — failures are caught and logged as warnings.

```python
import logging
logging.getLogger("agentwatch").setLevel(logging.WARNING)
```

Call `aw.flush()` before your process exits to ensure queued events are sent (registered automatically via `atexit`).

---

## Ecosystem

AgentWatch is part of a reliability stack for production AI agents:

- **[FiGuard](https://github.com/figuard/figuard-core)** — budget enforcement before tool calls (FiGuard enforces → AgentWatch observes)
- **AgentWatch** — observability, cost tracking, session reports
- **Armorer** — recovery decisions after failures

---

## API

AgentWatch SDK talks to the AgentWatch API:

- `POST /sessions` — create a session
- `POST /events` — send an event
- `GET /report/{session_id}` — get a session report

API docs: [agentwatch-api.up.railway.app/docs](https://agentwatch-api.up.railway.app/docs)

---

## Development

```bash
git clone https://github.com/hubert8120/agentwatch-sdk.git
cd agentwatch-sdk
pip install -e ".[dev]"
pytest
```

---

## Links

- **Landing page:** [agentwatch-two.vercel.app](https://agentwatch-two.vercel.app)
- **PyPI:** [pypi.org/project/useagentwatch](https://pypi.org/project/useagentwatch)
- **API docs:** [agentwatch-api.up.railway.app/docs](https://agentwatch-api.up.railway.app/docs)
- **Early access:** [agentwatch-two.vercel.app](https://agentwatch-two.vercel.app)

---

*Built for agencies that deploy AI agents for clients — not for internal dev teams.*
