"""Minimal, runnable AgentWatch example — no API key required.

This builds a tiny agent out of langchain-core primitives only (a fake LLM plus
one tool), wraps it with AgentWatch, and runs a query. Because the agent is a
proper Runnable that invokes the LLM and tool with the propagated callbacks,
AgentWatch captures the LLM call, the tool call, and the session outcome.

Run with:
    python examples/basic_usage.py
"""

import agentwatch

# FakeListLLM moved to langchain_core.language_models.fake in langchain 0.3+.
try:  # newer/canonical location
    from langchain_core.language_models.fake import FakeListLLM
except ImportError:  # pragma: no cover - older alias
    from langchain_core.llms.fake import FakeListLLM

from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression like '2 + 2 * 3'."""
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:
        return "Error: only basic arithmetic is supported."
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))  # noqa: S307
    except Exception as exc:  # keep the demo robust
        return f"Error: {exc}"


# FakeListLLM replays canned responses in order — no network, no API key.
# Our "agent" reads the first response as the tool input to use.
llm = FakeListLLM(responses=["2 + 2 * 3", "The answer is 8."])


def run_agent(inputs: dict, config=None) -> dict:
    """A tiny ReAct-style loop: ask the LLM, call the tool, ask the LLM again.

    `config` carries the AgentWatch callbacks; passing it to each child
    `.invoke()` is what lets the LLM and tool calls get tracked.
    """
    question = inputs["input"]

    # 1. LLM decides what to compute.
    expression = llm.invoke(f"Question: {question}\nWhat should I calculate?", config)

    # 2. Run the tool.
    tool_result = calculator.invoke({"expression": expression}, config)

    # 3. LLM produces the final answer.
    final = llm.invoke(
        f"Question: {question}\nTool result: {tool_result}\nFinal answer:", config
    )
    return {"input": question, "output": final, "tool_result": tool_result}


# The agent is just a Runnable, so it has .invoke(), .stream(), etc.
agent = RunnableLambda(run_agent)


if __name__ == "__main__":
    # 1. Initialize AgentWatch (hosted API by default).
    aw = agentwatch.init(api_url="https://agentwatch-api.up.railway.app")

    # 2. Wrap the agent — drop-in, same interface.
    wrapped_agent = aw.wrap(agent, agent_version="example-v1", workspace_id="demo")

    # 3. Invoke it exactly like the original agent.
    result = wrapped_agent.invoke({"input": "What is 2 + 2 * 3?"})
    print("Result:", result)

    # Flush queued events before the process exits.
    aw.flush()

    # 6. Show the session id so you can find this run in the dashboard.
    print("AgentWatch session_id:", aw.session_id)
