"""LangChain callback handler and Runnable wrapper for AgentWatch."""

import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from .pricing import calculate_cost

logger = logging.getLogger("agentwatch")

try:
    from langchain_core.callbacks import BaseCallbackHandler
except Exception:  # pragma: no cover - langchain-core not installed
    class BaseCallbackHandler:  # type: ignore
        """Fallback no-op base so the module imports without langchain."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _truncate(value: Any, limit: int = 500) -> Any:
    """Stringify and truncate a value to ``limit`` characters."""
    if value is None:
        return None
    text = value if isinstance(value, str) else str(value)
    if len(text) > limit:
        return text[:limit] + "...[truncated]"
    return text


class AgentWatchCallbackHandler(BaseCallbackHandler):
    """Captures LLM and tool events and forwards them to an AgentWatch client."""

    def __init__(self, client: Any):
        super().__init__()
        self._client = client
        # run_id -> {"start": float, "name": str, "input": str}
        self._llm_runs: Dict[UUID, Dict[str, Any]] = {}
        self._tool_runs: Dict[UUID, Dict[str, Any]] = {}

    # -- helpers ------------------------------------------------------------

    def _emit(self, event: dict) -> None:
        try:
            self._client.send_event(event)
        except Exception as exc:  # never break the agent
            logger.warning("agentwatch: failed to emit event: %s", exc)

    @staticmethod
    def _latency_ms(start: Optional[float]) -> Optional[int]:
        if start is None:
            return None
        return int((time.perf_counter() - start) * 1000)

    # -- LLM callbacks ------------------------------------------------------

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        try:
            name = self._extract_model_name(serialized, kwargs)
            self._llm_runs[run_id] = {"start": time.perf_counter(), "name": name}
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("agentwatch: on_llm_start error: %s", exc)

    # chat models route through on_chat_model_start
    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: Any,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self.on_llm_start(serialized, [], run_id=run_id, **kwargs)

    def on_llm_end(self, response: Any, *, run_id: UUID, **kwargs: Any) -> None:
        try:
            run = self._llm_runs.pop(run_id, None)
            start = run["start"] if run else None
            name = run["name"] if run else None

            input_tokens, output_tokens, total_tokens = self._extract_tokens(response)
            model = self._model_from_response(response) or name

            self._emit(
                {
                    "session_id": self._client.session_id,
                    "event_type": "llm_call",
                    "event_name": model,
                    "timestamp": _now_iso(),
                    "latency_ms": self._latency_ms(start),
                    "cost_usd": calculate_cost(model, input_tokens, output_tokens),
                    "status": "success",
                    "payload": {
                        "model": model,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": total_tokens,
                    },
                }
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("agentwatch: on_llm_end error: %s", exc)

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        try:
            run = self._llm_runs.pop(run_id, None)
            start = run["start"] if run else None
            name = run["name"] if run else None
            self._emit(
                {
                    "session_id": self._client.session_id,
                    "event_type": "llm_call",
                    "event_name": name,
                    "timestamp": _now_iso(),
                    "latency_ms": self._latency_ms(start),
                    "cost_usd": None,
                    "status": "error",
                    "payload": {"model": name, "error": _truncate(error)},
                }
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("agentwatch: on_llm_error error: %s", exc)

    # -- Tool callbacks -----------------------------------------------------

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        try:
            name = (serialized or {}).get("name") or kwargs.get("name") or "tool"
            self._tool_runs[run_id] = {
                "start": time.perf_counter(),
                "name": name,
                "input": _truncate(input_str),
            }
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("agentwatch: on_tool_start error: %s", exc)

    def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        try:
            run = self._tool_runs.pop(run_id, None)
            start = run["start"] if run else None
            name = run["name"] if run else None
            tool_input = run["input"] if run else None
            self._emit(
                {
                    "session_id": self._client.session_id,
                    "event_type": "tool_call",
                    "event_name": name,
                    "timestamp": _now_iso(),
                    "latency_ms": self._latency_ms(start),
                    "cost_usd": None,
                    "status": "success",
                    "payload": {
                        "tool": name,
                        "input": tool_input,
                        "output": _truncate(output),
                    },
                }
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("agentwatch: on_tool_end error: %s", exc)

    def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        try:
            run = self._tool_runs.pop(run_id, None)
            start = run["start"] if run else None
            name = run["name"] if run else None
            tool_input = run["input"] if run else None
            self._emit(
                {
                    "session_id": self._client.session_id,
                    "event_type": "tool_call",
                    "event_name": name,
                    "timestamp": _now_iso(),
                    "latency_ms": self._latency_ms(start),
                    "cost_usd": None,
                    "status": "error",
                    "payload": {
                        "tool": name,
                        "input": tool_input,
                        "error": _truncate(error),
                    },
                }
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("agentwatch: on_tool_error error: %s", exc)

    # -- Chain callbacks (session outcome) ----------------------------------

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        # Only the top-level chain (no parent) represents the agent run.
        if parent_run_id is None:
            self._root_run_id = run_id

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        if parent_run_id is None:
            self._emit_session_outcome("success")

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        if parent_run_id is None:
            self._emit_session_outcome("error", error)

    def _emit_session_outcome(
        self, status: str, error: Optional[BaseException] = None
    ) -> None:
        try:
            payload: Dict[str, Any] = {"outcome": status}
            if error is not None:
                payload["error"] = _truncate(error)
            self._emit(
                {
                    "session_id": self._client.session_id,
                    "event_type": "session_outcome",
                    "event_name": status,
                    "timestamp": _now_iso(),
                    "latency_ms": None,
                    "cost_usd": None,
                    "status": status,
                    "payload": payload,
                }
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("agentwatch: session outcome error: %s", exc)

    # -- extraction helpers -------------------------------------------------

    @staticmethod
    def _extract_model_name(serialized: Dict[str, Any], kwargs: Dict[str, Any]) -> Optional[str]:
        invocation = kwargs.get("invocation_params") or {}
        for key in ("model", "model_name", "model_id", "deployment_name"):
            val = invocation.get(key)
            if isinstance(val, str) and val:
                return val
        serialized = serialized or {}
        kwargs_block = serialized.get("kwargs") or {}
        for key in ("model", "model_name", "model_id"):
            val = kwargs_block.get(key)
            if isinstance(val, str) and val:
                return val
        return serialized.get("name")

    @staticmethod
    def _model_from_response(response: Any) -> Optional[str]:
        try:
            output = getattr(response, "llm_output", None) or {}
            if isinstance(output, dict):
                name = output.get("model_name") or output.get("model")
                if isinstance(name, str) and name:
                    return name
        except Exception:  # pragma: no cover - defensive
            pass
        return None

    @staticmethod
    def _extract_tokens(response: Any):
        """Return (input_tokens, output_tokens, total_tokens) best-effort."""
        input_tokens = output_tokens = total_tokens = None
        try:
            llm_output = getattr(response, "llm_output", None) or {}
            usage = {}
            if isinstance(llm_output, dict):
                usage = (
                    llm_output.get("token_usage")
                    or llm_output.get("usage")
                    or {}
                )
            if usage:
                input_tokens = usage.get("prompt_tokens") or usage.get("input_tokens")
                output_tokens = usage.get("completion_tokens") or usage.get("output_tokens")
                total_tokens = usage.get("total_tokens")

            # Newer langchain surfaces usage_metadata on the generation message.
            if input_tokens is None and output_tokens is None:
                generations = getattr(response, "generations", None) or []
                for gen_list in generations:
                    for gen in gen_list:
                        message = getattr(gen, "message", None)
                        meta = getattr(message, "usage_metadata", None)
                        if meta:
                            input_tokens = meta.get("input_tokens")
                            output_tokens = meta.get("output_tokens")
                            total_tokens = meta.get("total_tokens")
                            raise StopIteration
        except StopIteration:
            pass
        except Exception:  # pragma: no cover - defensive
            pass
        if total_tokens is None and input_tokens is not None and output_tokens is not None:
            total_tokens = input_tokens + output_tokens
        return input_tokens, output_tokens, total_tokens


class WrappedRunnable:
    """Transparent proxy around a Runnable that injects the AgentWatch handler.

    Behaves like the wrapped object for ``invoke``/``ainvoke``/``stream``/
    ``astream``/``batch``; any other attribute is delegated unchanged.
    """

    def __init__(self, runnable: Any, handler: AgentWatchCallbackHandler):
        self._aw_runnable = runnable
        self._aw_handler = handler

    def _with_callbacks(self, config: Optional[dict]) -> dict:
        config = dict(config) if config else {}
        callbacks = config.get("callbacks")
        if callbacks is None:
            config["callbacks"] = [self._aw_handler]
        elif isinstance(callbacks, list):
            config["callbacks"] = callbacks + [self._aw_handler]
        else:
            # CallbackManager or handler list-like: best effort add.
            try:
                callbacks.add_handler(self._aw_handler)
            except Exception:
                config["callbacks"] = [self._aw_handler]
        return config

    def invoke(self, input: Any, config: Optional[dict] = None, **kwargs: Any) -> Any:
        return self._aw_runnable.invoke(input, self._with_callbacks(config), **kwargs)

    async def ainvoke(self, input: Any, config: Optional[dict] = None, **kwargs: Any) -> Any:
        return await self._aw_runnable.ainvoke(input, self._with_callbacks(config), **kwargs)

    def stream(self, input: Any, config: Optional[dict] = None, **kwargs: Any) -> Any:
        return self._aw_runnable.stream(input, self._with_callbacks(config), **kwargs)

    async def astream(self, input: Any, config: Optional[dict] = None, **kwargs: Any) -> Any:
        async for chunk in self._aw_runnable.astream(
            input, self._with_callbacks(config), **kwargs
        ):
            yield chunk

    def batch(self, inputs: List[Any], config: Optional[dict] = None, **kwargs: Any) -> Any:
        return self._aw_runnable.batch(inputs, self._with_callbacks(config), **kwargs)

    def __getattr__(self, name: str) -> Any:
        # Delegate everything else to the wrapped runnable.
        return getattr(self._aw_runnable, name)
