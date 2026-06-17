"""AgentWatch client: session management and non-blocking event delivery."""

import atexit
import logging
import queue
import threading
import uuid
from typing import Any, Optional

import requests

logger = logging.getLogger("agentwatch")

DEFAULT_API_URL = "https://agentwatch-api.up.railway.app"

# Sentinel pushed onto the queue to tell the worker thread to shut down.
_SHUTDOWN = object()


class AgentWatchClient:
    """A client tied to a single session.

    All HTTP work happens on a background thread fed by an in-memory queue, so
    the wrapped agent is never blocked or broken by AgentWatch I/O.
    """

    def __init__(
        self,
        api_url: str = DEFAULT_API_URL,
        api_key: Optional[str] = None,
        timeout: float = 5.0,
    ):
        self.api_url = (api_url or DEFAULT_API_URL).rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.session_id = str(uuid.uuid4())

        self._queue: "queue.Queue[Any]" = queue.Queue()
        self._session_started = False
        self._lock = threading.Lock()

        self._session = requests.Session()
        if self.api_key:
            self._session.headers.update({"Authorization": f"Bearer {self.api_key}"})
        self._session.headers.update({"Content-Type": "application/json"})

        self._worker = threading.Thread(
            target=self._run, name="agentwatch-worker", daemon=True
        )
        self._worker.start()
        atexit.register(self.flush)

    # -- public API ---------------------------------------------------------

    def wrap(
        self,
        agent_executor: Any,
        agent_version: Optional[str] = None,
        workspace_id: Optional[str] = None,
    ) -> Any:
        """Wrap a LangChain Runnable so its activity is tracked.

        Returns an object that behaves like the original (``invoke``,
        ``stream``, etc.) but injects an AgentWatch callback handler and starts
        the session on first use.
        """
        # Imported lazily so importing agentwatch doesn't hard-require langchain.
        from .callbacks import AgentWatchCallbackHandler, WrappedRunnable

        model_version = _detect_model_version(agent_executor)
        self._start_session(
            agent_version=agent_version,
            workspace_id=workspace_id,
            model_version=model_version,
        )
        handler = AgentWatchCallbackHandler(self)
        return WrappedRunnable(agent_executor, handler)

    def _start_session(
        self,
        agent_version: Optional[str] = None,
        workspace_id: Optional[str] = None,
        model_version: Optional[str] = None,
    ) -> None:
        """Enqueue the session-start POST exactly once."""
        with self._lock:
            if self._session_started:
                return
            self._session_started = True
        payload = {
            "session_id": self.session_id,
            "agent_version": agent_version,
            "workspace_id": workspace_id,
            "model_version": model_version,
        }
        self._enqueue(("POST", "/sessions", payload))

    def send_event(self, event: dict) -> None:
        """Enqueue an event for delivery to ``/events``."""
        self._enqueue(("POST", "/events", event))

    def flush(self, timeout: Optional[float] = 5.0) -> None:
        """Block until queued items are sent (or timeout). Safe to call twice."""
        try:
            self._queue.join()
        except Exception:  # pragma: no cover - defensive
            pass

    # -- background worker --------------------------------------------------

    def _enqueue(self, item: Any) -> None:
        try:
            self._queue.put_nowait(item)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("agentwatch: failed to enqueue event: %s", exc)

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is _SHUTDOWN:
                    return
                method, path, payload = item
                self._post(path, payload)
            except Exception as exc:  # never let the worker die
                logger.warning("agentwatch: failed to send event: %s", exc)
            finally:
                self._queue.task_done()

    def _post(self, path: str, payload: dict) -> None:
        url = f"{self.api_url}{path}"
        try:
            self._session.post(url, json=payload, timeout=self.timeout)
        except Exception as exc:
            # Fail silently; AgentWatch must never break the host agent.
            logger.warning("agentwatch: POST %s failed: %s", url, exc)


def _detect_model_version(agent_executor: Any) -> Optional[str]:
    """Best-effort extraction of a model name from a Runnable/agent."""
    candidates = []

    def _scan(obj: Any, depth: int = 0) -> None:
        if obj is None or depth > 4:
            return
        for attr in ("model_name", "model", "model_id", "deployment_name"):
            val = getattr(obj, attr, None)
            if isinstance(val, str) and val:
                candidates.append(val)
        # Walk common nesting points without importing langchain types.
        for attr in ("llm", "bound", "runnable", "agent", "steps", "middle", "first", "last"):
            child = getattr(obj, attr, None)
            if child is None:
                continue
            if isinstance(child, (list, tuple)):
                for c in child:
                    _scan(c, depth + 1)
            else:
                _scan(child, depth + 1)

    try:
        _scan(agent_executor)
    except Exception:  # pragma: no cover - defensive
        return None
    return candidates[0] if candidates else None
