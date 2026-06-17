"""AgentWatch SDK — drop-in observability for LangChain agents."""

from typing import Optional

from .client import DEFAULT_API_URL, AgentWatchClient

__version__ = "0.1.0"
__all__ = ["init", "AgentWatchClient", "DEFAULT_API_URL", "__version__"]


def init(api_url: str = DEFAULT_API_URL, api_key: Optional[str] = None) -> AgentWatchClient:
    """Create and return an AgentWatch client.

    Parameters
    ----------
    api_url:
        Base URL of the AgentWatch API. Defaults to the hosted instance.
    api_key:
        Optional API key for future authentication.
    """
    return AgentWatchClient(api_url=api_url, api_key=api_key)
