"""Per-1k-token pricing for common models (USD).

Prices are approximate list prices and may drift over time; update as needed.
Each entry has separate ``input`` and ``output`` per-1k-token costs.
"""

from typing import Optional

# Cost in USD per 1,000 tokens.
PRICING = {
    "gpt-4o": {"input": 0.0025, "output": 0.01},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    "claude-3-5-sonnet": {"input": 0.003, "output": 0.015},
    "claude-3-5-haiku": {"input": 0.0008, "output": 0.004},
}

# Aliases / substrings that map onto a canonical pricing key. The longest
# matching alias wins so e.g. "gpt-4o-mini" is not swallowed by "gpt-4o".
_ALIASES = {
    "gpt-4o-mini": "gpt-4o-mini",
    "gpt-4o": "gpt-4o",
    "gpt-3.5-turbo": "gpt-3.5-turbo",
    "gpt-35-turbo": "gpt-3.5-turbo",
    "claude-3-5-sonnet": "claude-3-5-sonnet",
    "claude-3.5-sonnet": "claude-3-5-sonnet",
    "claude-sonnet": "claude-3-5-sonnet",
    "claude-3-5-haiku": "claude-3-5-haiku",
    "claude-3.5-haiku": "claude-3-5-haiku",
    "claude-haiku": "claude-3-5-haiku",
}


def _resolve_key(model_name: str) -> Optional[str]:
    """Resolve a (possibly versioned) model name to a canonical pricing key."""
    if not model_name:
        return None
    name = model_name.lower()
    if name in PRICING:
        return name
    # Match the longest alias substring for stability.
    matches = [alias for alias in _ALIASES if alias in name]
    if not matches:
        return None
    best = max(matches, key=len)
    return _ALIASES[best]


def calculate_cost(
    model_name: Optional[str],
    input_tokens: Optional[int],
    output_tokens: Optional[int],
) -> Optional[float]:
    """Return the USD cost for a call, or ``None`` if it can't be computed.

    Returns ``None`` when the model is unknown or token counts are missing.
    """
    if not model_name or input_tokens is None or output_tokens is None:
        return None
    key = _resolve_key(model_name)
    if key is None:
        return None
    prices = PRICING[key]
    cost = (input_tokens / 1000.0) * prices["input"]
    cost += (output_tokens / 1000.0) * prices["output"]
    return round(cost, 8)
