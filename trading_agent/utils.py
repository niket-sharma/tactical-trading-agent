from __future__ import annotations


def clamp_transaction_cost_bps(bps: float) -> float:
    """Clamp transaction costs to a sane range."""
    return max(0.0, min(bps, 1000.0))
