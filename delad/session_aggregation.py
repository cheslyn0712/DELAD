"""Session-level score aggregation over transition anomaly probabilities."""

from __future__ import annotations

from typing import Literal

import torch

SESSION_AGGREGATION_METHODS = ("max", "topk_mean", "noisy_or")
SessionAggregationMethod = Literal["max", "topk_mean", "noisy_or"]


def normalize_session_aggregation_method(method: str) -> SessionAggregationMethod:
    """Normalize aggregation method name."""

    key = str(method or "max").strip().lower()
    if key in SESSION_AGGREGATION_METHODS:
        return key
    raise ValueError(
        f"Unsupported session_aggregation_method={method!r}; "
        f"expected one of {SESSION_AGGREGATION_METHODS}"
    )


def aggregate_transition_probabilities(
    probs: torch.Tensor,
    mask: torch.Tensor,
    method: SessionAggregationMethod,
    topk: int = 3,
) -> torch.Tensor:
    """Aggregate per-transition anomaly probabilities into session scores.

    Args:
        probs: [B, T] anomaly probabilities p_{c,t}.
        mask: [B, T] bool mask for valid transitions.
        method: max | topk_mean | noisy_or.
        topk: k for top-k mean pooling.

    Returns:
        [B] session scores s_c in [0, 1].
    """

    if probs.ndim != 2 or mask.ndim != 2:
        raise ValueError("probs and mask must be 2D tensors [B, T].")
    if probs.shape != mask.shape:
        raise ValueError(f"probs shape {probs.shape} != mask shape {mask.shape}.")

    method = normalize_session_aggregation_method(method)
    neg_inf = torch.finfo(probs.dtype).min
    safe = torch.where(mask.bool(), probs, neg_inf)

    if method == "max":
        scores = safe.max(dim=-1).values
        scores = scores.clamp(min=0.0, max=1.0)
        return scores

    if method == "topk_mean":
        k_eff = min(int(topk), int(probs.shape[-1]))
        topk_vals = torch.topk(safe, k=k_eff, dim=-1).values
        valid = topk_vals > neg_inf / 2
        denom = valid.float().sum(dim=-1).clamp_min(1.0)
        return (topk_vals * valid.float()).sum(dim=-1) / denom

    if method == "noisy_or":
        log_one_minus = torch.log((1.0 - probs).clamp_min(1e-7)) * mask.float()
        return (1.0 - torch.exp(log_one_minus.sum(dim=-1))).clamp(0.0, 1.0)

    raise ValueError(f"Unhandled aggregation method: {method}")
