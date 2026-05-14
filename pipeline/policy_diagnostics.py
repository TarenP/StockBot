import math
from typing import Iterable

import numpy as np
import torch

from pipeline.action_projection import project_actions


def _as_numpy(values: Iterable[float] | np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)


def weight_concentration_metrics(
    weights: Iterable[float] | np.ndarray,
    universe_size: int | None = None,
    cash_weight: float | None = None,
    eps: float = 1e-12,
) -> dict:
    asset_weights = _as_numpy(weights)
    if cash_weight is None and len(asset_weights) and universe_size is not None and len(asset_weights) == universe_size + 1:
        cash_weight = float(asset_weights[-1])
        asset_weights = asset_weights[:-1]
    elif cash_weight is None:
        cash_weight = 0.0

    if universe_size is None:
        universe_size = int(len(asset_weights))

    nonzero = asset_weights[asset_weights > eps]
    sorted_weights = np.sort(asset_weights)[::-1]
    asset_sum = float(asset_weights.sum())
    normalized = asset_weights / asset_sum if asset_sum > eps else np.zeros_like(asset_weights)
    entropy = float(-(normalized[normalized > eps] * np.log(normalized[normalized > eps])).sum())
    effective = float(math.exp(entropy)) if entropy > 0 else 0.0
    equal_weight_baseline = (1.0 / universe_size) if universe_size else 0.0
    top10_equal = min(10, universe_size) * equal_weight_baseline if universe_size else 0.0
    top10_sum = float(sorted_weights[:10].sum())
    top10_ratio = (top10_sum / top10_equal) if top10_equal > eps else 0.0

    return {
        "universe_size": int(universe_size),
        "nonzero_positions": int(len(nonzero)),
        "sum_weights": float(asset_weights.sum() + cash_weight),
        "max_weight": float(asset_weights.max()) if len(asset_weights) else 0.0,
        "min_nonzero_weight": float(nonzero.min()) if len(nonzero) else 0.0,
        "top_10_weight_sum": top10_sum,
        "top_20_weight_sum": float(sorted_weights[:20].sum()),
        "top_50_weight_sum": float(sorted_weights[:50].sum()),
        "cash_weight": float(cash_weight),
        "weight_std": float(asset_weights.std()) if len(asset_weights) else 0.0,
        "weight_entropy": entropy,
        "effective_number_of_positions": effective,
        "equal_weight_baseline": float(equal_weight_baseline),
        "top10_vs_equal_weight_ratio": float(top10_ratio),
    }


def is_near_uniform(metrics: dict) -> bool:
    universe_size = int(metrics.get("universe_size", 0))
    return (
        metrics.get("top10_vs_equal_weight_ratio", 0.0) < 2.0
        or metrics.get("effective_number_of_positions", 0.0) > 0.75 * universe_size
    )


def format_weight_diagnostics(metrics: dict, title: str = "RL Weight Diagnostics") -> list[str]:
    lines = [f"{title}:"]
    for key in ("projection_mode", "temperature", "top_k"):
        if key in metrics:
            value = metrics[key]
            if isinstance(value, str):
                lines.append(f"  {key}: {value}")
            else:
                lines.append(f"  {key}: {value:.6f}" if isinstance(value, float) else f"  {key}: {value}")
    for key in (
        "universe_size",
        "nonzero_positions",
        "sum_weights",
        "max_weight",
        "min_nonzero_weight",
        "top_10_weight_sum",
        "top_20_weight_sum",
        "top_50_weight_sum",
        "cash_weight",
        "weight_std",
        "weight_entropy",
        "effective_number_of_positions",
        "equal_weight_baseline",
        "top10_vs_equal_weight_ratio",
    ):
        value = metrics.get(key, 0.0)
        if isinstance(value, int):
            lines.append(f"  {key}: {value}")
        else:
            lines.append(f"  {key}: {value:.6f}")
    if is_near_uniform(metrics):
        lines.append("  WARNING: RL weights are near-uniform; policy may not be expressing conviction.")
    return lines


def raw_policy_diagnostics(
    logits: torch.Tensor,
    projection: str = "softplus",
    temperature: float = 1.0,
    top_k: int = 50,
) -> dict:
    logits_np = logits.detach().float().cpu().numpy().reshape(-1)
    weights = project_actions(
        logits.detach().float(),
        projection=projection,
        temperature=temperature,
        top_k=top_k,
    )
    weights_np = weights.cpu().numpy().reshape(-1)
    metrics = weight_concentration_metrics(weights_np[:-1], cash_weight=float(weights_np[-1]))
    metrics.update({
        "raw_action_min": float(logits_np.min()) if len(logits_np) else 0.0,
        "raw_action_max": float(logits_np.max()) if len(logits_np) else 0.0,
        "raw_action_mean": float(logits_np.mean()) if len(logits_np) else 0.0,
        "raw_action_std": float(logits_np.std()) if len(logits_np) else 0.0,
        "raw_action_entropy": metrics["weight_entropy"],
        "raw_action_top10_sum_after_projection": metrics["top_10_weight_sum"],
        "raw_action_top10_sum_after_softplus_projection": metrics["top_10_weight_sum"],
    })
    return metrics


def action_transform_trace(
    logits: torch.Tensor,
    final_weights: Iterable[float] | np.ndarray,
    projection: str = "softplus",
    temperature: float = 1.0,
    top_k: int = 50,
) -> list[dict]:
    logits_np = logits.detach().float().cpu().numpy().reshape(-1)
    raw_softmax = torch.softmax(logits.detach().float().reshape(1, -1), dim=-1)
    raw_softmax_np = raw_softmax.cpu().numpy().reshape(-1)
    projected_np = project_actions(
        logits.detach().float(),
        projection=projection,
        temperature=temperature,
        top_k=top_k,
    ).cpu().numpy().reshape(-1)
    final_np = _as_numpy(final_weights)
    raw_shifted = logits_np - logits_np.min()
    raw_weights = raw_shifted / raw_shifted.sum() if raw_shifted.sum() > 1e-12 else np.ones_like(raw_shifted) / max(len(raw_shifted), 1)
    selected = np.zeros_like(logits_np, dtype=np.float64)
    n_assets = max(len(logits_np) - 1, 0)
    if projection in {"top_k_softmax", "rank_top_k"} and n_assets:
        k = min(int(top_k), n_assets)
        top_idx = np.argsort(logits_np[:-1])[::-1][:k]
        selected[top_idx] = 1.0 / max(k, 1)
    elif n_assets:
        selected[:-1] = 1.0 / n_assets
    stages = [
        ("raw_policy", raw_softmax_np),
        ("shifted_or_centered_policy", raw_weights),
        ("projection_selected_universe", selected),
        ("projected_weights", projected_np),
        ("final_weights", final_np),
    ]
    trace = []
    for stage, values in stages:
        metrics = weight_concentration_metrics(values[:-1], cash_weight=float(values[-1]) if len(values) else 0.0)
        metrics["stage"] = stage
        trace.append(metrics)
    return trace


def average_metric_dicts(metrics: list[dict], prefix: str = "avg_") -> dict:
    if not metrics:
        return {}
    numeric_keys = [
        key for key, value in metrics[0].items()
        if isinstance(value, (int, float, np.integer, np.floating))
    ]
    return {
        f"{prefix}{key}": float(np.mean([m.get(key, 0.0) for m in metrics]))
        for key in numeric_keys
    }
