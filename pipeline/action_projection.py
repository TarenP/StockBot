from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


DEFAULT_ACTION_PROJECTION = "softplus"
DEFAULT_ACTION_TEMPERATURE = 1.0
DEFAULT_ACTION_TOP_K = 50
SUPPORTED_ACTION_PROJECTIONS = {"softplus", "softmax", "top_k_softmax", "rank_top_k"}


def normalize_projection_settings(
    projection: str | None = None,
    temperature: float | int | str | None = None,
    top_k: int | str | None = None,
    rl_action_projection: str | None = None,
    rl_action_temperature: float | int | str | None = None,
    rl_action_top_k: int | str | None = None,
) -> dict[str, Any]:
    if projection is None:
        projection = rl_action_projection
    if temperature is None:
        temperature = rl_action_temperature
    if top_k is None:
        top_k = rl_action_top_k

    projection = (projection or DEFAULT_ACTION_PROJECTION).strip().lower()
    if projection not in SUPPORTED_ACTION_PROJECTIONS:
        raise ValueError(
            f"Unknown RL action projection '{projection}'. "
            f"Expected one of {sorted(SUPPORTED_ACTION_PROJECTIONS)}."
        )

    try:
        temperature = DEFAULT_ACTION_TEMPERATURE if temperature is None else float(temperature)
    except (TypeError, ValueError) as exc:
        raise ValueError("rl_action_temperature must be a positive float.") from exc
    if temperature <= 0.0:
        raise ValueError("rl_action_temperature must be > 0.")

    try:
        top_k = DEFAULT_ACTION_TOP_K if top_k is None else int(top_k)
    except (TypeError, ValueError) as exc:
        raise ValueError("rl_action_top_k must be a positive integer.") from exc
    if top_k <= 0:
        raise ValueError("rl_action_top_k must be > 0.")

    return {
        "rl_action_projection": projection,
        "rl_action_temperature": temperature,
        "rl_action_top_k": top_k,
    }


def projection_kwargs(settings: dict[str, Any] | None = None) -> dict[str, Any]:
    settings = normalize_projection_settings(**(settings or {}))
    return {
        "projection": settings["rl_action_projection"],
        "temperature": settings["rl_action_temperature"],
        "top_k": settings["rl_action_top_k"],
    }


def _as_2d(logits: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if logits.ndim == 1:
        return logits.unsqueeze(0), True
    if logits.ndim != 2:
        raise ValueError("Action projection expects logits with shape (n,) or (batch, n).")
    return logits, False


def _sanitize_logits(logits: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(logits.float(), nan=0.0, posinf=1e6, neginf=-1e6)


def _restore_shape(weights: torch.Tensor, squeezed: bool) -> torch.Tensor:
    return weights.squeeze(0) if squeezed else weights


def project_actions(
    logits: torch.Tensor,
    projection: str = DEFAULT_ACTION_PROJECTION,
    temperature: float = DEFAULT_ACTION_TEMPERATURE,
    top_k: int = DEFAULT_ACTION_TOP_K,
    eps: float = 1e-6,
) -> torch.Tensor:
    settings = normalize_projection_settings(projection, temperature, top_k)
    projection = settings["rl_action_projection"]
    temperature = settings["rl_action_temperature"]
    top_k = settings["rl_action_top_k"]

    logits_2d, squeezed = _as_2d(_sanitize_logits(logits))
    if logits_2d.shape[-1] == 0:
        raise ValueError("Action projection received an empty action vector.")

    if projection == "softplus":
        concentration = F.softplus(logits_2d / temperature) + eps
        concentration[:, -1] = torch.clamp(concentration[:, -1], min=eps)
        weights = concentration / concentration.sum(dim=-1, keepdim=True).clamp_min(eps)
        return _restore_shape(weights, squeezed)

    if projection == "softmax":
        weights = torch.softmax(logits_2d / temperature, dim=-1)
        return _restore_shape(weights, squeezed)

    n_assets = max(int(logits_2d.shape[-1]) - 1, 0)
    if n_assets == 0:
        weights = torch.ones_like(logits_2d)
        return _restore_shape(weights, squeezed)

    k = min(top_k, n_assets)
    asset_logits = logits_2d[:, :-1]
    top_idx = torch.topk(asset_logits, k=k, dim=-1).indices

    if projection == "top_k_softmax":
        masked_assets = torch.full_like(asset_logits, -torch.inf)
        selected = asset_logits.gather(1, top_idx) / temperature
        masked_assets.scatter_(1, top_idx, selected)
        cash = logits_2d[:, -1:] / temperature
        weights = torch.softmax(torch.cat([masked_assets, cash], dim=-1), dim=-1)
        return _restore_shape(weights, squeezed)

    if projection == "rank_top_k":
        asset_weights = torch.zeros_like(asset_logits)
        decay = (1.0 / torch.arange(1, k + 1, device=logits_2d.device, dtype=logits_2d.dtype))
        decay = decay.unsqueeze(0).expand(logits_2d.shape[0], -1)
        asset_weights.scatter_(1, top_idx, decay)
        weights = torch.cat([asset_weights, torch.zeros_like(logits_2d[:, -1:])], dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(eps)
        return _restore_shape(weights, squeezed)

    raise ValueError(f"Unknown RL action projection '{projection}'.")
