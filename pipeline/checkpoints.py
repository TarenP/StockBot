"""
Shared checkpoint resolution helpers.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def resolve_checkpoint_path(
    checkpoint_path: str | None = None,
    save_dir: str = "models",
) -> str | None:
    """
    Resolve ``auto`` (or None) to the best checkpoint in ``save_dir`` by
    validation Sharpe. Returns an explicit path unchanged.
    """
    if checkpoint_path and str(checkpoint_path).strip().lower() != "auto":
        return checkpoint_path

    ckpts = sorted(Path(save_dir).glob("best_fold*.pt"))
    if not ckpts:
        return None

    best_path: str | None = None
    best_sharpe = float("-inf")
    for path in ckpts:
        try:
            meta = torch.load(str(path), map_location="cpu", weights_only=False)
            sharpe = float(meta.get("val_sharpe", float("-inf")))
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_path = str(path)
        except Exception as exc:
            logger.debug("Could not inspect checkpoint %s: %s", path, exc)

    return best_path or str(ckpts[-1])


def load_checkpoint_meta(
    checkpoint_path: str | None = None,
    save_dir: str = "models",
) -> tuple[str | None, dict | None]:
    path = resolve_checkpoint_path(checkpoint_path=checkpoint_path, save_dir=save_dir)
    if path is None:
        return None, None

    try:
        meta = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        logger.warning("Could not load checkpoint metadata from %s: %s", path, exc)
        return path, None
    return path, meta


def load_checkpoint_asset_list(
    checkpoint_path: str | None = None,
    save_dir: str = "models",
) -> list[str] | None:
    _, meta = load_checkpoint_meta(checkpoint_path=checkpoint_path, save_dir=save_dir)
    if not meta:
        return None
    asset_list = meta.get("asset_list")
    if not asset_list:
        return None
    cleaned: list[str] = []
    seen: set[str] = set()
    for ticker in asset_list:
        symbol = str(ticker).strip().upper().replace(".", "-")
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        cleaned.append(symbol)
    return cleaned or None
