"""
Policy-selection matrix runner.
Runs weak_sleeve then low_price family matrices, prints decisions,
and writes the winning config back to broker.config if promoted.

Usage:
    python run_policy_matrix.py
"""

import sys
import logging
import json
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Load config and data ──────────────────────────────────────────────────────

def _load_typed_config(path: str = "broker.config") -> dict:
    cfg = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            v = v.split("#")[0].strip()
            cfg[k.strip()] = (
                True if v.lower() == "true" else
                False if v.lower() == "false" else
                (int(v) if v.lstrip("-").isdigit() else
                 (float(v) if v.replace(".", "", 1).lstrip("-").isdigit() else v))
            )
    return cfg


def _write_config_key(key: str, value: str, path: str = "broker.config") -> None:
    import re
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    key_pattern = re.compile(rf"^(\s*{re.escape(key)}\s*=\s*)([^#\r\n]*)(.*)$")
    new_lines = []
    replaced = False
    for line in lines:
        m = key_pattern.match(line)
        if m and not replaced:
            prefix, old_val, suffix = m.groups()
            spacing = old_val[len(old_val.rstrip()):]
            new_lines.append(f"{prefix}{value}{spacing}{suffix}\n")
            replaced = True
        else:
            new_lines.append(line)
    if not replaced:
        new_lines.append(f"{key:<30} = {value}\n")
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)
    logger.info("broker.config updated: %s = %s", key, value)


def main():
    import torch
    from pipeline.data import load_master
    from broker.replay import _build_price_lookup, run_policy_family_matrix
    from pipeline.checkpoints import resolve_checkpoint_path

    live_config = _load_typed_config()
    top_n = int(live_config.get("top_n", 500))
    checkpoint_path = resolve_checkpoint_path(
        checkpoint_path=live_config.get("rl_checkpoint_path", "auto"),
        save_dir="models",
    )
    strategy = "screener_rl" if live_config.get("rl_enabled") and checkpoint_path else "heuristics_only"

    logger.info("Loading market data (top_n=%d)...", top_n)
    df_features = load_master(top_n=top_n, config=live_config)
    price_lookup = _build_price_lookup()

    logger.info("Checkpoint: %s", checkpoint_path)
    logger.info("Strategy: %s", strategy)

    # ── Step 1: Weak-sleeve family matrix ─────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 1: Running weak_sleeve policy family matrix...")
    logger.info("=" * 60)

    weak_result = run_policy_family_matrix(
        df_features=df_features,
        price_lookup=price_lookup,
        family="weak_sleeve",
        n_windows=5,
        window_years=1,
        step_months=3,
        output_root="experiments",
        live_config=live_config,
        strategy=strategy,
        checkpoint_path=checkpoint_path,
    )

    weak_summary = weak_result.get("summary_table")
    weak_decision = None
    if weak_summary is not None and not weak_summary.empty:
        print("\n" + "=" * 72)
        print("  WEAK-SLEEVE FAMILY — Summary Table")
        print("=" * 72)
        display_cols = [c for c in [
            "label", "family", "wins", "winner_rate", "avg_policy_score",
            "avg_total_return", "avg_sharpe", "avg_max_drawdown",
            "decision_status", "decision_reason",
        ] if c in weak_summary.columns]
        print(weak_summary[display_cols].to_string(index=False))
        print("=" * 72)

        # Find the promoted variant (if any)
        promoted = weak_summary[weak_summary.get("decision_status", "") == "promote"] if "decision_status" in weak_summary.columns else weak_summary.iloc[0:0]
        if not promoted.empty:
            winner_label = str(promoted.iloc[0]["label"])
            weak_decision = "promote"
            logger.info("WEAK-SLEEVE DECISION: PROMOTE → %s", winner_label)

            # Apply the winning weak-sleeve params to live_config
            from broker.replay import POLICY_FAMILY_VARIANTS
            variants_map = dict(POLICY_FAMILY_VARIANTS.get("weak_sleeve", []))
            if winner_label in variants_map:
                for k, v in variants_map[winner_label].items():
                    live_config[k] = v
                    _write_config_key(k, str(v))
                logger.info("Weak-sleeve winner params applied to broker.config")
        else:
            weak_decision = "hold_for_more_evidence"
            logger.info("WEAK-SLEEVE DECISION: HOLD — no variant cleared promotion gates")
    else:
        logger.warning("Weak-sleeve matrix produced no summary table.")

    # ── Step 2: Low-price family matrix (with weak-sleeve winner frozen) ──────
    logger.info("=" * 60)
    logger.info("STEP 2: Running low_price policy family matrix...")
    logger.info("(Using weak-sleeve winner config as baseline)")
    logger.info("=" * 60)

    low_result = run_policy_family_matrix(
        df_features=df_features,
        price_lookup=price_lookup,
        family="low_price",
        n_windows=5,
        window_years=1,
        step_months=3,
        output_root="experiments",
        live_config=live_config,   # frozen with weak-sleeve winner
        strategy=strategy,
        checkpoint_path=checkpoint_path,
    )

    low_summary = low_result.get("summary_table")
    low_decision = None
    if low_summary is not None and not low_summary.empty:
        print("\n" + "=" * 72)
        print("  LOW-PRICE FAMILY — Summary Table")
        print("=" * 72)
        display_cols = [c for c in [
            "label", "family", "wins", "winner_rate", "avg_policy_score",
            "avg_total_return", "avg_sharpe", "avg_max_drawdown",
            "decision_status", "decision_reason",
        ] if c in low_summary.columns]
        print(low_summary[display_cols].to_string(index=False))
        print("=" * 72)

        promoted = low_summary[low_summary.get("decision_status", "") == "promote"] if "decision_status" in low_summary.columns else low_summary.iloc[0:0]
        if not promoted.empty:
            winner_label = str(promoted.iloc[0]["label"])
            low_decision = "promote"
            logger.info("LOW-PRICE DECISION: PROMOTE → %s", winner_label)

            from broker.replay import POLICY_FAMILY_VARIANTS
            variants_map = dict(POLICY_FAMILY_VARIANTS.get("low_price", []))
            if winner_label in variants_map:
                for k, v in variants_map[winner_label].items():
                    live_config[k] = v
                    _write_config_key(k, str(v))
                logger.info("Low-price winner params applied to broker.config")
        else:
            low_decision = "hold_for_more_evidence"
            logger.info("LOW-PRICE DECISION: HOLD — no variant cleared promotion gates")
    else:
        logger.warning("Low-price matrix produced no summary table.")

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  POLICY MATRIX COMPLETE")
    print("=" * 72)
    print(f"  Weak-sleeve: {weak_decision or 'no result'}")
    print(f"  Low-price:   {low_decision or 'no result'}")
    print(f"  Artifacts:   experiments/")
    print(f"  Config:      broker.config (updated if any promotions)")
    print("=" * 72)
    print()
    print("  Next steps (not yet implemented — require new data sources):")
    print("  - earnings-reaction policy (needs EPS beat/miss data)")
    print("  - macro regime policy (needs yield curve / credit spread data)")
    print("  - insider soft-adjustment (needs Form 4 / SEC filing data)")
    print("=" * 72)


if __name__ == "__main__":
    main()
