"""Run the strict feature-ablation audit.

Default output:
    experiments/feature_ablation/<run_id>/
"""

from __future__ import annotations

import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _load_typed_config(path: str = "broker.config") -> dict:
    cfg = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            raw = value.split("#")[0].strip()
            if raw.lower() == "true":
                parsed = True
            elif raw.lower() == "false":
                parsed = False
            else:
                try:
                    parsed = int(raw)
                except ValueError:
                    try:
                        parsed = float(raw)
                    except ValueError:
                        parsed = raw
            cfg[key.strip()] = parsed
    return cfg


def parse_args():
    parser = argparse.ArgumentParser(description="Run StockBot feature ablation audit.")
    parser.add_argument("--n-windows", type=int, default=5)
    parser.add_argument("--window-years", type=int, default=1)
    parser.add_argument("--step-months", type=int, default=3)
    parser.add_argument("--initial-cash", type=float, default=10_000.0)
    parser.add_argument("--top-n", type=int, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--output-root", type=str, default="experiments/feature_ablation")
    parser.add_argument("--dry-run", action="store_true", help="Write audit layout without running replays")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from broker.feature_ablation import run_feature_ablation_audit
    from broker.replay import _build_price_lookup
    from pipeline.checkpoints import resolve_checkpoint_path
    from pipeline.data import load_master

    config = _load_typed_config()
    top_n = int(args.top_n or config.get("top_n", 500))
    checkpoint_path = resolve_checkpoint_path(
        checkpoint_path=config.get("rl_checkpoint_path", "auto"),
        save_dir="models",
    )
    strategy = "screener_rl" if config.get("rl_enabled") and checkpoint_path else "heuristics_only"

    logger.info("Loading feature panel top_n=%d", top_n)
    df_features = load_master(top_n=top_n, config=config)
    price_lookup = _build_price_lookup()
    result = run_feature_ablation_audit(
        df_features,
        price_lookup,
        n_windows=args.n_windows,
        window_years=args.window_years,
        step_months=args.step_months,
        initial_cash=args.initial_cash,
        live_config=config,
        strategy=strategy,
        checkpoint_path=checkpoint_path,
        output_root=args.output_root,
        run_id=args.run_id,
        dry_run=args.dry_run,
    )
    logger.info("Feature ablation artifacts written -> %s", result["output_dir"])


if __name__ == "__main__":
    main()

