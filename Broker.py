"""
Autonomous Broker
=================
Run once daily. Everything else is automatic.

    python Broker.py              # full cycle — data, decisions, shadows, maintenance
    python Broker.py --status     # show portfolio + shadow standings, no trading
    python Broker.py --trades     # show recent trade history

What happens on every run:
  1. Maintenance checks — updates prices, sentiment, model, and broker
     parameters automatically if they are stale. Nothing runs twice in a day.
  2. Live trading cycle — exits, screens, buys, logs vs SPY.
  3. Shadow portfolios — 5 paper strategies advance one cycle in parallel.
     After 30 days the best-performing one's parameters promote to live config.
     Options enable automatically once a shadow using them proves itself.

broker.config is managed by the system. The only values you need to set
manually are `cash` (your starting capital) and `max_positions`.
"""

import sys
import logging
from pathlib import Path
from datetime import date

for _stream in (sys.stdout, sys.stderr):
    _reconfigure = getattr(_stream, "reconfigure", None)
    if callable(_reconfigure):
        try:
            _reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

# ── Logging setup (before any imports that log) ───────────────────────────────
Path("logs").mkdir(exist_ok=True)
Path("plots").mkdir(exist_ok=True)
Path("broker/state").mkdir(parents=True, exist_ok=True)

import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/broker.log", encoding="utf-8"),
        logging.StreamHandler(stream=open(sys.stdout.fileno(), mode='w', encoding='utf-8', closefd=False)),
    ],
)
logger = logging.getLogger(__name__)

_LOCK_FILE  = Path("broker/state/.cycle_lock")
_STATE_FILE = Path("broker/state/.run_state.json")


def _write_run_state(stage: str, status: str = "running") -> None:
    """Atomic run-state record. Written at each stage; 'complete' only at end."""
    import os, json
    _STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _STATE_FILE.write_text(json.dumps({
        "date":   date.today().isoformat(),
        "pid":    os.getpid(),
        "stage":  stage,
        "status": status,
    }))


# ── Config loader ─────────────────────────────────────────────────────────────

def _load_config(path: str = "broker.config") -> dict:
    defaults = {}
    config_path = Path(path)
    if not config_path.exists():
        return defaults
    for line in config_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, val = line.partition("=")
            key = key.strip().replace("-", "_")
            val = val.split("#")[0].strip()
            if val.lower() == "true":
                defaults[key] = True
            elif val.lower() == "false":
                defaults[key] = False
            else:
                try:
                    defaults[key] = int(val)
                except ValueError:
                    try:
                        defaults[key] = float(val)
                    except ValueError:
                        defaults[key] = val
    return defaults


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Autonomous broker — run once daily.")
    p.add_argument("--status",          action="store_true", help="Show portfolio, no trading")
    p.add_argument("--trades",          action="store_true", help="Show recent trade history")
    p.add_argument("--no_shadows",      action="store_true", help="Skip shadow portfolio step")
    p.add_argument("--no_maintenance",  action="store_true", help="Skip staleness checks")
    p.add_argument("--approve-promotion", action="store_true",
                   help="Allow shadow genomes to auto-promote to live config this run")
    args = p.parse_args()

    # Always reload config fresh — the auto-tuner may have updated it
    config = _load_config()

    # ── Duplicate run prevention ──────────────────────────────────────────────
    if not (args.status or args.trades):
        today_str = date.today().isoformat()
        import json as _json
        prev = {}
        try:
            prev = _json.loads(_STATE_FILE.read_text()) if _STATE_FILE.exists() else {}
        except Exception:
            pass

        if prev.get("date") == today_str and prev.get("status") == "complete":
            logger.warning(
                "Broker already completed a full cycle today (%s). "
                "Delete broker/state/.run_state.json to force a re-run.", today_str
            )
            print(f"\n  Already ran today ({today_str}). Use --status to check portfolio.\n")
            sys.exit(0)
        elif prev.get("date") == today_str and prev.get("status") == "running":
            import os
            prev_pid = prev.get("pid", 0)
            try:
                os.kill(prev_pid, 0)   # check if PID is still alive
                logger.error(
                    "Another broker process (PID %d) appears to be running. "
                    "If it crashed, delete broker/state/.run_state.json.", prev_pid
                )
                sys.exit(1)
            except (OSError, ProcessLookupError):
                logger.warning("Previous run (PID %d) appears crashed at stage '%s' — resuming.",
                               prev_pid, prev.get("stage", "unknown"))

        _write_run_state("startup")

    # ── Status / trades — no trading ──────────────────────────────────────────
    if args.status or args.trades:
        from broker.broker import main as broker_main
        if args.status:
            sys.argv = [sys.argv[0], "--status"]
        else:
            sys.argv = [sys.argv[0], "--trades"]
        broker_main(config)

        # Show shadow standings alongside status
        if args.status:
            try:
                from broker.shadows import get_shadow_summary
                print(get_shadow_summary())
            except Exception:
                pass
        sys.exit(0)

    # ── Step 1: Maintenance — update stale data / model / params ─────────────
    maintenance_context = None
    if not args.no_maintenance:
        _write_run_state("maintenance")
        try:
            from pipeline.maintenance import run_maintenance
            maintenance_context = run_maintenance(
                initial_cash=float(config.get("cash", 10_000))
            )
            # Reload config — maintenance may have updated it
            config = _load_config()
        except Exception as exc:
            logger.warning("Maintenance step failed (continuing): %s", exc)

    # ── Step 2: Live trading cycle ────────────────────────────────────────────
    _write_run_state("trading")
    try:
        from broker.broker import main as broker_main
        broker_main(config, maintenance_context=maintenance_context)
    except FileNotFoundError as exc:
        if "stooq_panel.parquet" in str(exc) or "Price data not found" in str(exc):
            print("\n" + "="*60)
            print("  FIRST-TIME SETUP REQUIRED")
            print("="*60)
            print("  Price data not found. Train the model first:\n")
            print("    python Agent.py --mode train --folds 10\n")
            print("  This downloads historical data and trains the RL model.")
            print("  Takes 2-6 hours on CPU, 30-60 min on GPU.")
            print("  After that, just run: python Broker.py")
            print("="*60 + "\n")
        else:
            raise

    # ── Step 3: Shadow portfolios ─────────────────────────────────────────────
    if not args.no_shadows:
        _write_run_state("shadows")
        try:
            from pipeline.data import load_master
            from broker.replay import _build_price_lookup
            from broker.shadows import run_shadow_cycle, get_shadow_summary

            logger.info("Loading data for shadow portfolios...")
            df_features  = load_master(top_n=int(config.get("top_n", 500)))
            price_lookup = _build_price_lookup()

            run_shadow_cycle(
                df_features     = df_features,
                price_lookup    = price_lookup,
                live_config     = config,
                checkpoint_path = config.get("rl_checkpoint_path"),
                allow_promotion = getattr(args, "approve_promotion", False),
            )

            # Reload config — shadows may have promoted new parameters
            config = _load_config()
            print(get_shadow_summary())

        except Exception as exc:
            logger.warning("Shadow portfolio step failed (continuing): %s", exc)

    # ── Mark cycle complete ───────────────────────────────────────────────────
    _write_run_state("complete", status="complete")
