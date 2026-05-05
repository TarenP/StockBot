"""
Autonomous Broker
=================
Run this whenever you want the broker to do the right thing.

    python Broker.py              # smart default: full cycle once/day, status after
    python Broker.py --status     # refresh holding prices/marks/status, no trading
    python Broker.py --force      # rerun today's full cycle and overwrite current outputs
    python Broker.py --snapshot   # same as --status
    python Broker.py --trades     # show recent trade history

The default command is daily-idempotent:
  1. If today's full broker cycle has not run, run due maintenance and trading.
  2. If today's full broker cycle already ran, refresh marks and status only.
  3. Periodic heavy work runs only when explicitly configured or forced.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    _reconfigure = getattr(_stream, "reconfigure", None)
    if callable(_reconfigure):
        try:
            _reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

Path("logs").mkdir(exist_ok=True)
Path("plots").mkdir(exist_ok=True)
Path("broker/state").mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/broker.log", encoding="utf-8"),
        logging.StreamHandler(
            stream=open(sys.stdout.fileno(), mode="w", encoding="utf-8", closefd=False)
        ),
    ],
)
logger = logging.getLogger(__name__)


def _load_config(path: str = "broker.config") -> dict:
    defaults = {}
    config_path = Path(path)
    if not config_path.exists():
        return defaults
    for line in config_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
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


def _parse_args():
    import argparse

    p = argparse.ArgumentParser(description="Autonomous broker smart orchestrator.")
    p.add_argument(
        "--status",
        action="store_true",
        help="Fetch holding prices and update portfolio status, no trading",
    )
    p.add_argument("--snapshot", action="store_true", help="Alias for --status")
    p.add_argument(
        "--refresh-prices",
        action="store_true",
        help="Refresh cached broad-universe prices before status/snapshot",
    )
    p.add_argument("--trades", action="store_true", help="Show recent trade history")
    p.add_argument(
        "--force",
        action="store_true",
        help="Rerun today's full broker cycle and overwrite current-day canonical outputs",
    )
    p.add_argument(
        "--no-periodic",
        dest="no_periodic",
        action="store_true",
        help="Skip post-cycle periodic tasks",
    )
    p.add_argument(
        "--only-periodic",
        dest="only_periodic",
        action="store_true",
        help="Run due periodic tasks without trading",
    )
    p.add_argument(
        "--force-periodic",
        dest="force_periodic",
        action="store_true",
        help="Bypass due checks for registered periodic tasks",
    )
    p.add_argument(
        "--refresh-ai-sidecar",
        dest="refresh_ai_sidecar",
        action="store_true",
        help="Precompute cached local-LLM document features before the broker cycle",
    )
    p.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="Print the orchestrator decision without running tasks",
    )
    p.add_argument("--no_shadows", action="store_true", help="Skip shadow portfolio step")
    p.add_argument("--no_maintenance", action="store_true", help="Skip staleness checks")
    p.add_argument(
        "--approve-promotion",
        action="store_true",
        help="Allow shadow genomes to auto-promote to live config this run",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    config = _load_config()
    from broker.orchestrator import run_smart_broker_command

    run_smart_broker_command(args, config)
