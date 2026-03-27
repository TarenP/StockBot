"""
Entry point for the autonomous broker.

Settings are loaded from broker.config (edit that file to set your defaults).
Any flag passed on the command line overrides the config file.

Usage:
    python Broker.py              # run a cycle with config defaults
    python Broker.py --status     # show portfolio + SPY benchmark
    python Broker.py --trades     # show recent trade history
"""

from pathlib import Path


def _load_config(path: str = "broker.config") -> dict:
    """Parse broker.config into a dict of defaults."""
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
            val = val.strip()
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


if __name__ == "__main__":
    import sys
    from broker.broker import main
    config = _load_config()
    main(config)
