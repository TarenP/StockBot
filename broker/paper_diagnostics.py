"""
Paper-trading diagnostics.

These helpers persist the parts of the paper broker that should be auditable
over time: cap pressure, execution drag, replay/live parity, and P&L
attribution. They avoid broker APIs entirely.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from broker.exposure import low_price_bucket, theme_bucket

STATE_DIR = Path("broker/state")
PORTFOLIO_HISTORY_PATH = STATE_DIR / "portfolio_history.jsonl"
CAP_IMPACT_SUMMARY_PATH = STATE_DIR / "cap_impact_summary.json"
PERFORMANCE_ATTRIBUTION_PATH = STATE_DIR / "performance_attribution.json"
PARITY_REPORT_PATH = STATE_DIR / "replay_live_parity.json"

CAP_IMPACT_KEYS = [
    "sentiment_cap_impact",
    "volatility_cap_impact",
    "sector_cap_impact",
    "theme_cap_impact",
    "correlation_cap_impact",
    "low_price_cap_impact",
    "cash_or_risk_cap_impact",
    "total_cap_impact",
]

CAP_REASON_TO_IMPACT_KEY = {
    "sentiment_cap": "sentiment_cap_impact",
    "volatility_cap": "volatility_cap_impact",
    "sector_cap": "sector_cap_impact",
    "theme_cap": "theme_cap_impact",
    "correlation_cap": "correlation_cap_impact",
    "low_price_or_penny_cap": "low_price_cap_impact",
    "penny_cap": "low_price_cap_impact",
    "low_price_cap": "low_price_cap_impact",
    "cash_or_risk_cap": "cash_or_risk_cap_impact",
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    if not np.isfinite(out):
        return default
    return out


def _parse_reason_field(reason: str, field: str) -> str | None:
    match = re.search(rf"(?:^|\|\s*){re.escape(field)}=([^|]+)", str(reason or ""))
    if not match:
        return None
    value = match.group(1).strip()
    return value or None


def _parse_downweight_reason(reason: str) -> str:
    value = _parse_reason_field(reason, "downweight_reason")
    if not value:
        return "none"
    return value.split()[0].strip()


def _parse_entry_theme(ticker: str, reason: str) -> str:
    value = _parse_reason_field(reason, "Theme")
    if value:
        return value.split()[0].strip()
    return theme_bucket(ticker)


def _classify_exit_reason(reason: str) -> str:
    text = str(reason or "").strip().lower()
    if not text:
        return "unknown"
    if "stop-loss" in text or "stop loss" in text:
        return "stop_loss"
    if "trailing stop" in text:
        return "trailing_stop"
    if "dead-money" in text or "dead money" in text:
        return "dead_money"
    if "rl exit" in text or "rl conviction drop" in text:
        return "rl_phase2"
    if "partial take-profit" in text or "take-profit" in text or "take profit" in text:
        return "take_profit"
    if "signal deteriorated" in text or "weak signal" in text:
        return "signal_exit"
    return "other"


def _entry_cap_summary_from_trade_log(trade_log: list[dict]) -> dict:
    counts = {key: 0 for key in CAP_IMPACT_KEYS}
    by_reason: dict[str, int] = {}
    by_ticker: dict[str, dict[str, Any]] = {}

    for rec in trade_log or []:
        if str(rec.get("action", "")).upper() != "BUY":
            continue
        reason = _parse_downweight_reason(str(rec.get("reason", "") or ""))
        if reason in {"", "none", "nan"}:
            continue
        impact_key = CAP_REASON_TO_IMPACT_KEY.get(reason)
        if impact_key:
            counts[impact_key] += 1
        by_reason[reason] = by_reason.get(reason, 0) + 1
        ticker = str(rec.get("ticker", "")).upper()
        if ticker:
            slot = by_ticker.setdefault(
                ticker,
                {"ticker": ticker, "count": 0, "cap_reasons": {}},
            )
            slot["count"] += 1
            slot["cap_reasons"][reason] = slot["cap_reasons"].get(reason, 0) + 1

    counts["total_cap_impact"] = sum(
        count for key, count in counts.items() if key != "total_cap_impact"
    )
    return {
        "buy_trades_with_caps": sum(by_reason.values()),
        "intervention_counts": counts,
        "by_reason": dict(sorted(by_reason.items(), key=lambda item: item[1], reverse=True)),
        "by_ticker": sorted(
            by_ticker.values(),
            key=lambda item: item["count"],
            reverse=True,
        )[:10],
    }


def write_json(path: Path | str, payload: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


def load_jsonl(path: Path | str) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open() as fh:
        for line in fh:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def summarize_cap_impact_history(
    history_path: Path | str = PORTFOLIO_HISTORY_PATH,
    lookback: int | None = None,
    trade_log: list[dict] | None = None,
) -> dict:
    rows = load_jsonl(history_path)
    if lookback is not None and lookback > 0:
        rows = rows[-int(lookback):]

    totals = {key: 0.0 for key in CAP_IMPACT_KEYS}
    counts = {key: 0 for key in CAP_IMPACT_KEYS}
    intervention_totals: dict[tuple[str, str], dict[str, Any]] = {}
    cycles_with_caps = 0

    for row in rows:
        summary = row.get("allocation_summary") or {}
        cap_impact = summary.get("cap_impact") or {}
        cap_counts = summary.get("cap_intervention_counts") or {}
        cycle_total = 0.0
        for key in CAP_IMPACT_KEYS:
            impact = _safe_float(cap_impact.get(key))
            totals[key] += impact
            counts[key] += int(cap_counts.get(key, 0) or 0)
            if key != "total_cap_impact":
                cycle_total += impact
        if cycle_total > 0:
            cycles_with_caps += 1

        for item in summary.get("top_cap_interventions") or []:
            ticker = str(item.get("ticker", "")).upper()
            cap_class = str(item.get("cap_class", "unknown"))
            if not ticker:
                continue
            key = (ticker, cap_class)
            slot = intervention_totals.setdefault(
                key,
                {"ticker": ticker, "cap_class": cap_class, "count": 0, "total_impact": 0.0},
            )
            slot["count"] += 1
            slot["total_impact"] += _safe_float(item.get("impact"))

    n_cycles = len(rows)
    average = {
        key: (totals[key] / n_cycles if n_cycles else 0.0)
        for key in CAP_IMPACT_KEYS
    }
    top_interventions = sorted(
        intervention_totals.values(),
        key=lambda item: item["total_impact"],
        reverse=True,
    )[:10]

    entry_caps = _entry_cap_summary_from_trade_log(trade_log or [])

    return {
        "generated_at": datetime.now().isoformat(),
        "cycles": n_cycles,
        "cycles_with_cap_interventions": cycles_with_caps,
        "scope": {
            "cycle_caps": "allocation_summary rows in portfolio_history.jsonl",
            "entry_caps": "BUY trade reason downweight_reason fields",
        },
        "total_impact": totals,
        "average_impact_per_cycle": average,
        "intervention_counts": counts,
        "top_interventions": top_interventions,
        "entry_cap_interventions": entry_caps,
    }


def build_trade_attribution(trade_log: list[dict]) -> list[dict]:
    if not trade_log:
        return []

    open_lots: dict[str, list[dict]] = {}
    closed: list[dict] = []

    def _trade_time(rec: dict) -> datetime:
        raw = rec.get("fill_date") or rec.get("date") or rec.get("time")
        try:
            return datetime.fromisoformat(str(raw).replace("Z", "+00:00")).replace(tzinfo=None)
        except Exception:
            return datetime.min

    for rec in sorted(trade_log, key=_trade_time):
        action = str(rec.get("action", "")).upper()
        ticker = str(rec.get("ticker", "")).upper()
        shares = _safe_float(rec.get("shares"))
        price = _safe_float(rec.get("price"))
        execution_cost = _safe_float(rec.get("execution_cost"))
        if not ticker or shares <= 0 or price <= 0:
            continue

        fill_time = _trade_time(rec)
        reason = str(rec.get("reason", "") or "")
        if action == "BUY":
            basis = shares * price + execution_cost
            open_lots.setdefault(ticker, []).append(
                {
                    "shares": shares,
                    "entry_time": fill_time,
                    "entry_price": price,
                    "entry_cost_basis": basis,
                    "entry_execution_cost": execution_cost,
                    "entry_reason": reason,
                }
            )
            continue

        if action not in {"SELL", "SELL_PARTIAL"}:
            continue

        lots = open_lots.get(ticker, [])
        if not lots:
            continue

        remaining = shares
        while remaining > 1e-9 and lots:
            lot = lots[0]
            used = min(remaining, _safe_float(lot["shares"]))
            exit_share_ratio = used / shares if shares > 0 else 0.0
            lot_share_ratio = used / lot["shares"] if lot["shares"] > 0 else 0.0
            exit_cost = execution_cost * exit_share_ratio
            entry_cost = lot["entry_execution_cost"] * lot_share_ratio
            entry_unit_basis = lot["entry_cost_basis"] / lot["shares"]
            entry_basis = entry_unit_basis * used
            gross_pnl = used * (price - lot["entry_price"])
            net_pnl = used * price - exit_cost - entry_basis
            holding_days = max(0, (fill_time - lot["entry_time"]).days)
            closed.append(
                {
                    "ticker": ticker,
                    "entry_time": lot["entry_time"].isoformat(),
                    "exit_time": fill_time.isoformat(),
                    "holding_days": holding_days,
                    "shares": used,
                    "entry_price": lot["entry_price"],
                    "exit_price": price,
                    "gross_pnl": gross_pnl,
                    "net_pnl": net_pnl,
                    "execution_cost": entry_cost + exit_cost,
                    "return_pct": net_pnl / entry_basis if entry_basis > 0 else 0.0,
                    "entry_reason": lot["entry_reason"],
                    "entry_theme": _parse_entry_theme(ticker, lot["entry_reason"]),
                    "entry_price_bucket": low_price_bucket(lot["entry_price"]),
                    "exit_reason": reason,
                    "exit_reason_class": _classify_exit_reason(reason),
                }
            )
            lot["shares"] -= used
            lot["entry_cost_basis"] -= entry_basis
            lot["entry_execution_cost"] -= entry_cost
            remaining -= used
            if lot["shares"] <= 1e-9:
                lots.pop(0)

    return closed


def summarize_performance_attribution(portfolio) -> dict:
    trade_log = list(getattr(portfolio, "trade_log", []) or [])
    closed = build_trade_attribution(trade_log)
    positions = getattr(portfolio, "positions", {}) or {}

    realized_net_pnl = sum(_safe_float(row.get("net_pnl")) for row in closed)
    realized_gross_pnl = sum(_safe_float(row.get("gross_pnl")) for row in closed)
    unrealized_pnl = sum(
        (_safe_float(pos.get("last_price")) - _safe_float(pos.get("avg_cost")))
        * _safe_float(pos.get("shares"))
        for pos in positions.values()
    )
    dividend_income = _safe_float(getattr(portfolio, "dividend_cash_total", 0.0))
    if dividend_income <= 0:
        dividend_income = sum(
            _safe_float(t.get("net_cash_flow"))
            for t in trade_log
            if str(t.get("action", "")).upper() == "DIVIDEND"
        )
    total_execution_cost = sum(_safe_float(t.get("execution_cost")) for t in trade_log)
    gross_traded_value = sum(
        _safe_float(t.get("shares")) * _safe_float(t.get("price"))
        for t in trade_log
        if str(t.get("action", "")).upper() in {"BUY", "SELL", "SELL_PARTIAL"}
    )
    wins = [row for row in closed if _safe_float(row.get("net_pnl")) > 0]

    by_ticker: dict[str, dict[str, Any]] = {}
    by_theme: dict[str, dict[str, Any]] = {}
    by_price_bucket: dict[str, dict[str, Any]] = {}
    exit_reason_counts: dict[str, int] = {}
    for row in closed:
        ticker = row["ticker"]
        slot = by_ticker.setdefault(
            ticker,
            {"ticker": ticker, "closed_trades": 0, "realized_net_pnl": 0.0, "execution_cost": 0.0},
        )
        slot["closed_trades"] += 1
        slot["realized_net_pnl"] += _safe_float(row.get("net_pnl"))
        slot["execution_cost"] += _safe_float(row.get("execution_cost"))

        theme = str(row.get("entry_theme") or theme_bucket(ticker))
        theme_slot = by_theme.setdefault(
            theme,
            {
                "theme": theme,
                "closed_trades": 0,
                "wins": 0,
                "stop_loss_exits": 0,
                "realized_net_pnl": 0.0,
                "execution_cost": 0.0,
                "return_pct_sum": 0.0,
            },
        )
        theme_slot["closed_trades"] += 1
        theme_slot["wins"] += 1 if _safe_float(row.get("net_pnl")) > 0 else 0
        theme_slot["stop_loss_exits"] += (
            1 if row.get("exit_reason_class") == "stop_loss" else 0
        )
        theme_slot["realized_net_pnl"] += _safe_float(row.get("net_pnl"))
        theme_slot["execution_cost"] += _safe_float(row.get("execution_cost"))
        theme_slot["return_pct_sum"] += _safe_float(row.get("return_pct"))

        price_bucket = str(row.get("entry_price_bucket") or "unknown")
        bucket_slot = by_price_bucket.setdefault(
            price_bucket,
            {
                "price_bucket": price_bucket,
                "closed_trades": 0,
                "wins": 0,
                "stop_loss_exits": 0,
                "realized_net_pnl": 0.0,
                "execution_cost": 0.0,
                "return_pct_sum": 0.0,
            },
        )
        bucket_slot["closed_trades"] += 1
        bucket_slot["wins"] += 1 if _safe_float(row.get("net_pnl")) > 0 else 0
        bucket_slot["stop_loss_exits"] += (
            1 if row.get("exit_reason_class") == "stop_loss" else 0
        )
        bucket_slot["realized_net_pnl"] += _safe_float(row.get("net_pnl"))
        bucket_slot["execution_cost"] += _safe_float(row.get("execution_cost"))
        bucket_slot["return_pct_sum"] += _safe_float(row.get("return_pct"))

        exit_class = str(row.get("exit_reason_class") or "unknown")
        exit_reason_counts[exit_class] = exit_reason_counts.get(exit_class, 0) + 1

    latest_buy_reason: dict[str, str] = {}
    for rec in trade_log:
        action = str(rec.get("action", "")).upper()
        ticker = str(rec.get("ticker", "")).upper()
        if action == "BUY" and ticker:
            latest_buy_reason[ticker] = str(rec.get("reason", "") or "")

    for ticker, pos in positions.items():
        ticker = str(ticker).upper()
        shares = _safe_float(pos.get("shares"))
        avg_cost = _safe_float(pos.get("avg_cost"))
        last_price = _safe_float(pos.get("last_price"))
        value = shares * last_price
        unrealized = (last_price - avg_cost) * shares
        theme = _parse_entry_theme(ticker, latest_buy_reason.get(ticker, ""))
        theme_slot = by_theme.setdefault(
            theme,
            {
                "theme": theme,
                "closed_trades": 0,
                "wins": 0,
                "stop_loss_exits": 0,
                "realized_net_pnl": 0.0,
                "execution_cost": 0.0,
                "return_pct_sum": 0.0,
            },
        )
        theme_slot["open_positions"] = int(theme_slot.get("open_positions", 0)) + 1
        theme_slot["open_value"] = _safe_float(theme_slot.get("open_value")) + value
        theme_slot["unrealized_pnl"] = _safe_float(theme_slot.get("unrealized_pnl")) + unrealized

        price_bucket = low_price_bucket(last_price)
        bucket_slot = by_price_bucket.setdefault(
            price_bucket,
            {
                "price_bucket": price_bucket,
                "closed_trades": 0,
                "wins": 0,
                "stop_loss_exits": 0,
                "realized_net_pnl": 0.0,
                "execution_cost": 0.0,
                "return_pct_sum": 0.0,
            },
        )
        bucket_slot["open_positions"] = int(bucket_slot.get("open_positions", 0)) + 1
        bucket_slot["open_value"] = _safe_float(bucket_slot.get("open_value")) + value
        bucket_slot["unrealized_pnl"] = _safe_float(bucket_slot.get("unrealized_pnl")) + unrealized

    def _finalize_group(rows: dict[str, dict[str, Any]], sort_key: str) -> list[dict]:
        finalized = []
        for item in rows.values():
            closed_count = int(item.get("closed_trades", 0) or 0)
            wins_count = int(item.get("wins", 0) or 0)
            stop_count = int(item.get("stop_loss_exits", 0) or 0)
            item["win_rate"] = wins_count / closed_count if closed_count else None
            item["stop_out_rate"] = stop_count / closed_count if closed_count else None
            item["avg_return_pct"] = (
                _safe_float(item.get("return_pct_sum")) / closed_count
                if closed_count else None
            )
            item["total_pnl"] = (
                _safe_float(item.get("realized_net_pnl"))
                + _safe_float(item.get("unrealized_pnl"))
            )
            item.pop("return_pct_sum", None)
            item.pop("wins", None)
            finalized.append(item)
        return sorted(finalized, key=lambda item: _safe_float(item.get(sort_key)), reverse=True)

    return {
        "generated_at": datetime.now().isoformat(),
        "closed_trades": len(closed),
        "open_positions": len(positions),
        "realized_gross_pnl": realized_gross_pnl,
        "realized_net_pnl": realized_net_pnl,
        "unrealized_pnl": unrealized_pnl,
        "dividend_income": dividend_income,
        "total_pnl": realized_net_pnl + unrealized_pnl + dividend_income,
        "total_execution_cost": total_execution_cost,
        "gross_traded_value": gross_traded_value,
        "win_rate": len(wins) / len(closed) if closed else None,
        "exit_reason_counts": dict(
            sorted(exit_reason_counts.items(), key=lambda item: item[1], reverse=True)
        ),
        "by_theme": _finalize_group(by_theme, "total_pnl"),
        "by_price_bucket": _finalize_group(by_price_bucket, "price_bucket"),
        "by_ticker": sorted(
            by_ticker.values(),
            key=lambda item: item["realized_net_pnl"],
            reverse=True,
        ),
        "closed_trade_sample": closed[-20:],
    }


def build_replay_live_parity_report(config: dict | None = None, brain=None) -> dict:
    cfg = dict(config or {})
    try:
        from broker.replay import _replay_kwargs_from_live_config, _resolve_replay_strategy

        replay_kwargs = _replay_kwargs_from_live_config(cfg)
        strategy = _resolve_replay_strategy(cfg)
    except Exception:
        replay_kwargs = {}
        strategy = "unknown"

    runtime_map = {
        "max_positions": "max_positions",
        "max_position_pct": "max_position_pct",
        "min_score": "_base_min_score",
        "penny_pct": "penny_max_pct",
        "max_sector_pct": "max_sector_pct",
        "max_pair_correlation": "max_pair_correlation",
        "avoid_earnings_days": "avoid_earnings_days",
        "rl_phase": "rl_phase",
        "rl_exit_threshold": "rl_exit_threshold",
        "rl_conviction_drop": "rl_conviction_drop",
        "rl_min_score": "rl_min_score",
    }
    mismatches = []
    if brain is not None:
        for replay_key, attr in runtime_map.items():
            if replay_key not in replay_kwargs or not hasattr(brain, attr):
                continue
            live_value = _safe_float(getattr(brain, attr))
            replay_value = _safe_float(replay_kwargs[replay_key])
            if abs(live_value - replay_value) > 1e-9:
                mismatches.append(
                    {
                        "parameter": replay_key,
                        "live_value": live_value,
                        "replay_value": replay_value,
                    }
                )

    return {
        "generated_at": datetime.now().isoformat(),
        "strategy": strategy,
        "compatible": len(mismatches) == 0,
        "runtime_parameter_mismatches": mismatches,
        "shared_replay_kwargs": replay_kwargs,
        "checks": {
            "uses_broker_brain_replay_path": True,
            "execution_spread_configured": "execution_spread" in replay_kwargs,
            "universe_snapshot_configured": bool(cfg.get("universe_snapshot_path")),
        },
    }
