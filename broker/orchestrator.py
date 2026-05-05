"""
Smart Broker.py orchestration.

This module keeps the top-level command daily-idempotent. A plain
``python Broker.py`` runs the full broker cycle once per trading day; later
same-day invocations refresh marks and status only. Heavy supporting work is
registered here with due checks instead of being run blindly on every command.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

STATE_PATH = Path("broker/state/orchestrator_state.json")
LEGACY_RUN_STATE_PATH = Path("broker/state/.run_state.json")
RUNS_DIR = Path("broker/state/runs")
CURRENT_DIR = Path("broker/state/current")
ET = ZoneInfo("America/New_York")

ORCHESTRATOR_STATE_FIELDS = {
    "last_full_broker_run_at": None,
    "last_full_broker_run_date": None,
    "last_price_refresh_at": None,
    "last_status_refresh_at": None,
    "last_policy_matrix_run_at": None,
    "last_autotune_run_at": None,
    "last_shadow_validation_at": None,
    "last_replay_validation_at": None,
    "last_universe_refresh_at": None,
    "last_daily_run_id": None,
    "last_daily_output_root": None,
    "last_force_run_at": None,
    "last_force_run_date": None,
    "last_failed_task": None,
    "last_failed_reason": None,
}


@dataclass(frozen=True)
class PeriodicTask:
    name: str
    cadence_days: int
    last_run_field: str
    phase: str
    blocking: bool = False
    heavy: bool = False
    skip_on_same_day_rerun: bool = True


TASK_REGISTRY = (
    PeriodicTask(
        name="maintenance",
        cadence_days=1,
        last_run_field="last_maintenance_run_at",
        phase="pre_cycle",
        blocking=False,
        heavy=False,
    ),
    PeriodicTask(
        name="shadow_validation",
        cadence_days=7,
        last_run_field="last_shadow_validation_at",
        phase="post_cycle",
        blocking=False,
        heavy=True,
    ),
    PeriodicTask(
        name="replay_validation",
        cadence_days=7,
        last_run_field="last_replay_validation_at",
        phase="post_cycle",
        blocking=False,
        heavy=True,
    ),
    PeriodicTask(
        name="policy_review",
        cadence_days=30,
        last_run_field="last_policy_matrix_run_at",
        phase="post_cycle",
        blocking=False,
        heavy=True,
    ),
)


def now_et() -> datetime:
    return datetime.now(ET)


def _iso_now(now: datetime | None = None) -> str:
    return (now or now_et()).isoformat()


def _today_iso(now: datetime | None = None) -> str:
    return (now or now_et()).date().isoformat()


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def load_orchestrator_state(path: str | Path = STATE_PATH) -> dict[str, Any]:
    state = dict(ORCHESTRATOR_STATE_FIELDS)
    loaded = _read_json(Path(path), {})
    if isinstance(loaded, dict):
        state.update(loaded)
    legacy = _read_json(LEGACY_RUN_STATE_PATH, {})
    if (
        not state.get("last_full_broker_run_date")
        and isinstance(legacy, dict)
        and legacy.get("status") == "complete"
        and legacy.get("date")
    ):
        state["last_full_broker_run_date"] = legacy.get("date")
        state["last_full_broker_run_at"] = legacy.get("date")
    state.setdefault("schema_version", 1)
    state.setdefault("task_history", [])
    return state


def save_orchestrator_state(state: dict[str, Any], path: str | Path = STATE_PATH) -> None:
    _write_json(Path(path), state)


def already_ran_full_cycle_today(state: dict[str, Any], now: datetime | None = None) -> bool:
    return state.get("last_full_broker_run_date") == _today_iso(now)


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(ET)


def _days_since(value: str | None, now: datetime | None = None) -> int:
    dt = _parse_iso(value)
    if dt is None:
        return 9999
    return ((now or now_et()).date() - dt.date()).days


def get_due_periodic_tasks(
    state: dict[str, Any],
    now: datetime | None = None,
    config: dict | None = None,
    *,
    phase: str | None = None,
    include_heavy: bool = False,
    force: bool = False,
) -> list[PeriodicTask]:
    cfg = config or {}
    due: list[PeriodicTask] = []
    for task in TASK_REGISTRY:
        if phase and task.phase != phase:
            continue
        if task.heavy and not include_heavy and not force:
            continue
        if bool(cfg.get(f"disable_{task.name}", False)):
            continue
        if force or _days_since(state.get(task.last_run_field), now) >= task.cadence_days:
            due.append(task)
    return due


def _record_task(
    state: dict[str, Any],
    name: str,
    status: str,
    *,
    now: datetime | None = None,
    reason: str | None = None,
    run_id: str | None = None,
) -> None:
    stamp = _iso_now(now)
    history = list(state.get("task_history") or [])
    history.append(
        {
            "time": stamp,
            "task": name,
            "status": status,
            "reason": reason,
            "run_id": run_id,
        }
    )
    state["task_history"] = history[-100:]
    if status == "failed":
        state["last_failed_task"] = name
        state["last_failed_reason"] = reason
    elif name == state.get("last_failed_task"):
        state["last_failed_task"] = None
        state["last_failed_reason"] = None


def _run_id(now: datetime | None = None, *, force: bool = False, mode: str = "full") -> str:
    dt = now or now_et()
    suffix = "_force" if force else ""
    return f"{dt.strftime('%Y%m%d_%H%M%S')}_{mode}{suffix}"


def _write_legacy_run_state(stage: str, status: str = "running", now: datetime | None = None) -> None:
    _write_json(
        LEGACY_RUN_STATE_PATH,
        {
            "date": _today_iso(now),
            "pid": os.getpid(),
            "stage": stage,
            "status": status,
        },
    )


def _ensure_no_live_duplicate(now: datetime | None = None, *, force: bool = False) -> None:
    prev = _read_json(LEGACY_RUN_STATE_PATH, {})
    if not isinstance(prev, dict):
        return
    if prev.get("date") != _today_iso(now) or prev.get("status") != "running":
        return
    prev_pid = int(prev.get("pid", 0) or 0)
    if prev_pid <= 0 or prev_pid == os.getpid():
        return
    try:
        os.kill(prev_pid, 0)
    except (OSError, ProcessLookupError):
        logger.warning(
            "Previous run PID %d appears crashed at stage '%s'; continuing.",
            prev_pid,
            prev.get("stage", "unknown"),
        )
        return
    if force:
        logger.warning("Force run requested while PID %d appears active; continuing by request.", prev_pid)
        return
    raise RuntimeError(f"Another broker process appears to be running (PID {prev_pid}).")


@contextmanager
def _temporary_argv(args: list[str]):
    old = sys.argv[:]
    sys.argv = [old[0] if old else "Broker.py", *args]
    try:
        yield
    finally:
        sys.argv = old


@contextmanager
def _temporary_env(name: str, value: str | None):
    old = os.environ.get(name)
    if value is None:
        yield
        return
    os.environ[name] = value
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = old


def _invoke_broker_main(
    config: dict,
    argv: list[str],
    maintenance_context: dict | None = None,
    display_command: str | None = None,
) -> None:
    from broker.broker import main as broker_main

    with _temporary_argv(argv), _temporary_env("BROKER_DISPLAY_COMMAND", display_command):
        broker_main(config, maintenance_context=maintenance_context)


def _refresh_broader_price_cache(config: dict) -> None:
    from broker.broker import _resolve_cycle_universe
    from pipeline.updater import update_parquet

    universe = _resolve_cycle_universe(config=config, save_dir="models")
    n_prices = update_parquet(universe=universe, save_dir="models", config=config)
    logger.info("Price cache refresh complete: %s new row(s)", n_prices)


def _show_shadow_summary() -> None:
    try:
        from broker.shadows import get_shadow_summary

        print(get_shadow_summary())
    except Exception as exc:
        logger.debug("Could not show shadow summary: %s", exc)


def _portfolio_state() -> dict[str, Any]:
    path = Path("broker/state/portfolio.json")
    data = _read_json(path, {})
    return data if isinstance(data, dict) else {}


def _trade_log_for_today(portfolio: dict[str, Any], today: str) -> list[dict[str, Any]]:
    out = []
    for rec in portfolio.get("trade_log", []) or []:
        if str(rec.get("time", "")).startswith(today):
            out.append(rec)
    return out


def write_current_outputs(
    *,
    state: dict[str, Any],
    run_id: str,
    mode: str,
    force: bool = False,
    now: datetime | None = None,
) -> None:
    today = _today_iso(now)
    portfolio = _portfolio_state()
    latest_status = {
        "date": today,
        "run_id": run_id,
        "mode": mode,
        "force": bool(force),
        "cash": portfolio.get("cash"),
        "positions": portfolio.get("positions", {}),
        "last_saved": portfolio.get("last_saved"),
        "orchestrator": {
            "last_full_broker_run_at": state.get("last_full_broker_run_at"),
            "last_full_broker_run_date": state.get("last_full_broker_run_date"),
            "last_status_refresh_at": state.get("last_status_refresh_at"),
            "last_failed_task": state.get("last_failed_task"),
            "last_failed_reason": state.get("last_failed_reason"),
        },
    }
    _write_json(CURRENT_DIR / "latest_status.json", latest_status)
    _write_json(
        CURRENT_DIR / "today_summary.json",
        {
            "date": today,
            "run_id": run_id,
            "mode": mode,
            "force": bool(force),
            "cash": portfolio.get("cash"),
            "position_count": len(portfolio.get("positions", {}) or {}),
            "trade_count_today": len(_trade_log_for_today(portfolio, today)),
        },
    )
    _write_json(CURRENT_DIR / "today_trade_log.json", _trade_log_for_today(portfolio, today))

    manifest = Path("broker/state/last_live_manifest.json")
    if manifest.exists():
        shutil.copyfile(manifest, CURRENT_DIR / "today_manifest.json")


def _write_run_metadata(
    *,
    run_id: str,
    mode: str,
    force: bool,
    tasks_executed: list[str],
    tasks_skipped: list[str],
    state_before: dict[str, Any],
    state_after: dict[str, Any],
    error: str | None = None,
) -> Path:
    path = RUNS_DIR / run_id / "run_metadata.json"
    _write_json(
        path,
        {
            "run_id": run_id,
            "mode": mode,
            "force": bool(force),
            "created_at": _iso_now(),
            "tasks_executed": tasks_executed,
            "tasks_skipped": tasks_skipped,
            "state_before": state_before,
            "state_after": state_after,
            "error": error,
        },
    )
    return path


def _run_maintenance_task(config: dict) -> dict[str, Any]:
    from pipeline.maintenance import run_maintenance

    return run_maintenance(
        initial_cash=float(config.get("cash", 10_000)),
        config=config,
    )


def _reload_config(default: dict) -> dict:
    try:
        from pipeline.universe_resolver import load_typed_config

        return load_typed_config("broker.config")
    except Exception as exc:
        logger.debug("Could not reload broker.config after periodic task: %s", exc)
        return default


def _run_shadow_task(config: dict, args: Any) -> None:
    from pipeline.data import load_master
    from broker.replay import _build_price_lookup
    from broker.shadows import run_shadow_cycle, get_shadow_summary

    logger.info("Loading data for shadow portfolios...")
    df_features = load_master(top_n=int(config.get("top_n", 500)), config=config)
    price_lookup = _build_price_lookup()
    run_shadow_cycle(
        df_features=df_features,
        price_lookup=price_lookup,
        live_config=config,
        checkpoint_path=config.get("rl_checkpoint_path"),
        allow_promotion=getattr(args, "approve_promotion", False),
        validation_top_n=int(config.get("shadow_validation_top_n", 6)),
        validation_replay_years=int(config.get("shadow_replay_years", 1)),
    )
    print(get_shadow_summary())


def run_due_periodic_tasks(
    state: dict[str, Any],
    now: datetime,
    config: dict,
    args: Any,
    *,
    phase: str,
    run_id: str,
    include_heavy: bool = False,
    force: bool = False,
) -> tuple[dict[str, Any], list[str], list[str], dict[str, Any] | None]:
    tasks = get_due_periodic_tasks(
        state,
        now,
        config,
        phase=phase,
        include_heavy=include_heavy,
        force=force,
    )
    executed: list[str] = []
    skipped = [
        task.name for task in TASK_REGISTRY
        if task.phase == phase and task.name not in {due.name for due in tasks}
    ]
    maintenance_context: dict[str, Any] | None = None

    for task in tasks:
        if task.name == "maintenance" and getattr(args, "no_maintenance", False):
            skipped.append(task.name)
            continue
        if task.name == "shadow_validation" and getattr(args, "no_shadows", False):
            skipped.append(task.name)
            continue
        if task.name in {"policy_review", "replay_validation"} and not bool(
            config.get(f"enable_{task.name}", False)
        ):
            skipped.append(task.name)
            continue
        try:
            if task.name == "maintenance":
                maintenance_context = _run_maintenance_task(config)
                config.update(_reload_config(config))
                if maintenance_context.get("prices_ran"):
                    state["last_price_refresh_at"] = _iso_now(now)
                    state["last_universe_refresh_at"] = _iso_now(now)
                if maintenance_context.get("autotune_ran"):
                    state["last_autotune_run_at"] = _iso_now(now)
            elif task.name == "shadow_validation":
                _run_shadow_task(config, args)
            else:
                logger.info("Periodic task %s is registered but disabled until configured.", task.name)
                skipped.append(task.name)
                continue
            state[task.last_run_field] = _iso_now(now)
            _record_task(state, task.name, "complete", now=now, run_id=run_id)
            executed.append(task.name)
        except Exception as exc:
            logger.warning("Periodic task %s failed (continuing): %s", task.name, exc)
            _record_task(state, task.name, "failed", now=now, reason=str(exc), run_id=run_id)
            if task.blocking:
                raise
    return state, executed, skipped, maintenance_context


def refresh_prices_and_status_only(
    args: Any,
    config: dict,
    state: dict[str, Any] | None = None,
    *,
    invocation_mode: str = "status_only",
    run_id: str | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    state = load_orchestrator_state() if state is None else state
    now = now or now_et()
    run_id = run_id or _run_id(now, mode="status")

    if getattr(args, "refresh_prices", False):
        try:
            _refresh_broader_price_cache(config)
            state["last_price_refresh_at"] = _iso_now(now)
        except Exception as exc:
            logger.warning("Status price refresh failed; using cached prices: %s", exc)
            _record_task(state, "price_refresh", "failed", now=now, reason=str(exc), run_id=run_id)

    display_command = (
        "python Broker.py"
        if invocation_mode == "same_day_status"
        else "python Broker.py --status"
    )
    _invoke_broker_main(config, ["--status"], display_command=display_command)
    _show_shadow_summary()
    state["last_price_refresh_at"] = _iso_now(now)
    state["last_status_refresh_at"] = _iso_now(now)
    write_current_outputs(state=state, run_id=run_id, mode=invocation_mode, now=now)
    save_orchestrator_state(state)
    return state


def run_full_daily_cycle(
    args: Any,
    config: dict,
    state: dict[str, Any] | None = None,
    *,
    force: bool = False,
    run_id: str | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    state = load_orchestrator_state() if state is None else state
    now = now or now_et()
    run_id = run_id or _run_id(now, force=force, mode="full")
    state_before = json.loads(json.dumps(state, default=str))
    executed: list[str] = []
    skipped: list[str] = []
    error: str | None = None

    _ensure_no_live_duplicate(now, force=force)
    _write_legacy_run_state("startup", now=now)
    try:
        _write_legacy_run_state("maintenance", now=now)
        state, pre_executed, pre_skipped, maintenance_context = run_due_periodic_tasks(
            state,
            now,
            config,
            args,
            phase="pre_cycle",
            run_id=run_id,
            include_heavy=False,
            force=force,
        )
        executed.extend(pre_executed)
        skipped.extend(pre_skipped)

        _write_legacy_run_state("trading", now=now)
        try:
            _invoke_broker_main(config, [], maintenance_context=maintenance_context)
            executed.append("full_broker_cycle")
        except FileNotFoundError as exc:
            if "stooq_panel.parquet" in str(exc) or "Price data not found" in str(exc):
                error = str(exc)
                print("\n" + "=" * 60)
                print("  FIRST-TIME SETUP REQUIRED")
                print("=" * 60)
                print("  Price data not found. Train the model first:\n")
                print("    python Agent.py --mode train --folds 10\n")
                print("  This downloads historical data and trains the RL model.")
                print("  Takes 2-6 hours on CPU, 30-60 min on GPU.")
                print("  After that, just run: python Broker.py")
                print("=" * 60 + "\n")
                state["last_failed_task"] = "full_daily_cycle"
                state["last_failed_reason"] = error
                _write_legacy_run_state("failed", status="failed", now=now)
                return state
            else:
                raise

        include_heavy = bool(getattr(args, "force_periodic", False)) or bool(
            config.get("run_heavy_periodic_from_broker", False)
        )
        if not getattr(args, "no_periodic", False):
            _write_legacy_run_state("post_cycle_periodic", now=now)
            state, post_executed, post_skipped, _unused = run_due_periodic_tasks(
                state,
                now,
                config,
                args,
                phase="post_cycle",
                run_id=run_id,
                include_heavy=include_heavy,
                force=bool(getattr(args, "force_periodic", False)),
            )
            executed.extend(post_executed)
            skipped.extend(post_skipped)

        state["last_full_broker_run_at"] = _iso_now(now)
        state["last_full_broker_run_date"] = _today_iso(now)
        state["last_price_refresh_at"] = _iso_now(now)
        state["last_daily_run_id"] = run_id
        state["last_daily_output_root"] = str(RUNS_DIR / run_id)
        if force:
            state["last_force_run_at"] = _iso_now(now)
            state["last_force_run_date"] = _today_iso(now)
        state["last_status_refresh_at"] = _iso_now(now)
        _write_legacy_run_state("complete", status="complete", now=now)
        write_current_outputs(state=state, run_id=run_id, mode="full_cycle", force=force, now=now)
        return state
    except Exception as exc:
        error = str(exc)
        state["last_failed_task"] = "full_daily_cycle"
        state["last_failed_reason"] = error
        _write_legacy_run_state("failed", status="failed", now=now)
        raise
    finally:
        _write_run_metadata(
            run_id=run_id,
            mode="full_cycle",
            force=force,
            tasks_executed=executed,
            tasks_skipped=skipped,
            state_before=state_before,
            state_after=state,
            error=error,
        )
        save_orchestrator_state(state)


def run_only_periodic(args: Any, config: dict, state: dict[str, Any] | None = None) -> dict[str, Any]:
    state = load_orchestrator_state() if state is None else state
    now = now_et()
    run_id = _run_id(now, force=bool(getattr(args, "force_periodic", False)), mode="periodic")
    state_before = json.loads(json.dumps(state, default=str))
    executed: list[str] = []
    skipped: list[str] = []
    try:
        for phase in ("pre_cycle", "post_cycle"):
            state, ran, missed, _context = run_due_periodic_tasks(
                state,
                now,
                config,
                args,
                phase=phase,
                run_id=run_id,
                include_heavy=True,
                force=bool(getattr(args, "force_periodic", False)),
            )
            executed.extend(ran)
            skipped.extend(missed)
        return state
    finally:
        _write_run_metadata(
            run_id=run_id,
            mode="periodic_only",
            force=bool(getattr(args, "force_periodic", False)),
            tasks_executed=executed,
            tasks_skipped=skipped,
            state_before=state_before,
            state_after=state,
        )
        save_orchestrator_state(state)


def run_smart_broker_command(args: Any, config: dict) -> dict[str, Any]:
    state = load_orchestrator_state()
    now = now_et()
    force = bool(getattr(args, "force", False))

    if getattr(args, "dry_run", False):
        already_ran = already_ran_full_cycle_today(state, now)
        if getattr(args, "status", False) or getattr(args, "snapshot", False):
            planned_mode = "status_only"
        elif getattr(args, "trades", False):
            planned_mode = "trades"
        elif getattr(args, "only_periodic", False):
            planned_mode = "periodic_only"
        elif force:
            planned_mode = "full_cycle_force"
        elif already_ran:
            planned_mode = "same_day_status"
        else:
            planned_mode = "full_cycle"
        if planned_mode in {"status_only", "same_day_status", "trades"}:
            due_pre = []
            due_post = []
        else:
            due_pre = [
                task.name
                for task in get_due_periodic_tasks(state, now, config, phase="pre_cycle")
            ]
            due_post = [
                task.name
                for task in get_due_periodic_tasks(
                    state,
                    now,
                    config,
                    phase="post_cycle",
                    include_heavy=bool(getattr(args, "force_periodic", False))
                    or planned_mode == "periodic_only",
                )
            ]
        print(
            json.dumps(
                {
                    "already_ran_today": already_ran,
                    "would_force": force,
                    "mode": planned_mode,
                    "due_pre_cycle_tasks": due_pre,
                    "due_post_cycle_tasks": due_post,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return state

    if getattr(args, "trades", False):
        _invoke_broker_main(config, ["--trades"])
        return state

    if getattr(args, "only_periodic", False):
        return run_only_periodic(args, config, state)

    if getattr(args, "status", False) or getattr(args, "snapshot", False):
        return refresh_prices_and_status_only(args, config, state, invocation_mode="status")

    if force:
        logger.info("Force run requested; rerunning full daily cycle.")
        return run_full_daily_cycle(args, config, state, force=True, now=now)

    if already_ran_full_cycle_today(state, now):
        today = _today_iso(now)
        logger.info("Full broker cycle already ran today (%s); refreshing status only.", today)
        print(f"\n  Already ran full broker cycle today ({today}). Refreshing prices and status only.\n")
        return refresh_prices_and_status_only(args, config, state, invocation_mode="same_day_status")

    return run_full_daily_cycle(args, config, state, force=False, now=now)
