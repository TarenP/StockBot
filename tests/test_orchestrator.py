from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4
from zoneinfo import ZoneInfo

import broker.orchestrator as orch


ET = ZoneInfo("America/New_York")


def test_already_ran_full_cycle_today_uses_orchestrator_state():
    now = datetime(2026, 5, 4, 12, 0, tzinfo=ET)

    assert orch.already_ran_full_cycle_today(
        {"last_full_broker_run_date": "2026-05-04"},
        now,
    )
    assert not orch.already_ran_full_cycle_today(
        {"last_full_broker_run_date": "2026-05-03"},
        now,
    )


def test_load_state_migrates_legacy_complete_run(monkeypatch):
    temp_dir = Path("tests/_tmp") / f"orchestrator_{uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    missing_state = temp_dir / "missing_orchestrator.json"
    legacy_state = temp_dir / ".run_state.json"
    legacy_state.write_text(
        '{"date": "2026-05-04", "pid": 123, "stage": "complete", "status": "complete"}'
    )
    monkeypatch.setattr(orch, "LEGACY_RUN_STATE_PATH", legacy_state)

    state = orch.load_orchestrator_state(missing_state)

    assert state["last_full_broker_run_date"] == "2026-05-04"


def test_get_due_periodic_tasks_respects_phase_and_heavy_gate():
    now = datetime(2026, 5, 4, 12, 0, tzinfo=ET)
    state = {
        "last_maintenance_run_at": "2026-05-03T12:00:00-04:00",
        "last_shadow_validation_at": "2026-04-20T12:00:00-04:00",
    }

    pre = orch.get_due_periodic_tasks(state, now, {}, phase="pre_cycle")
    post_light = orch.get_due_periodic_tasks(state, now, {}, phase="post_cycle")
    post_heavy = orch.get_due_periodic_tasks(
        state,
        now,
        {},
        phase="post_cycle",
        include_heavy=True,
    )

    assert [task.name for task in pre] == ["maintenance"]
    assert [task.name for task in post_light] == []
    assert "shadow_validation" in [task.name for task in post_heavy]


def test_smart_command_defaults_to_status_after_same_day_run(monkeypatch):
    calls = []
    now = datetime(2026, 5, 4, 12, 0, tzinfo=ET)
    state = {"last_full_broker_run_date": "2026-05-04"}

    monkeypatch.setattr(orch, "now_et", lambda: now)
    monkeypatch.setattr(orch, "load_orchestrator_state", lambda: dict(state))
    monkeypatch.setattr(
        orch,
        "refresh_prices_and_status_only",
        lambda args, config, state, invocation_mode: calls.append((invocation_mode, state)) or state,
    )
    monkeypatch.setattr(
        orch,
        "run_full_daily_cycle",
        lambda *args, **kwargs: calls.append(("full", None)) or {},
    )

    args = SimpleNamespace(
        status=False,
        snapshot=False,
        trades=False,
        only_periodic=False,
        force=False,
        dry_run=False,
    )

    orch.run_smart_broker_command(args, {})

    assert calls == [("same_day_status", state)]


def test_smart_command_force_runs_full_cycle(monkeypatch):
    calls = []
    state = {"last_full_broker_run_date": "2026-05-04"}

    monkeypatch.setattr(orch, "load_orchestrator_state", lambda: dict(state))
    monkeypatch.setattr(
        orch,
        "run_full_daily_cycle",
        lambda args, config, state, force, now: calls.append((force, state)) or state,
    )
    monkeypatch.setattr(
        orch,
        "refresh_prices_and_status_only",
        lambda *args, **kwargs: calls.append(("status", None)) or {},
    )

    args = SimpleNamespace(
        status=False,
        snapshot=False,
        trades=False,
        only_periodic=False,
        force=True,
        dry_run=False,
    )

    orch.run_smart_broker_command(args, {})

    assert calls == [(True, state)]
