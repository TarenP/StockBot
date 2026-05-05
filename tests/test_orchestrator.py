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


def test_status_refresh_writes_current_outputs_and_state(monkeypatch):
    state = {}
    writes = {}
    invoked = []
    now = datetime(2026, 5, 4, 12, 0, tzinfo=ET)

    monkeypatch.setattr(orch, "_invoke_broker_main", lambda config, argv: invoked.append(argv))
    monkeypatch.setattr(orch, "_show_shadow_summary", lambda: None)
    monkeypatch.setattr(orch, "save_orchestrator_state", lambda payload: writes.setdefault("state", payload))
    monkeypatch.setattr(
        orch,
        "write_current_outputs",
        lambda **kwargs: writes.setdefault("outputs", kwargs),
    )

    args = SimpleNamespace(refresh_prices=False)
    updated = orch.refresh_prices_and_status_only(
        args,
        {},
        state,
        invocation_mode="same_day_status",
        run_id="run_status",
        now=now,
    )

    assert invoked == [["--status"]]
    assert updated["last_status_refresh_at"].startswith("2026-05-04T12:00:00")
    assert updated["last_price_refresh_at"].startswith("2026-05-04T12:00:00")
    assert writes["outputs"]["mode"] == "same_day_status"
    assert writes["outputs"]["run_id"] == "run_status"


def test_write_current_outputs_keeps_canonical_files(monkeypatch):
    temp_dir = Path("tests/_tmp") / f"orchestrator_{uuid4().hex}"
    current_dir = temp_dir / "current"
    monkeypatch.setattr(orch, "CURRENT_DIR", current_dir)
    monkeypatch.setattr(
        orch,
        "_portfolio_state",
        lambda: {
            "cash": 123.0,
            "positions": {"AAA": {"shares": 1}},
            "last_saved": "2026-05-04T12:00:00",
            "trade_log": [{"time": "2026-05-04T10:00:00", "ticker": "AAA"}],
        },
    )

    orch.write_current_outputs(
        state={"last_full_broker_run_date": "2026-05-04"},
        run_id="run_1",
        mode="full_cycle",
        force=True,
        now=datetime(2026, 5, 4, 12, 0, tzinfo=ET),
    )

    assert (current_dir / "latest_status.json").exists()
    assert (current_dir / "today_summary.json").exists()
    assert (current_dir / "today_trade_log.json").exists()
    assert '"run_id": "run_1"' in (current_dir / "latest_status.json").read_text()


def test_full_cycle_writes_metadata_and_marks_complete(monkeypatch):
    now = datetime(2026, 5, 4, 12, 0, tzinfo=ET)
    state = {}
    metadata = {}
    legacy = []
    temp_dir = Path("tests/_tmp") / f"orchestrator_{uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(orch, "_ensure_no_live_duplicate", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        orch,
        "_write_legacy_run_state",
        lambda stage, status="running", now=None: legacy.append((stage, status)),
    )
    monkeypatch.setattr(
        orch,
        "run_due_periodic_tasks",
        lambda state, now, config, args, phase, run_id, include_heavy=False, force=False: (
            state,
            [f"{phase}_task"],
            [],
            {"prices_updated": "2026-05-04"} if phase == "pre_cycle" else None,
        ),
    )
    monkeypatch.setattr(orch, "_invoke_broker_main", lambda config, argv, maintenance_context=None: None)
    monkeypatch.setattr(orch, "write_current_outputs", lambda **kwargs: None)
    monkeypatch.setattr(orch, "save_orchestrator_state", lambda state: None)
    monkeypatch.setattr(
        orch,
        "_write_run_metadata",
        lambda **kwargs: metadata.update(kwargs) or temp_dir / "run_metadata.json",
    )

    args = SimpleNamespace(no_periodic=False, force_periodic=False)
    updated = orch.run_full_daily_cycle(
        args,
        {},
        state,
        run_id="run_full",
        now=now,
    )

    assert updated["last_full_broker_run_date"] == "2026-05-04"
    assert updated["last_daily_run_id"] == "run_full"
    assert ("complete", "complete") in legacy
    assert "full_broker_cycle" in metadata["tasks_executed"]


def test_full_cycle_setup_failure_does_not_mark_complete(monkeypatch):
    now = datetime(2026, 5, 4, 12, 0, tzinfo=ET)
    legacy = []
    temp_dir = Path("tests/_tmp") / f"orchestrator_{uuid4().hex}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(orch, "_ensure_no_live_duplicate", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        orch,
        "_write_legacy_run_state",
        lambda stage, status="running", now=None: legacy.append((stage, status)),
    )
    monkeypatch.setattr(
        orch,
        "run_due_periodic_tasks",
        lambda state, now, config, args, phase, run_id, include_heavy=False, force=False: (
            state,
            [],
            [],
            None,
        ),
    )
    monkeypatch.setattr(
        orch,
        "_invoke_broker_main",
        lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError("Price data not found")),
    )
    monkeypatch.setattr(orch, "save_orchestrator_state", lambda state: None)
    monkeypatch.setattr(orch, "_write_run_metadata", lambda **kwargs: temp_dir / "run_metadata.json")

    args = SimpleNamespace(no_periodic=False, force_periodic=False)
    updated = orch.run_full_daily_cycle(
        args,
        {},
        {},
        run_id="run_fail",
        now=now,
    )

    assert updated.get("last_full_broker_run_date") is None
    assert updated["last_failed_task"] == "full_daily_cycle"
    assert ("failed", "failed") in legacy
    assert ("complete", "complete") not in legacy
