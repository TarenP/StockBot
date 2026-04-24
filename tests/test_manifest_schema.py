"""
Tier 1 audit tests: manifest schema completeness.

Ensures every manifest kind contains all required fields so regressions
are caught immediately rather than silently producing incomplete manifests.
"""

import pytest
from pipeline.run_manifest import (
    REQUIRED_MANIFEST_FIELDS,
    REQUIRED_MANIFEST_FIELDS_COMMON,
    validate_manifest,
    write_run_manifest,
    hash_config,
    hash_ticker_list,
)


# ── validate_manifest ─────────────────────────────────────────────────────────

class TestValidateManifest:

    def test_valid_live_cycle_manifest_returns_no_missing(self):
        manifest = {
            "kind": "live_cycle",
            "generated_at": "2026-04-24T12:00:00Z",
            "mode": "live",
            "config_hash": "abc123",
            "resolved_universe_size": 500,
            "resolved_universe_hash": "def456",
            "freshness": {"fresh_price_coverage": 0.98},
            "freshness_gate": {"passed": True},
        }
        missing = validate_manifest(manifest, "live_cycle")
        assert missing == [], f"Unexpected missing fields: {missing}"

    def test_valid_replay_manifest_returns_no_missing(self):
        manifest = {
            "kind": "replay",
            "generated_at": "2026-04-24T12:00:00Z",
            "mode": "replay",
            "config_hash": "abc123",
            "checkpoint_path": "models/best_fold1.pt",
            "resolved_universe_size": 500,
            "resolved_universe_hash": "def456",
            "replay_window": {"start": "2024-01-01", "end": "2025-01-01"},
            "benchmark": {"available": True},
            "friction": {"execution_spread": 0.001},
        }
        missing = validate_manifest(manifest, "replay")
        assert missing == [], f"Unexpected missing fields: {missing}"

    def test_missing_common_fields_are_reported(self):
        manifest = {
            "mode": "live",
            "config_hash": "abc",
            "resolved_universe_size": 100,
            "resolved_universe_hash": "xyz",
            "freshness": {},
            "freshness_gate": {},
            # missing: kind, generated_at
        }
        missing = validate_manifest(manifest, "live_cycle")
        assert "kind" in missing
        assert "generated_at" in missing

    def test_missing_kind_specific_fields_are_reported(self):
        manifest = {
            "kind": "live_cycle",
            "generated_at": "2026-04-24T12:00:00Z",
            # missing all live_cycle-specific fields
        }
        missing = validate_manifest(manifest, "live_cycle")
        for field in REQUIRED_MANIFEST_FIELDS["live_cycle"]:
            assert field in missing, f"Expected '{field}' in missing but got: {missing}"

    def test_unknown_kind_only_checks_common_fields(self):
        manifest = {
            "kind": "custom_kind",
            "generated_at": "2026-04-24T12:00:00Z",
        }
        missing = validate_manifest(manifest, "custom_kind")
        assert missing == []

    def test_all_required_fields_are_strings(self):
        for kind, fields in REQUIRED_MANIFEST_FIELDS.items():
            for field in fields:
                assert isinstance(field, str), f"Field {field!r} in {kind!r} is not a string"
        for field in REQUIRED_MANIFEST_FIELDS_COMMON:
            assert isinstance(field, str)


# ── write_run_manifest auto-injection ─────────────────────────────────────────

class TestWriteRunManifestAutoFields:

    def test_write_injects_kind_and_generated_at(self, tmp_path):
        out = tmp_path / "manifest.json"
        write_run_manifest("test_kind", {"foo": "bar"}, output_path=out)
        import json
        data = json.loads(out.read_text())
        assert data["kind"] == "test_kind"
        assert "generated_at" in data

    def test_write_injects_code_version(self, tmp_path):
        out = tmp_path / "manifest.json"
        write_run_manifest("test_kind", {"foo": "bar"}, output_path=out)
        import json
        data = json.loads(out.read_text())
        assert "code_version" in data
        assert isinstance(data["code_version"], str)
        assert len(data["code_version"]) > 0

    def test_write_warns_on_missing_required_fields(self, tmp_path, caplog):
        import logging
        out = tmp_path / "manifest.json"
        with caplog.at_level(logging.WARNING, logger="pipeline.run_manifest"):
            write_run_manifest(
                "live_cycle",
                {"mode": "live"},  # missing most required fields
                output_path=out,
            )
        assert any("missing required fields" in r.message for r in caplog.records)

    def test_write_does_not_warn_when_all_fields_present(self, tmp_path, caplog):
        import logging
        out = tmp_path / "manifest.json"
        complete = {
            "mode": "live",
            "config_hash": "abc",
            "resolved_universe_size": 100,
            "resolved_universe_hash": "xyz",
            "freshness": {},
            "freshness_gate": {},
        }
        with caplog.at_level(logging.WARNING, logger="pipeline.run_manifest"):
            write_run_manifest("live_cycle", complete, output_path=out)
        assert not any("missing required fields" in r.message for r in caplog.records)

    def test_manifest_is_valid_json(self, tmp_path):
        out = tmp_path / "manifest.json"
        write_run_manifest("replay", {
            "mode": "replay",
            "config_hash": hash_config({"min_score": 0.6}),
            "checkpoint_path": "models/best_fold1.pt",
            "resolved_universe_size": 500,
            "resolved_universe_hash": hash_ticker_list(["AAPL", "MSFT"]),
            "replay_window": {"start": "2024-01-01", "end": "2025-01-01"},
            "benchmark": {"available": True},
            "friction": {"execution_spread": 0.001},
        }, output_path=out)
        import json
        data = json.loads(out.read_text())
        assert isinstance(data, dict)
        assert data["mode"] == "replay"
