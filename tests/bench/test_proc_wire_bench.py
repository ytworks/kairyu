"""Deterministic complexity gate for the process delta wire."""

from bench import proc_wire_bench as proc_wire
from bench.proc_wire_bench import (
    add_provenance,
    artifact_sha256,
    evaluate,
    measure_wire_bytes,
)


def test_raw_wire_volume_is_linear_for_v2_and_quadratic_for_legacy():
    result = evaluate((64, 128, 256))
    assert result["all_passed"] is True
    assert all(check["passed"] for check in result["checks"])
    assert all(
        row["v2_empirical_exponent"] < 1.15
        and row["legacy_empirical_exponent"] > 1.80
        for row in result["growth"]
    )


def test_measurement_retains_every_frame_and_each_item_once():
    result = measure_wire_bytes(96)
    assert len(result["legacy_event_bytes"]) == 96
    assert len(result["v2_event_bytes"]) == 96
    assert result["v2_new_token_items"] == 96
    assert result["v2_new_logprob_items"] == 96
    assert result["v2_new_logprob_content_items"] == 96
    assert result["v2_text_delta_chars"] == 96
    assert result["v2_structure_valid"] is True


def test_artifact_hash_ignores_only_its_own_field():
    result = {"schema_version": 1, "measurements": [1, 2, 3]}
    digest = artifact_sha256(result)
    result["artifact_sha256"] = digest
    assert artifact_sha256(result) == digest
    result["measurements"].append(4)
    assert artifact_sha256(result) != digest


def test_unexpected_or_missing_delta_payload_fails_the_structure_gate(monkeypatch):
    production_encoder = proc_wire.event_from_update

    def unexpected_key(*args, **kwargs):
        event = production_encoder(*args, **kwargs)
        if event is not None and event.get("event") == "delta":
            event["accidental_cumulative"] = "x"
        return event

    monkeypatch.setattr(proc_wire, "event_from_update", unexpected_key)
    assert evaluate((32, 64))["all_passed"] is False

    def missing_rich_content(*args, **kwargs):
        event = production_encoder(*args, **kwargs)
        if event is not None and event.get("event") == "delta":
            event.pop("new_logprob_content", None)
        return event

    monkeypatch.setattr(proc_wire, "event_from_update", missing_rich_content)
    assert evaluate((32, 64))["all_passed"] is False


def test_dirty_tracked_source_is_a_binding_failure(monkeypatch):
    def fake_git_output(*args):
        if args[0] == "status":
            return " M kairyu/engine/zmq_backend.py"
        if args == ("rev-parse", "HEAD"):
            return "a" * 40
        raise AssertionError(args)

    monkeypatch.setattr(proc_wire, "_git_output", fake_git_output)
    result = add_provenance(evaluate((32, 64)))
    assert result["source"]["tracked_dirty"] is True
    assert result["all_passed"] is False
    assert result["checks"][-1] == {
        "name": "clean_tracked_source",
        "passed": False,
        "actual": True,
        "required": "false",
    }
