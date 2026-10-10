"""Contracts of the one-GPU Quyet-1.0-Large System One example."""

from __future__ import annotations

import importlib
import json
import sys
import threading
from pathlib import Path

import httpx
import pytest
import yaml
from fastapi.testclient import TestClient

from kairyu.deploy.spec import load_deployment_spec

EXAMPLE = Path(__file__).resolve().parents[2] / "examples/quyet-1.0-large-1gpu"
CONFIG = {
    "name": "Quyet-1.0-Large",
    "kind": "llm",
    "temperatures": {"choice": 1.3, "score": 1.3, "noul": 1.5},
    "limits": {"max_state_tokens": 6000, "max_prompt_tokens": 8000, "min_state_tokens": 256},
}
QUESTIONS = {"q": {"type": "noul", "instructions": "The message is urgent."}}


EXAMPLE_MODULES = ("control", "verification", "quyet_systemone")


@pytest.fixture
def example(monkeypatch):
    """Import the example's top-level modules, then restore the module table: its
    `verification` must not shadow the repository's `verification` package afterwards."""

    saved = {name: sys.modules.pop(name) for name in EXAMPLE_MODULES if name in sys.modules}
    monkeypatch.syspath_prepend(str(EXAMPLE))
    yield importlib.import_module
    for name in EXAMPLE_MODULES:
        sys.modules.pop(name, None)
    sys.modules.update(saved)


@pytest.fixture
def adapter(example):
    return example("quyet_systemone")


class FakeModel:
    """Stands in for the package's model: answers, or raises what the package would."""

    prompt_version, temps, letter_ids = 2, CONFIG["temperatures"], list(range(10))

    def __init__(self, error: Exception | None = None, gate: threading.Event | None = None):
        self.error, self.gate, self.running = error, gate, threading.Event()

    def predict(self, state, questions):
        """The adapter's startup read."""

        answers = {qid: {"type": "noul", "noul": 0.9, "confidence": 0.9} for qid in questions}
        usage = {"input_tokens": 80, "output_tokens": 0}
        return {"model": "Quyet-1.0-Large", "answers": answers, "usage": usage, "warnings": []}

    def predict_timed(self, state, questions):
        self.running.set()
        if self.gate is not None:
            self.gate.wait(10)
        if self.error is not None:
            raise self.error
        return self.predict(state, questions), 0.01


class FakeRuntime:
    version, config = "1.0.2", CONFIG

    class question_error(ValueError):  # noqa: N801 - the package's QuestionError stand-in
        pass

    def __init__(self, adapter, model: FakeModel):
        self.adapter, self.model = adapter, model

    def build(self, settings):
        def vllm(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/v1/models":
                return httpx.Response(200, json={"data": [{"id": settings.upstream_model}]})
            return httpx.Response(200, json={})

        return self.model, self.adapter.VllmReader(settings, transport=httpx.MockTransport(vllm))


def _client(adapter, model: FakeModel, monkeypatch, **env) -> TestClient:
    for key, value in {"QUYET_MAX_INFLIGHT": "4", "QUYET_MAX_QUEUE": "4", **env}.items():
        monkeypatch.setenv(key, value)
    settings = adapter.Settings()
    return TestClient(adapter.create_app(settings, runtime=FakeRuntime(adapter, model)))


@pytest.mark.parametrize(
    ("body", "error", "status", "detail"),
    [
        (
            {"model": "no-such-model"},
            None,
            400,
            {"error_type": "api_usage_error", "message": "Unknown model: no-such-model"},
        ),
        (
            {"images": ["data:image/png;base64,AAAA"]},
            None,
            400,
            "Quyet-1.0-Large does not support images",
        ),
        ({"think": 64}, None, 400, "Quyet-1.0-Large does not support think"),
        ({"samples": 4}, None, 400, "Quyet-1.0-Large does not support samples"),
        (
            {"questions": {"q": {"type": "rank", "instructions": "x"}}},
            None,
            400,
            {"error_type": "api_usage_error", "message": "Invalid request."},
        ),
        ({"questions": "not an object"}, None, 422, None),
        ({}, "question_error", 400, "too many options"),
        ({}, "refused", 400, "the model rejected this request: context too long"),
        ({}, "unavailable", 503, None),
        # Jev's neutral option values are a plain read.
        (
            {"model": "jev-latest", "think": 0, "samples": 1, "steps": 1, "sequential": False},
            None,
            200,
            None,
        ),
    ],
)
def test_adapter_answers_in_jev_shapes_and_refuses_what_quyet_cannot_read(
    adapter, monkeypatch, body, error, status, detail
):
    """A Jev client gets Jev's status and shape; an option Quyet lacks is refused, not ignored."""

    raised = {
        "question_error": FakeRuntime.question_error("too many options"),
        "refused": adapter.UpstreamRefused("context too long"),
        "unavailable": httpx.ConnectError("down"),
    }.get(error)
    with _client(adapter, FakeModel(raised), monkeypatch) as client:
        payload = {"model": "Quyet-1.0-Large", "state": "Help!", "questions": QUESTIONS}
        response = client.post("/v1/systemone", json={**payload, **body})
    assert response.status_code == status, response.text
    if detail is not None:
        assert response.json()["detail"] == detail
    if status == 422:
        assert isinstance(response.json()["detail"], list)
    if status == 200:
        assert response.json()["answers"]["q"]["noul"] == 0.9
        assert response.headers["server-timing"].startswith("model;dur=10.0")


def test_full_adapter_queue_answers_529(adapter, monkeypatch):
    """Past its in-flight reads and queue the adapter sheds load with Jev's 529."""

    gate = threading.Event()
    model = FakeModel(gate=gate)
    payload = {"model": "Quyet-1.0-Large", "state": "Help!", "questions": QUESTIONS}
    with _client(
        adapter, model, monkeypatch, QUYET_MAX_INFLIGHT="1", QUYET_MAX_QUEUE="0"
    ) as client:
        model.running.clear()
        first = []
        thread = threading.Thread(
            target=lambda: first.append(client.post("/v1/systemone", json=payload))
        )
        thread.start()
        assert model.running.wait(5)
        overloaded = client.post("/v1/systemone", json=payload)
        gate.set()
        thread.join(10)
    assert overloaded.status_code == 529
    assert overloaded.json()["detail"]["error_type"] == "overloaded_error"
    assert first[0].status_code == 200


def test_served_config_matches_example_json():
    """kairyu.yaml publishes System One only, forwards no more reads than the adapter accepts
    (else callers see its 529), and describes the vLLM and checkpoint example.json pins."""

    spec = json.loads((EXAMPLE / "example.json").read_text())
    raw = yaml.safe_load((EXAMPLE / "kairyu.yaml").read_text())
    deployment = load_deployment_spec((EXAMPLE / "kairyu.yaml").read_text())
    systemone = deployment.systemone[spec["systemone"]["model"]]
    adapter = spec["systemone"]["settings"]
    assert systemone.max_concurrency <= adapter["QUYET_MAX_INFLIGHT"] + adapter["QUYET_MAX_QUEUE"]
    assert systemone.max_questions <= adapter["QUYET_MAX_QUESTIONS"]
    assert systemone.upstream_model == spec["systemone"]["upstream_model"]
    assert set(systemone.aliases) == set(spec["systemone"]["aliases"])
    assert deployment.public_models == {spec["systemone"]["model"]}
    (replica,) = raw["pools"][spec["model"]["served_name"]]["replicas"]
    options = replica["options"]
    assert options["model"] == spec["model"]["served_name"]
    assert options["model_revision"] == spec["model"]["revision"]
    assert options["max_model_len"] == spec["vllm"]["settings"]["VLLM_MAX_MODEL_LEN"]
    assert spec["vllm"]["repo_digest"].endswith("@" + options["container_image_digest"])


def test_reference_comparison_fails_only_flipped_confident_decisions(example):
    """The systemone gate fails a served answer whose top option differs from a confident
    official one, and tolerates a flip of a near-even official answer."""

    verification = example("verification")

    def answer(a: float) -> dict:
        return {
            "usage": {"input_tokens": 90},
            "warnings": [],
            "answers": {
                "team": {"type": "choice", "choice": "a", "probabilities": {"a": a, "b": 1 - a}}
            },
        }

    diffs, problems = verification.compare_answers(answer(0.8), answer(0.79), 0.5)
    assert problems == [] and max(diffs) == pytest.approx(0.01)
    _, problems = verification.compare_answers(answer(0.8), answer(0.3), 0.5)
    assert problems == ["team: top 'b' vs 'a'"]
    _, problems = verification.compare_answers(answer(0.56), answer(0.47), 0.5)
    assert problems == []


def test_reference_is_reused_only_for_the_same_requests(example, monkeypatch, tmp_path):
    """A stored reference answers only the requests it was made from: an edited request
    body with an unchanged id must force a new reference run, not pass on old answers."""

    verification = example("verification")
    old = [{"id": "authored-0", "state": "Old state", "questions": QUESTIONS}]
    new = [{"id": "authored-0", "state": "New state", "questions": QUESTIONS}]
    stored = tmp_path / "20261010-old"
    stored.mkdir()
    (stored / "reference.json").write_text(
        json.dumps({"passed": True, "fingerprint": verification.reference_fingerprint(old)})
    )
    (stored / verification.REFERENCE_ANSWERS).write_text(json.dumps({"answers": {}}) + "\n")
    monkeypatch.setattr(verification, "RESULTS_ROOT", tmp_path)
    monkeypatch.setattr(verification, "jevbench_checkout", lambda: tmp_path)
    monkeypatch.setattr(verification, "_reference_requests", lambda _checkout: new)
    with pytest.raises(RuntimeError, match="no passed reference"):
        verification._reference(tmp_path / "20261010-new")
    monkeypatch.setattr(verification, "_reference_requests", lambda _checkout: old)
    requests, answers, source = verification._reference(tmp_path / "20261010-new")
    assert requests == old and source == stored and list(answers) == ["authored-0"]
