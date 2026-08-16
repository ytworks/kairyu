"""Sandbox executor stages in the Conductor DAG (ECO-D3)."""

import json

import pytest

from kairyu.engine.backend import GenerationRequest, GenerationResult
from kairyu.orchestration.budget import Budget
from kairyu.orchestration.conductor import (
    Conductor,
    ExecutorRoleConfig,
    RoleSpec,
)
from kairyu.orchestration.execution import (
    CaseOutcome,
    ExecutionReport,
    ExecutionRequest,
)
from kairyu.outputs import CompletionOutput


class ScriptedBackend:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.requests_seen: list[GenerationRequest] = []

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        self.requests_seen.append(request)
        text = self._responses.pop(0)
        return GenerationResult(
            request_id=request.request_id,
            prompt=request.prompt,
            completions=(CompletionOutput(index=0, text=text, token_ids=()),),
        )

    async def stream(self, request):  # pragma: no cover - unused
        yield await self.generate(request)

    async def shutdown(self) -> None:
        return None


class FakeExecutor:
    def __init__(
        self,
        report: ExecutionReport | None = None,
        *,
        error: Exception | None = None,
    ) -> None:
        self.report = report or ExecutionReport(
            status="ok",
            tests=(CaseOutcome(id="test_solution.py::test_x", outcome="passed"),),
            passed=1,
            duration_ms=42,
        )
        self.error = error
        self.requests_seen: list[ExecutionRequest] = []

    async def execute(self, request: ExecutionRequest) -> ExecutionReport:
        self.requests_seen.append(request)
        if self.error is not None:
            raise self.error
        return self.report

    async def shutdown(self) -> None:
        return None


CODE_ANSWER = "Here is the solution:\n```python\ndef add(a, b):\n    return a + b\n```\n"
TEST_ANSWER = "```python\nfrom solution import *\n\ndef test_x():\n    assert add(1, 2) == 3\n```"


def _executor_roles() -> tuple[RoleSpec, ...]:
    return (
        RoleSpec(name="testgen", worker="gen", prompt="[testgen] {query}"),
        RoleSpec(name="proposal", worker="gen", prompt="[proposal] {query}"),
        RoleSpec(
            name="exec",
            worker="sandbox",
            role_type="executor",
            prompt="",
            depends_on=("testgen", "proposal"),
            executor=ExecutorRoleConfig(
                code_from=("proposal",),
                tests_from=("testgen",),
            ),
        ),
        RoleSpec(
            name="final",
            worker="gen",
            role_type="synthesizer",
            prompt="[final] {query} :: {exec}",
            depends_on=("proposal", "exec"),
        ),
    )


async def test_executor_step_and_report_feed_downstream_prompt():
    gen = ScriptedBackend([TEST_ANSWER, CODE_ANSWER, "final answer"])
    sandbox = FakeExecutor()
    conductor = Conductor(
        _executor_roles(),
        {"gen": gen},
        execution_workers={"sandbox": sandbox},
    )
    result = await conductor.run("task")

    assert result.final_text == "final answer"
    # Exactly one execution request, built from the extracted fenced blocks.
    assert len(sandbox.requests_seen) == 1
    files = sandbox.requests_seen[0].files
    assert "def add" in files["solution.py"]
    assert "def test_x" in files["test_solution.py"]
    # The structured report was interpolated into the downstream prompt.
    final_prompt = gen.requests_seen[-1].prompt
    assert '"status": "ok"' in final_prompt
    # One step consumed, zero tokens (m9 truthful usage).
    exec_events = [e for e in result.trace if e.node == "exec"]
    assert exec_events and exec_events[0].status == "success"
    assert exec_events[0].metadata["execution_status"] == "ok"
    assert exec_events[0].metadata["passed"] == 1
    assert exec_events[0].usage is None
    assert result.budget_state.steps_used == 4
    assert result.usage == (0, 0)


async def test_executor_skips_locally_without_step_on_prose():
    gen = ScriptedBackend(["NOT_APPLICABLE", "just prose, no code", "final answer"])
    sandbox = FakeExecutor()
    conductor = Conductor(
        _executor_roles(),
        {"gen": gen},
        execution_workers={"sandbox": sandbox},
    )
    result = await conductor.run("task")

    assert result.final_text == "final answer"
    assert sandbox.requests_seen == []
    exec_events = [e for e in result.trace if e.node == "exec"]
    assert exec_events and exec_events[0].status == "skipped"
    # The everyday non-coding path consumes zero executor steps.
    assert result.budget_state.steps_used == 3
    report = json.loads(result.outputs["exec"])
    assert report["status"] == "skipped"


async def test_executor_backend_failure_degrades_without_failing_request():
    gen = ScriptedBackend([TEST_ANSWER, CODE_ANSWER, "final answer"])
    sandbox = FakeExecutor(error=RuntimeError("sandbox down"))
    conductor = Conductor(
        _executor_roles(),
        {"gen": gen},
        execution_workers={"sandbox": sandbox},
    )
    result = await conductor.run("task")

    assert result.final_text == "final answer"
    report = json.loads(result.outputs["exec"])
    assert report["status"] == "unavailable"
    # The reserved step was reconciled, not leaked.
    assert result.budget_state.steps_reserved == 0


def _verifier_executor_roles() -> tuple[RoleSpec, ...]:
    return (
        RoleSpec(name="testgen", worker="gen", prompt="[testgen] {query}"),
        RoleSpec(name="draft", worker="gen", prompt="[draft] {query}",
                 depends_on=("testgen",)),
        RoleSpec(
            name="exec_draft",
            worker="sandbox",
            role_type="executor",
            prompt="",
            depends_on=("draft", "testgen"),
            executor=ExecutorRoleConfig(
                code_from=("draft",),
                tests_from=("testgen",),
            ),
        ),
        RoleSpec(
            name="verifier",
            worker="gen",
            role_type="verifier",
            verifies="draft",
            prompt="[verifier] {draft} :: {exec_draft}",
            depends_on=("draft", "exec_draft"),
        ),
        RoleSpec(
            name="final",
            worker="gen",
            role_type="synthesizer",
            prompt="[final] {draft} {verifier}",
            depends_on=("draft", "verifier"),
        ),
    )


async def test_inline_executor_reruns_on_each_refinement_attempt():
    draft_v1 = CODE_ANSWER
    draft_v2 = CODE_ANSWER.replace("a + b", "a + b  # fixed")
    gen = ScriptedBackend(
        [
            TEST_ANSWER,  # testgen
            draft_v1,  # draft attempt 0
            "FAIL: broken",  # verifier attempt 0
            draft_v2,  # draft attempt 1
            "PASS",  # verifier attempt 1
            "final answer",  # synthesizer
        ]
    )
    sandbox = FakeExecutor()
    conductor = Conductor(
        _verifier_executor_roles(),
        {"gen": gen},
        execution_workers={"sandbox": sandbox},
    )
    result = await conductor.run("task", budget=Budget(max_steps=16))

    assert result.final_text == "final answer"
    # The executor ran once per draft attempt against the FRESH draft, so the
    # verifier never judged stale execution evidence.
    assert len(sandbox.requests_seen) == 2
    assert "# fixed" not in sandbox.requests_seen[0].files["solution.py"]
    assert "# fixed" in sandbox.requests_seen[1].files["solution.py"]
    exec_attempts = [
        e.attempt for e in result.trace if e.node == "exec_draft" and e.status == "success"
    ]
    assert exec_attempts == [0, 1]


def test_executor_validation_rejects_bad_bindings():
    gen = ScriptedBackend([])
    sandbox = FakeExecutor()
    # Executor worker must be an execution worker, not a generation engine.
    with pytest.raises(ValueError, match="execution worker"):
        Conductor(
            _executor_roles(),
            {"gen": gen, "sandbox": gen},
        )
    # Executor inputs must reference existing generation roles.
    bad = (
        RoleSpec(name="a", worker="gen", prompt="{query}"),
        RoleSpec(
            name="exec",
            worker="sandbox",
            role_type="executor",
            prompt="",
            depends_on=("a",),
            executor=ExecutorRoleConfig(code_from=("a",), tests_from=("missing",)),
        ),
        RoleSpec(name="final", worker="gen", role_type="synthesizer",
                 prompt="{a}{exec}", depends_on=("a", "exec")),
    )
    with pytest.raises(ValueError, match="generation role"):
        Conductor(bad, {"gen": gen}, execution_workers={"sandbox": sandbox})


async def test_matrix_mode_reports_per_candidate_consensus():
    roles = (
        RoleSpec(name="testgen", worker="gen", prompt="[testgen] {query}"),
        RoleSpec(name="p1", worker="gen", prompt="[p1] {query}"),
        RoleSpec(name="p2", worker="gen", prompt="[p2] {query}"),
        RoleSpec(
            name="exec",
            worker="sandbox",
            role_type="executor",
            prompt="",
            depends_on=("testgen", "p1", "p2"),
            executor=ExecutorRoleConfig(
                code_from=("p1", "p2"),
                tests_from=("testgen",),
                mode="matrix",
            ),
        ),
        RoleSpec(name="final", worker="gen", role_type="synthesizer",
                 prompt="[final] {exec}", depends_on=("exec",)),
    )
    gen = ScriptedBackend([TEST_ANSWER, CODE_ANSWER, CODE_ANSWER, "final"])
    sandbox = FakeExecutor()
    conductor = Conductor(
        roles,
        {"gen": gen},
        execution_workers={"sandbox": sandbox},
    )
    result = await conductor.run("task")

    assert len(sandbox.requests_seen) == 2
    report = json.loads(result.outputs["exec"])
    assert set(report["candidates"]) == {"p1", "p2"}
    assert report["consensus"]["test_solution.py::test_x"] == {"passed": 2, "of": 2}
    exec_event = next(e for e in result.trace if e.node == "exec")
    assert exec_event.metadata["candidates"] == 2
    # A matrix stage is one orchestration step regardless of candidate count.
    assert result.budget_state.steps_used == 5


async def test_budget_skip_fallback_never_publishes_executor_output():
    """Issue #496: best-so-far falls back to a generation stage, never the
    executor's machine report, when the final unit is budget-refused."""

    gen = ScriptedBackend([TEST_ANSWER, CODE_ANSWER, "unused final"])
    sandbox = FakeExecutor()
    conductor = Conductor(
        _executor_roles(),
        {"gen": gen},
        execution_workers={"sandbox": sandbox},
    )
    # testgen + proposal consume both steps; exec is free; final is refused.
    result = await conductor.run("task", budget=Budget(max_steps=2))

    assert "final" not in result.outputs
    assert any(event.kind == "skipped:budget" for event in result.trace)
    assert result.final_text in {TEST_ANSWER, CODE_ANSWER}
    assert "UNTRUSTED" not in result.final_text
