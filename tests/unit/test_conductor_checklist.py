"""Checklist verifiers: System One reads, the acceptance read, curation of an
upstream list, and the guarantee report published with the final answer."""

import asyncio
import json
from dataclasses import replace

import pytest

from kairyu.engine.backend import GenerationRequest, GenerationResult
from kairyu.engine.systemone import SystemOneReply, SystemOneUnavailableError
from kairyu.orchestration.budget import Budget
from kairyu.orchestration.checklist import (
    AcceptanceConfig,
    ChecklistConfig,
    ChecklistQuestion,
    CurationConfig,
    ItemSource,
    StateSection,
)
from kairyu.orchestration.conductor import Conductor, RoleSamplingOverrides, RoleSpec
from kairyu.orchestration.request import CONVERSATION_JSON_CLOSE, CONVERSATION_JSON_OPEN
from kairyu.outputs import CompletionOutput
from kairyu.sampling_params import SamplingParams


class RoutedBackend:
    """Answers by the role tag at the start of the rendered prompt."""

    def __init__(self, replies: dict[str, list[str]]) -> None:
        self._replies = {tag: list(texts) for tag, texts in replies.items()}
        self.prompts: list[str] = []
        self.requests: list[GenerationRequest] = []

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        prompt = str(request.prompt)
        self.prompts.append(prompt)
        self.requests.append(request)
        tag = prompt.split("]", 1)[0].lstrip("[")
        text = self._replies[tag].pop(0)
        return GenerationResult(
            request_id=request.request_id,
            prompt=request.prompt,
            completions=(CompletionOutput(index=0, text=text, token_ids=()),),
        )

    async def stream(self, request):  # pragma: no cover - unused
        yield await self.generate(request)

    async def shutdown(self) -> None:
        return None


class FakeSystemOne:
    """P(yes) per question from a rule over the state and the question."""

    def __init__(self, rule=None, *, down: bool = False) -> None:
        self._rule = rule or (lambda state, question: 0.99)
        self._down = down
        self.bodies: list[dict] = []

    async def decide(self, body: dict) -> SystemOneReply:
        self.bodies.append(body)
        if self._down:
            raise SystemOneUnavailableError("ConnectError")
        answers = {
            key: {"noul": self._rule(body["state"], json.dumps(question["instructions"]))}
            for key, question in body["questions"].items()
        }
        return SystemOneReply(
            status=200,
            body=json.dumps({"answers": answers}).encode(),
            headers={},
            input_tokens=10,
            output_tokens=1,
        )


BASH = ({"type": "function", "function": {"name": "bash"}},)


def _contains_42(state, question):
    return 0.99 if "42" in str(state.get("answer", "")) else 0.1


def _answer_roles(**config_overrides) -> tuple[RoleSpec, ...]:
    config = ChecklistConfig(
        **{
            "questions": (ChecklistQuestion(id="R1", proposition="states the answer 42"),),
            "state": (StateSection("answer", "answer"),),
            "subject": "the answer",
            "threshold": 0.8,
            "max_refinements": 2,
            "on_unavailable": "publish_unverified",
            "unverified_from": "answer",
            **config_overrides,
        }
    )
    return (
        RoleSpec(
            name="answer",
            worker="gen",
            prompt="[answer] {query}",
            refine_prompt="[repair] {previous}\n{feedback}",
        ),
        RoleSpec(
            name="checklist",
            worker="judge",
            prompt="",
            role_type="verifier",
            verifies="answer",
            depends_on=("answer",),
            checklist=config,
        ),
    )


async def test_a_draft_that_passes_publishes_with_a_guarantee():
    backend = RoutedBackend({"answer": ["The answer is 42."]})
    judge = FakeSystemOne(_contains_42)
    conductor = Conductor(_answer_roles(), {"gen": backend}, decision_workers={"judge": judge})

    result = await conductor.run("What is six times seven?", budget=Budget(max_steps=12))

    assert result.final_text == "The answer is 42."
    report = result.verification.as_dict()
    assert report["guaranteed"] is True and report["reason"] is None
    assert [item["id"] for item in report["requirements"]] == ["R1"]
    assert [prompt.split("]")[0] for prompt in backend.prompts] == ["[answer"]
    # Jev request shape: a JSON state object and self-contained noul questions.
    (body,) = judge.bodies
    assert body["state"] == {"answer": "The answer is 42."}
    (question,) = body["questions"].values()
    assert question["type"] == "noul"
    assert question["instructions"]["requirement"] == "states the answer 42"


async def test_a_failed_item_is_repaired_and_judged_again():
    backend = RoutedBackend({"answer": ["The answer is 41."], "repair": ["The answer is 42."]})
    judge = FakeSystemOne(_contains_42)
    conductor = Conductor(_answer_roles(), {"gen": backend}, decision_workers={"judge": judge})

    result = await conductor.run("What is six times seven?", budget=Budget(max_steps=12))

    assert result.final_text == "The answer is 42."
    assert result.verification.guaranteed is True
    assert result.verification.attempts == 2
    repair_prompt = next(p for p in backend.prompts if p.startswith("[repair]"))
    assert "The answer is 41." in repair_prompt and "[R1] states the answer 42" in repair_prompt
    assert len(judge.bodies) == 2


async def test_exhausted_refinements_publish_the_last_attempt_unguaranteed():
    backend = RoutedBackend(
        {"answer": ["It is 41."], "repair": ["It is 40.", "It is 43."]}
    )
    conductor = Conductor(
        _answer_roles(), {"gen": backend}, decision_workers={"judge": FakeSystemOne(_contains_42)}
    )

    result = await conductor.run("What is six times seven?", budget=Budget(max_steps=16))

    assert result.final_text == "It is 43."
    report = result.verification.as_dict()
    assert report["guaranteed"] is False and report["reason"] == "refinement_limit"
    assert report["attempts"] == 3


async def test_an_empty_repair_never_replaces_a_draft_even_when_it_passes_vacuously():
    # Issue #617: an empty repair can pass every item vacuously; accepting or
    # publishing it failed the request although the non-empty draft existed.
    backend = RoutedBackend({"answer": ["It is 41."], "repair": ["", "", "", ""]})

    def judge_rule(state, question):
        return 0.99 if state["answer"] == "" else 0.1

    conductor = Conductor(
        _answer_roles(), {"gen": backend}, decision_workers={"judge": FakeSystemOne(judge_rule)}
    )

    result = await conductor.run("What is six times seven?", budget=Budget(max_steps=16))

    assert result.final_unit_ok
    assert result.final_text == "It is 41."
    report = result.verification.as_dict()
    assert report["guaranteed"] is False and report["reason"] == "refinement_limit"


async def test_an_empty_intermediate_output_stays_governed_by_its_own_checklist():
    # PR #618 review: only the published answer must have text; an extractor
    # whose checklist passes on an empty result is not refined.
    roles = (
        RoleSpec(name="quotes", worker="gen", prompt="[quotes] {query}"),
        RoleSpec(
            name="quotes_check",
            worker="judge",
            prompt="",
            role_type="verifier",
            verifies="quotes",
            depends_on=("quotes",),
            checklist=ChecklistConfig(
                questions=(ChecklistQuestion(id="Q1", proposition="no invented quote"),),
                state=(StateSection("quotes", "quotes"),),
                max_refinements=2,
            ),
        ),
        RoleSpec(
            name="answer",
            worker="gen",
            prompt="[answer] {quotes}",
            depends_on=("quotes", "quotes_check"),
        ),
    )
    backend = RoutedBackend({"quotes": [""], "answer": ["There are no matching citations."]})
    conductor = Conductor(roles, {"gen": backend}, decision_workers={"judge": FakeSystemOne()})

    result = await conductor.run("Cite the sources.", budget=Budget(max_steps=3))

    assert result.final_text == "There are no matching citations."
    assert [p.split("]")[0] for p in backend.prompts] == ["[quotes", "[answer"]


def _chat_query(turns: list[dict]) -> str:
    return f"{CONVERSATION_JSON_OPEN}{json.dumps(turns)}{CONVERSATION_JSON_CLOSE}"


def _points(prefix: str, *texts: str) -> str:
    return json.dumps(
        {"points": [{"id": f"{prefix}{n}", "point": text} for n, text in enumerate(texts, 1)]}
    )


async def test_a_point_list_is_adopted_over_the_request_before_the_answer_reads_it():
    # The points are judged against the request alone (system + latest user
    # message); points judged unnecessary leave the list the answer reads. An
    # analysing role reads the caller's tool definitions but is not given them.
    turns = [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer"},
        {"role": "tool", "content": "x" * 200},
        {"role": "user", "content": "Name a primary colour."},
    ]
    backend = RoutedBackend(
        {
            "points": [_points("E", "names a colour", "cites a poem")],
            "answer": ["Red"],
        }
    )

    def necessity(state, question):
        return 0.1 if "poem" in question else 0.9

    judge = FakeSystemOne(necessity)
    roles = (
        RoleSpec(name="points", worker="gen", prompt="[points] {query} tools={tools}"),
        RoleSpec(
            name="adopt",
            worker="judge",
            prompt="",
            role_type="verifier",
            verifies="points",
            depends_on=("points",),
            checklist=ChecklistConfig(
                questions=(
                    ChecklistQuestion(
                        id="{item[id]}",
                        proposition="{item[point]}",
                        foreach=ItemSource(role="points", path="points"),
                        group="necessity",
                        threshold=0.0,
                        ask="Is this point necessary to answer the request?",
                        context={"point": "{item[point]}"},
                    ),
                ),
                state=(StateSection("request", "request"),),
                max_refinements=0,
                curate=CurationConfig(items_path="points", drop_group="necessity"),
            ),
        ),
        RoleSpec(
            name="answer",
            worker="gen",
            prompt="[answer] {points}",
            depends_on=("points", "adopt"),
        ),
    )
    conductor = Conductor(
        roles, {"gen": backend}, decision_workers={"judge": judge}, final_tools=BASH
    )

    await conductor.run(_chat_query(turns), budget=Budget(max_steps=8))

    (body,) = judge.bodies
    assert len(body["questions"]) == 2
    assert body["state"]["request"] == [turns[0], turns[-1]]
    answer_prompt = next(p for p in backend.prompts if p.startswith("[answer]"))
    assert "names a colour" in answer_prompt and "poem" not in answer_prompt
    points = next(r for r in backend.requests if str(r.prompt).startswith("[points]"))
    assert points.tools == () and '"name": "bash"' in str(points.prompt)


async def test_a_long_conversation_is_bounded_so_the_checklist_is_judged():
    # Issue #617 GPU rerun: per-message cuts left a long agent conversation
    # above max_state_chars, so every checklist was unavailable.
    roles = _answer_roles(max_state_chars=4000)
    config = roles[1].checklist
    roles = (
        *roles[:1],
        RoleSpec(
            name="checklist",
            worker="judge",
            prompt="",
            role_type="verifier",
            verifies="answer",
            depends_on=("answer",),
            checklist=ChecklistConfig(
                questions=config.questions,
                state=(
                    StateSection("request", "query", max_chars=500, max_total_chars=2000),
                    StateSection("answer", "answer"),
                ),
                threshold=config.threshold,
                max_state_chars=4000,
            ),
        ),
    )
    turns = [{"role": "user", "content": "What is six times seven?"}]
    turns += [{"role": "tool", "content": f"log {i} " + "x" * 400} for i in range(30)]
    backend = RoutedBackend({"answer": ["The answer is 42."]})
    judge = FakeSystemOne(_contains_42)
    conductor = Conductor(roles, {"gen": backend}, decision_workers={"judge": judge})

    result = await conductor.run(_chat_query(turns), budget=Budget(max_steps=12))

    assert result.verification.guaranteed is True
    state = judge.bodies[0]["state"]
    assert state["request_omitted_messages"] == len(turns) - len(state["request"])


def test_a_final_checklist_cannot_curate_what_it_publishes():
    # Review P1 (round 3): the guarantee must describe the published text.
    roles = (
        RoleSpec(name="answer", worker="gen", prompt="[answer] {query}"),
        RoleSpec(
            name="check",
            worker="judge",
            prompt="",
            role_type="verifier",
            verifies="answer",
            depends_on=("answer",),
            checklist=ChecklistConfig(
                questions=(ChecklistQuestion(id="R1", proposition="ok"),),
                state=(StateSection("answer", "answer"),),
                curate=CurationConfig(items_path="items", drop_group="checklist"),
            ),
        ),
    )
    with pytest.raises(ValueError, match="cannot curate"):
        Conductor(roles, {"gen": RoutedBackend({})}, decision_workers={"judge": FakeSystemOne()})


def _accepting_roles(threshold: float = 0.5) -> tuple[RoleSpec, ...]:
    roles = _answer_roles()
    config = roles[1].checklist
    return (
        *roles[:1],
        RoleSpec(
            name="checklist",
            worker="judge",
            prompt="",
            role_type="verifier",
            verifies="answer",
            depends_on=("answer",),
            checklist=ChecklistConfig(
                questions=config.questions,
                state=config.state,
                threshold=config.threshold,
                max_refinements=2,
                on_unavailable=config.on_unavailable,
                unverified_from=config.unverified_from,
                acceptance=AcceptanceConfig(
                    ask="May this answer be adopted?",
                    state=(StateSection("answer", "answer"),),
                    threshold=threshold,
                ),
            ),
        ),
    )


async def test_the_acceptance_read_decides_the_guarantee_over_the_point_results():
    # VCO-D15 (wave 4): Jev reads the point results and the answer and decides
    # whether the answer may be adopted; a missed point alone does not decide.
    backend = RoutedBackend({"answer": ["It is 41."]})

    def rule(state, question):
        if "adopted" in question:
            assert state["checklist"][0]["passed"] is False
            return 0.9
        return 0.1

    judge = FakeSystemOne(rule)
    conductor = Conductor(_accepting_roles(), {"gen": backend}, decision_workers={"judge": judge})

    result = await conductor.run("What is six times seven?", budget=Budget(max_steps=12))

    coverage, acceptance = judge.bodies
    assert set(acceptance["state"]) == {"answer", "checklist"}
    assert acceptance["state"]["checklist"][0]["point"] == "states the answer 42"
    report = result.verification.as_dict()
    assert report["guaranteed"] is True and report["acceptance"] == pytest.approx(0.9)
    assert report["requirements"][0]["passed"] is False


async def test_a_rejected_answer_is_repaired_on_its_missed_points_and_judged_again():
    backend = RoutedBackend({"answer": ["It is 41."], "repair": ["The answer is 42."]})

    def rule(state, question):
        if "adopted" in question:
            return 0.9 if all(item["passed"] for item in state["checklist"]) else 0.1
        return _contains_42(state, question)

    judge = FakeSystemOne(rule)
    conductor = Conductor(_accepting_roles(), {"gen": backend}, decision_workers={"judge": judge})

    result = await conductor.run("What is six times seven?", budget=Budget(max_steps=12))

    assert result.final_text == "The answer is 42."
    assert result.verification.guaranteed is True and result.verification.attempts == 2
    repair = next(p for p in backend.prompts if p.startswith("[repair]"))
    assert "It is 41." in repair and "[R1] states the answer 42" in repair
    assert len(judge.bodies) == 4


async def test_a_rejection_with_every_point_met_is_published_unverified_without_repair():
    # DeepSWE (PR #618): a rejection that names no unmet point gave the repair
    # nothing to fix, and the repair rewrote a sound reply.
    backend = RoutedBackend({"answer": ["The answer is 42."]})

    def rule(state, question):
        return 0.1 if "adopted" in question else 0.99

    judge = FakeSystemOne(rule)
    conductor = Conductor(_accepting_roles(), {"gen": backend}, decision_workers={"judge": judge})

    result = await conductor.run("What is six times seven?", budget=Budget(max_steps=12))

    assert result.final_text == "The answer is 42."
    assert [p.split("]")[0] for p in backend.prompts] == ["[answer"]
    report = result.verification.as_dict()
    assert report["guaranteed"] is False and report["reason"] == "not_accepted"


async def test_an_empty_repair_is_repaired_again_even_when_every_point_is_met():
    # Issue #617 with the acceptance read: an empty reply meets every point
    # vacuously; whether or not it is accepted, it is repaired, never kept
    # (the first empty reply is retried as an empty output, the second judged).
    for acceptance in (0.1, 0.9):
        backend = RoutedBackend({"answer": ["It is 41."], "repair": ["", "", "The answer is 42."]})

        def rule(state, question, acceptance=acceptance):
            empty = state["answer"] == ""
            if "adopted" in question:
                return acceptance if empty else _contains_42(state, question)
            return 0.99 if empty else _contains_42(state, question)

        conductor = Conductor(
            _accepting_roles(), {"gen": backend}, decision_workers={"judge": FakeSystemOne(rule)}
        )

        result = await conductor.run("What is six times seven?", budget=Budget(max_steps=12))

        assert result.final_text == "The answer is 42.", acceptance
        assert result.verification.guaranteed is True


async def test_a_failed_read_cancels_its_siblings_and_keeps_their_usage():
    from kairyu.orchestration.checklist import ChecklistUnavailable, judge

    hanging = asyncio.Event()
    cancelled = []

    class SplitJev:
        async def decide(self, body):
            (key,) = body["questions"]
            if key == "q0":
                return SystemOneReply(status=529, body=b"{}", headers={})
            if key == "q1":
                return SystemOneReply(
                    status=200,
                    body=json.dumps({"answers": {"q1": {"noul": 0.9}}}).encode(),
                    headers={},
                    input_tokens=7,
                )
            try:
                await hanging.wait()
            except asyncio.CancelledError:
                cancelled.append(key)
                raise

    config = ChecklistConfig(
        questions=tuple(ChecklistQuestion(id=f"R{n}", proposition=f"p{n}") for n in range(3)),
        state=(StateSection("answer", "answer"),),
        max_questions_per_call=1,
    )

    with pytest.raises(ChecklistUnavailable) as raised:
        await judge(config, SplitJev(), {"answer": "x"}, "q")

    # No read outlives the decision, and the completed read is still billed.
    assert cancelled == ["q2"]
    assert raised.value.usage == (7, 0)


class _AcceptanceDown(FakeSystemOne):
    async def decide(self, body: dict) -> SystemOneReply:
        self._down = bool(self.bodies)
        return await super().decide(body)


async def test_an_unavailable_acceptance_read_still_spends_the_coverage_read():
    usages, steps = [], []
    for judge in (FakeSystemOne(down=True), _AcceptanceDown()):
        backend = RoutedBackend({"answer": ["It is 42."]})
        conductor = Conductor(
            _accepting_roles(), {"gen": backend}, decision_workers={"judge": judge}
        )
        result = await conductor.run("What is six times seven?", budget=Budget(max_steps=12))
        # An unavailable judge publishes the draft, unverified.
        assert result.final_unit_ok and result.final_text == "It is 42."
        assert result.verification.reason == "judge_unavailable"
        assert result.verification.guaranteed is False
        usages.append(result.usage)
        steps.append(result.budget_state.steps_used)

    assert (usages[1][0] - usages[0][0], usages[1][1] - usages[0][1]) == (10, 1)
    assert steps[1] - steps[0] == 1


class _AcceptanceHangs(FakeSystemOne):
    def __init__(self) -> None:
        super().__init__()
        self.waiting = asyncio.Event()

    async def decide(self, body: dict) -> SystemOneReply:
        if self.bodies:
            self.waiting.set()
            await asyncio.Event().wait()
        return await super().decide(body)


async def test_a_cancelled_acceptance_read_keeps_the_coverage_usage_observed():
    # Disconnect metering reads the usage observer, not the result.
    observed = []
    judge = _AcceptanceHangs()
    conductor = Conductor(
        _accepting_roles(),
        {"gen": RoutedBackend({"answer": ["It is 42."]})},
        decision_workers={"judge": judge},
        usage_observer=observed.append,
    )
    task = asyncio.create_task(conductor.run("What is six?", budget=Budget(max_steps=12)))
    await judge.waiting.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # The generator reports no usage; the coverage read billed (10, 1).
    assert observed and (observed[-1].prompt_tokens, observed[-1].completion_tokens) == (10, 1)


async def test_the_acceptance_read_needs_its_own_budget_step():
    backend = RoutedBackend({"answer": ["It is 42."]})
    judge = FakeSystemOne()
    conductor = Conductor(_accepting_roles(), {"gen": backend}, decision_workers={"judge": judge})

    result = await conductor.run("What is six times seven?", budget=Budget(max_steps=2))

    assert judge.bodies == [] and result.verification.reason == "budget"


async def test_a_committed_head_is_a_public_answer_when_the_continuation_is_empty():
    # The public answer is the committed head plus the continuation; an empty
    # continuation after a complete head is not an empty answer.
    backend = RoutedBackend({"head": ["42"], "answer": ["", ""]})
    roles = (
        RoleSpec(
            name="head",
            worker="gen",
            role_type="head",
            prompt="[head] {query}",
            sampling=RoleSamplingOverrides(max_tokens=8),
        ),
        RoleSpec(name="answer", worker="gen", prompt="[answer] {head}", depends_on=("head",)),
        RoleSpec(
            name="checklist",
            worker="judge",
            prompt="",
            role_type="verifier",
            verifies="answer",
            depends_on=("answer",),
            checklist=ChecklistConfig(
                questions=(ChecklistQuestion(id="R1", proposition="states 42"),),
                state=(StateSection("head", "head"), StateSection("answer", "answer")),
                max_refinements=1,
            ),
        ),
    )
    conductor = Conductor(
        roles,
        {"gen": backend},
        decision_workers={"judge": FakeSystemOne()},
        final_sampling_params=SamplingParams(max_tokens=64),
    )

    result = await conductor.run("What is six times seven?", budget=Budget(max_steps=8))

    assert result.final_text == "42"
    assert result.verification.guaranteed is True and result.verification.attempts == 1


async def test_a_failed_checklist_target_leaves_the_run_unguaranteed():
    # Codex review: a failed `history` skipped its adoption checklist, yet
    # the final checklist still guaranteed the answer.
    backend = RoutedBackend({"answer": ["It is 42."]})  # no "history" reply: it fails
    answer, *rest = _answer_roles()
    roles = (
        RoleSpec(name="history", worker="gen", prompt="[history] {query}"),
        RoleSpec(
            name="adopt",
            worker="judge",
            prompt="",
            role_type="verifier",
            verifies="history",
            depends_on=("history",),
            checklist=ChecklistConfig(
                questions=(ChecklistQuestion(id="N1", proposition="is needed"),),
                state=(StateSection("history", "history"),),
                max_refinements=0,
                on_unavailable="publish_unverified",
            ),
        ),
        replace(answer, depends_on=("history",)),
        *rest,
    )
    judge = FakeSystemOne()
    conductor = Conductor(roles, {"gen": backend}, decision_workers={"judge": judge})

    result = await conductor.run("What is six times seven?", budget=Budget(max_steps=12))

    assert result.final_text == "It is 42."
    assert result.verification.guaranteed is False
    assert result.verification.reason == "checklist_unavailable"