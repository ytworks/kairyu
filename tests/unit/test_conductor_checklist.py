"""Checklist verifiers: System One reads in one request, curation of upstream
lists, and the guarantee report published with the final answer."""

import json

import pytest

from kairyu.engine.backend import GenerationRequest, GenerationResult
from kairyu.engine.systemone import SystemOneReply, SystemOneUnavailableError
from kairyu.orchestration.budget import Budget
from kairyu.orchestration.checklist import (
    ChecklistConfig,
    ChecklistQuestion,
    CurationConfig,
    CurationTarget,
    ItemSource,
    StateSection,
)
from kairyu.orchestration.conductor import Conductor, RoleSpec
from kairyu.orchestration.request import CONVERSATION_JSON_CLOSE, CONVERSATION_JSON_OPEN
from kairyu.outputs import CompletionOutput


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


def _contains_42(state, question):
    return 0.99 if "42" in str(state.get("answer", "")) else 0.1


def _answer_roles(**config_overrides) -> tuple[RoleSpec, ...]:
    config = ChecklistConfig(
        questions=(ChecklistQuestion(id="R1", proposition="states the answer 42"),),
        state=(StateSection("answer", "answer"),),
        subject="the answer",
        threshold=0.8,
        max_refinements=2,
        on_unavailable="publish_unverified",
        unverified_from="generator",
        **config_overrides,
    )
    return (
        RoleSpec(name="generator", worker="gen", prompt="[generator] {query}"),
        RoleSpec(
            name="answer",
            worker="gen",
            prompt="",
            depends_on=("generator",),
            seed_from="generator",
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


async def test_seeded_draft_that_passes_publishes_with_a_guarantee():
    backend = RoutedBackend({"generator": ["The answer is 42."]})
    judge = FakeSystemOne(_contains_42)
    conductor = Conductor(_answer_roles(), {"gen": backend}, decision_workers={"judge": judge})

    result = await conductor.run("What is six times seven?", budget=Budget(max_steps=12))

    assert result.final_text == "The answer is 42."
    report = result.verification.as_dict()
    assert report["guaranteed"] is True and report["reason"] is None
    assert [item["id"] for item in report["requirements"]] == ["R1"]
    # The draft is published as-is: no generation for the answer itself.
    assert [prompt.split("]")[0] for prompt in backend.prompts] == ["[generator"]
    # Jev request shape: a JSON state object and self-contained noul questions.
    (body,) = judge.bodies
    assert body["state"] == {"answer": "The answer is 42."}
    (question,) = body["questions"].values()
    assert question["type"] == "noul"
    assert question["instructions"]["requirement"] == "states the answer 42"


async def test_a_failed_item_is_repaired_and_judged_again():
    backend = RoutedBackend({"generator": ["The answer is 41."], "repair": ["The answer is 42."]})
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
        {"generator": ["It is 41."], "repair": ["It is 40.", "It is 43."]}
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
    backend = RoutedBackend({"generator": ["It is 41."], "repair": ["", "", "", ""]})

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


async def test_seed_of_the_final_unit_is_generated_under_the_caller_tool_contract():
    # Issue #617: the seeded answer publishes the generator's draft as-is, so
    # only a draft written with the caller's tools can carry a real tool call.
    tools = ({"type": "function", "function": {"name": "bash"}},)
    backend = RoutedBackend({"generator": ["The answer is 42."], "points": ["[]"]})
    roles = (
        *_answer_roles(),
        RoleSpec(name="points", worker="gen", prompt="[points] tools={tools}"),
    )
    conductor = Conductor(
        roles,
        {"gen": backend},
        decision_workers={"judge": FakeSystemOne(_contains_42)},
        final_tools=tools,
        final_tool_choice="auto",
    )

    await conductor.run("What is six times seven?", budget=Budget(max_steps=12))

    by_role = {str(r.prompt).split("]")[0].lstrip("["): r for r in backend.requests}
    assert by_role["generator"].tools == tools
    assert by_role["generator"].tool_choice == "auto"
    # An analysing role is not given the tools, but reads their definitions.
    assert by_role["points"].tools == ()
    assert '"name": "bash"' in str(by_role["points"].prompt)


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


async def test_two_point_lists_are_adopted_in_one_read_over_the_request():
    # Points from two parallel extractors are judged in one System One
    # request against the request alone (system + latest user message);
    # points judged unnecessary leave both lists before the answer reads them.
    turns = [
        {"role": "system", "content": "You are terse."},
        {"role": "user", "content": "old question"},
        {"role": "assistant", "content": "old answer"},
        {"role": "tool", "content": "x" * 200},
        {"role": "user", "content": "Name a primary colour."},
    ]
    backend = RoutedBackend(
        {
            "explicit": [_points("E", "names a colour", "cites a poem")],
            "implicit": [_points("I", "is one word", "uses French")],
            "history": ["The user asked an old question."],
            "answer": ["Red"],
        }
    )

    def necessity(state, question):
        return 0.1 if "poem" in question or "French" in question else 0.9

    judge = FakeSystemOne(necessity)
    roles = (
        RoleSpec(name="explicit", worker="gen", prompt="[explicit] {query}"),
        RoleSpec(name="implicit", worker="gen", prompt="[implicit] {query}"),
        RoleSpec(
            name="history",
            worker="gen",
            prompt="[history] {query}",
            depends_on=("explicit", "implicit"),
        ),
        RoleSpec(
            name="adopt",
            worker="judge",
            prompt="",
            role_type="verifier",
            verifies="history",
            depends_on=("history",),
            checklist=ChecklistConfig(
                questions=tuple(
                    ChecklistQuestion(
                        id="{item[id]}",
                        proposition="{item[point]}",
                        foreach=ItemSource(role=role, path="points"),
                        group="necessity",
                        threshold=0.0,
                        ask="Is this point necessary to answer the request?",
                        context={"point": "{item[point]}"},
                    )
                    for role in ("explicit", "implicit")
                ),
                state=(
                    StateSection("request", "request"),
                    StateSection("history", "history"),
                ),
                max_refinements=0,
                curate=CurationConfig(
                    targets=(
                        CurationTarget("explicit", "points"),
                        CurationTarget("implicit", "points"),
                    ),
                    drop_group="necessity",
                ),
            ),
        ),
        RoleSpec(
            name="answer",
            worker="gen",
            prompt="[answer] {explicit} {implicit}",
            depends_on=("history",),
        ),
    )
    conductor = Conductor(roles, {"gen": backend}, decision_workers={"judge": judge})

    await conductor.run(_chat_query(turns), budget=Budget(max_steps=8))

    (body,) = judge.bodies
    assert len(body["questions"]) == 4
    assert body["state"]["request"] == [turns[0], turns[-1]]
    assert body["state"]["history"] == "The user asked an old question."
    answer_prompt = next(p for p in backend.prompts if p.startswith("[answer]"))
    assert "names a colour" in answer_prompt and "is one word" in answer_prompt
    assert "poem" not in answer_prompt and "French" not in answer_prompt


async def test_a_long_conversation_is_bounded_so_the_checklist_is_judged():
    # Issue #617 GPU rerun: per-message cuts left a long agent conversation
    # above max_state_chars, so every checklist was unavailable.
    roles = _answer_roles(max_state_chars=4000)
    config = roles[2].checklist
    roles = (
        *roles[:2],
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
    backend = RoutedBackend({"generator": ["The answer is 42."]})
    judge = FakeSystemOne(_contains_42)
    conductor = Conductor(roles, {"gen": backend}, decision_workers={"judge": judge})

    result = await conductor.run(_chat_query(turns), budget=Budget(max_steps=12))

    assert result.verification.guaranteed is True
    state = judge.bodies[0]["state"]
    assert state["request_omitted_messages"] == len(turns) - len(state["request"])


async def test_unavailable_judge_publishes_the_draft_unverified():
    backend = RoutedBackend({"generator": ["The answer is 42."]})
    conductor = Conductor(
        _answer_roles(),
        {"gen": backend},
        decision_workers={"judge": FakeSystemOne(down=True)},
    )

    result = await conductor.run("What is six times seven?", budget=Budget(max_steps=12))

    assert result.final_unit_ok
    assert result.final_text == "The answer is 42."
    assert result.verification.as_dict()["guaranteed"] is False
    assert result.verification.reason == "judge_unavailable"


async def test_a_seeded_answer_keeps_its_seeds_finish_reason():
    class LengthBackend(RoutedBackend):
        async def generate(self, request):
            result = await super().generate(request)
            completion = result.completions[0]
            object.__setattr__(completion, "finish_reason", "length")
            return result

    backend = LengthBackend({"generator": ["The answer is 42"]})
    conductor = Conductor(
        _answer_roles(), {"gen": backend}, decision_workers={"judge": FakeSystemOne(_contains_42)}
    )

    result = await conductor.run("What is six times seven?", budget=Budget(max_steps=12))

    assert result.completions[0].finish_reason == "length"


def test_a_final_checklist_cannot_curate_what_it_publishes():
    # Review P1 (round 3): the guarantee must describe the published text.
    roles = (
        RoleSpec(name="draft", worker="gen", prompt="[draft] {query}"),
        RoleSpec(name="answer", worker="gen", prompt="[answer] {draft}", depends_on=("draft",)),
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
                curate=CurationConfig(
                    targets=(CurationTarget("answer", "items"),), drop_group="checklist"
                ),
            ),
        ),
    )
    with pytest.raises(ValueError, match="cannot curate"):
        Conductor(roles, {"gen": RoutedBackend({})}, decision_workers={"judge": FakeSystemOne()})


async def test_a_seed_republishes_the_curated_list_not_the_generated_one():
    # PR #618 review: curation rewrote the text but a later seed published
    # the stored completions, which still held the dropped item.
    backend = RoutedBackend({"draft": [_points("E", "keep me", "drop me")]})

    def necessity(state, question):
        return 0.1 if "drop me" in question else 0.9

    roles = (
        RoleSpec(name="draft", worker="gen", prompt="[draft] {query}"),
        RoleSpec(
            name="adopt",
            worker="judge",
            prompt="",
            role_type="verifier",
            verifies="draft",
            depends_on=("draft",),
            checklist=ChecklistConfig(
                questions=(
                    ChecklistQuestion(
                        id="{item[id]}",
                        proposition="{item[point]}",
                        foreach=ItemSource(role="draft", path="points"),
                        group="necessity",
                        threshold=0.0,
                    ),
                ),
                state=(StateSection("draft", "draft"),),
                max_refinements=0,
                curate=CurationConfig(
                    targets=(CurationTarget("draft", "points"),), drop_group="necessity"
                ),
            ),
        ),
        RoleSpec(
            name="answer",
            worker="gen",
            prompt="",
            depends_on=("draft",),
            seed_from="draft",
            refine_prompt="[repair] {previous}",
        ),
    )
    conductor = Conductor(
        roles, {"gen": backend}, decision_workers={"judge": FakeSystemOne(necessity)}
    )

    result = await conductor.run("q", budget=Budget(max_steps=8))

    assert "drop me" not in result.final_text
    assert result.completions[0].text == result.final_text
