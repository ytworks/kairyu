"""Checklist verifiers: System One probabilities over requirement items.

A checklist verifier judges its target attempt without a generation call:

1. Its questions (static, or one per item of an upstream role's JSON list)
   go to a System One (Jev wire API) decision backend together, in one
   request, as ``noul`` reads over a JSON state; each answer is a
   probability.
2. Each checklist item gets a probability ``p`` of being satisfied
   (questions sharing an item id aggregate by minimum) and passes when
   ``p >= threshold``. PASS needs every item to pass.
3. A curation may then drop items whose probability is low from upstream
   JSON lists, so downstream roles read the kept items only.

The verdict text is the existing verifier contract (first line PASS/FAIL,
then one feedback line per failing item), so the Conductor's refine loop is
unchanged. Policy — which questions exist, their wording, the state, the
threshold and the curation — is configuration.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from string import Formatter
from typing import Protocol

from kairyu.orchestration.request import (
    MIN_CONVERSATION_CHARS,
    bounded_conversation,
    conversation_messages,
)
from kairyu.tool_call_markup import generic_tool_calls, tool_selection

# A JSON value as produced by json.loads.
JSONValue = object


class DecisionBackend(Protocol):
    """The System One surface a checklist needs (HTTPSystemOneBackend)."""

    async def decide(self, body: dict): ...


class ChecklistUnavailable(Exception):
    """The checklist could not be judged (backend down, overloaded, malformed)."""

    def __init__(self, reason: str, detail: str = "", usage: tuple[int, int] = (0, 0)) -> None:
        super().__init__(detail or reason)
        self.reason = reason
        self.detail = detail
        # Tokens the reads that did complete still billed.
        self.usage = usage


def parse_json_output(text: str) -> JSONValue:
    """Parse a role's JSON output, tolerating one surrounding code fence."""

    stripped = text.strip()
    fence = re.fullmatch(r"```[A-Za-z0-9_-]*\s*\n(.*)\n```", stripped, re.DOTALL)
    if fence is not None:
        stripped = fence.group(1).strip()
    return json.loads(stripped)


def json_path(value: JSONValue, path: str) -> JSONValue:
    """Follow a dotted key path through JSON objects (empty path = value)."""

    current = value
    for key in filter(None, path.split(".")):
        if not isinstance(current, Mapping) or key not in current:
            raise KeyError(path)
        current = current[key]
    return current


class TemplateError(ValueError):
    """A checklist template cannot be rendered for an item."""


@dataclass(frozen=True)
class ItemSource:
    """Items taken from a role's JSON output.

    ``path`` is a dotted key path to a list of objects; ``where`` keeps only
    objects whose keys equal the given values.
    """

    role: str
    path: str = ""
    where: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "where", dict(self.where))


@dataclass(frozen=True)
class ChecklistQuestion:
    """One System One ``noul`` question, static or per selected item.

    By default the question is built from the requirement: Jev is asked
    whether the checklist ``subject`` satisfies ``proposition``, with yes/no
    criteria for full versus partial or missing satisfaction. ``ask`` (with
    optional ``criteria_true`` / ``criteria_false``) replaces that with an
    explicit yes/no question. ``context`` adds item-specific fields to the
    question object, so the question stays self-contained without placing
    item text in the state.
    """

    id: str
    proposition: str
    foreach: ItemSource | None = None
    threshold: float | None = None
    group: str = "checklist"
    subject: str = ""
    ask: str = ""
    criteria_true: str = ""
    criteria_false: str = ""
    context: Mapping[str, str] = field(default_factory=dict)
    # Report-only labels rendered per item (for example the item's origin);
    # they never reach System One.
    tags: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "context", dict(self.context))
        object.__setattr__(self, "tags", dict(self.tags))
        if self.threshold is not None and not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f"question {self.id!r}: threshold must be in [0, 1]")
        if (self.criteria_true or self.criteria_false) and not self.ask:
            raise ValueError(f"question {self.id!r}: criteria need an explicit ask")
        if {"question", "requirement"} & set(self.context):
            raise ValueError(f"question {self.id!r}: context cannot redefine question/requirement")
        for name in ("id", "proposition", "subject", "ask", "criteria_true", "criteria_false"):
            _validate_template(f"question {self.id!r} {name}", getattr(self, name))
        for key, template in self.context.items():
            _validate_template(f"question {self.id!r} context {key}", template)


# State sources read from the request rather than from a role output.
_REQUEST_SOURCES = frozenset({"query", "request"})
# The caller's tool definitions ("none" without tools).
_TOOLS_SOURCE = "tools"


@dataclass(frozen=True)
class StateSection:
    """One field of the System One state object.

    ``source`` is ``query`` (the whole request: its role-tagged messages when
    the query is Kairyu's chat transcript), ``request`` (only the system and
    developer messages plus the latest user message, verbatim), ``tools``
    (the caller's tool definitions, or "none") or a role name (its output,
    embedded as a JSON value when it parses as JSON; tool-call markup becomes
    ``{text, tool_calls}``). A
    text value longer than ``max_chars`` is cut with an explicit marker.
    ``max_total_chars`` bounds the messages of a ``query`` or ``request``
    section (``bounded_conversation``); the omitted middle is counted in
    ``<key>_omitted_messages``.
    """

    key: str
    source: str
    max_chars: int | None = None
    max_total_chars: int | None = None

    def __post_init__(self) -> None:
        if not self.key or not self.source:
            raise ValueError("a state section needs a key and a source")
        if self.max_chars is not None and self.max_chars < 1:
            raise ValueError(f"state section {self.key!r}: max_chars must be positive")
        if self.max_total_chars is not None:
            if self.source not in _REQUEST_SOURCES:
                raise ValueError(
                    f"state section {self.key!r}: max_total_chars bounds only a "
                    "query or request section"
                )
            if self.max_total_chars < MIN_CONVERSATION_CHARS:
                raise ValueError(
                    f"state section {self.key!r}: max_total_chars must be at least "
                    f"{MIN_CONVERSATION_CHARS}"
                )


@dataclass(frozen=True)
class CurationTarget:
    """An upstream JSON list a curation edits: ``role``'s list at ``items_path``."""

    role: str
    items_path: str


@dataclass(frozen=True)
class CurationConfig:
    """Drop items from upstream JSON lists after the verdict (no model call).

    Items of ``drop_group`` whose probability is below ``drop_below`` are
    removed from every target list, matched by ``id_key``; the edited lists
    replace the targets' outputs, so downstream roles read the kept items.
    """

    targets: tuple[CurationTarget, ...]
    drop_group: str
    drop_below: float = 0.5
    id_key: str = "id"

    def __post_init__(self) -> None:
        object.__setattr__(self, "targets", tuple(self.targets))
        if not self.targets:
            raise ValueError("a curation needs at least one target list")
        if not self.drop_group:
            raise ValueError("a curation needs a drop_group")
        if not 0.0 <= self.drop_below <= 1.0:
            raise ValueError("curation drop_below must be in [0, 1]")


@dataclass(frozen=True)
class AcceptanceConfig:
    """A final System One read on whether the target may be adopted as is.

    After the per-item read, one ``noul`` question (``ask`` with optional
    criteria) is asked over ``state`` plus the item results under
    ``results_key`` (each item's id, proposition, p and pass). The verdict
    passes when its probability reaches ``threshold``; the item results stay
    in the report, and failing items still go to a repair as feedback.
    """

    ask: str
    state: tuple[StateSection, ...]
    criteria_true: str = ""
    criteria_false: str = ""
    threshold: float = 0.5
    results_key: str = "checklist"

    def __post_init__(self) -> None:
        object.__setattr__(self, "state", tuple(self.state))
        if not self.ask:
            raise ValueError("an acceptance read needs an ask")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("acceptance threshold must be in [0, 1]")
        keys = [section.key for section in self.state]
        if self.results_key in keys or len(set(keys)) != len(keys):
            raise ValueError("acceptance state keys must be unique and differ from results_key")
        for name in ("ask", "criteria_true", "criteria_false"):
            _validate_template(f"acceptance {name}", getattr(self, name))


@dataclass(frozen=True)
class ChecklistConfig:
    questions: tuple[ChecklistQuestion, ...]
    # System One state: a JSON object with one field per section.
    state: tuple[StateSection, ...]
    # What the requirement questions are about (for example "the answer").
    subject: str = "the state"
    threshold: float = 0.5
    samples: int | None = None
    think: int | None = None
    steps: int | None = None
    # Every question of a verdict goes in one System One request.
    max_questions: int = 256
    max_state_chars: int | None = None
    feedback_header: str = "The following requirements are not met:"
    feedback_item: str = "- [{id}] {proposition} (p={p})"
    max_refinements: int | None = None
    # "error": an unjudgeable checklist fails the unit (verifier contract).
    # "publish_unverified": publish ``unverified_from`` (or the current
    # attempt) and report the result as unverified.
    on_unavailable: str = "error"
    unverified_from: str = ""
    curate: CurationConfig | None = None
    acceptance: AcceptanceConfig | None = None
    # "unavailable": a verdict with no item vouches for nothing. "pass": a
    # curation read whose lists are empty has nothing to drop and passes
    # (curation checklists only; a guarantee still needs judged items).
    on_empty: str = "unavailable"

    def __post_init__(self) -> None:
        object.__setattr__(self, "questions", tuple(self.questions))
        object.__setattr__(self, "state", tuple(self.state))
        if not self.questions:
            raise ValueError("a checklist needs at least one question")
        if not self.state:
            raise ValueError("checklist questions need System One state sections")
        keys = [section.key for section in self.state]
        if len(set(keys)) != len(keys):
            raise ValueError("checklist state section keys must be unique")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("checklist threshold must be in [0, 1]")
        if self.on_unavailable not in {"error", "publish_unverified"}:
            raise ValueError("on_unavailable must be 'error' or 'publish_unverified'")
        if self.on_empty not in {"unavailable", "pass"}:
            raise ValueError("on_empty must be 'unavailable' or 'pass'")
        if self.on_empty == "pass" and self.curate is None:
            raise ValueError("on_empty: pass is for curation checklists (curate) only")
        if self.unverified_from and self.on_unavailable != "publish_unverified":
            raise ValueError("unverified_from requires on_unavailable: publish_unverified")
        if self.max_refinements is not None and self.max_refinements < 0:
            raise ValueError("max_refinements must be >= 0")
        if self.max_questions < 1:
            raise ValueError("max_questions must be >= 1")
        for name in ("subject", "feedback_header"):
            _validate_template(f"checklist {name}", getattr(self, name))

    def referenced_roles(self) -> set[str]:
        """Roles whose outputs this checklist reads or edits (for DAG validation)."""

        names = {question.foreach.role for question in self.questions if question.foreach}
        sections = (*self.state, *(self.acceptance.state if self.acceptance else ()))
        names.update(
            section.source
            for section in sections
            if section.source not in _REQUEST_SOURCES and section.source != _TOOLS_SOURCE
        )
        if self.curate is not None:
            names.update(target.role for target in self.curate.targets)
        return names


@dataclass(frozen=True)
class ChecklistItem:
    """One judged requirement as reported to the caller."""

    id: str
    proposition: str
    group: str
    p: float
    passed: bool
    tags: Mapping[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "proposition": self.proposition,
            "group": self.group,
            "p": round(self.p, 6),
            "passed": self.passed,
            **({"tags": dict(self.tags)} if self.tags else {}),
        }


@dataclass(frozen=True)
class ChecklistVerdict:
    passed: bool
    items: tuple[ChecklistItem, ...]
    text: str
    usage: tuple[int, int] = (0, 0)
    reads: int = 0
    # The acceptance read's probability; None without one.
    acceptance: float | None = None


@dataclass(frozen=True)
class VerificationReport:
    """The public verification outcome of the final unit."""

    guaranteed: bool
    reason: str | None
    threshold: float | None
    items: tuple[ChecklistItem, ...] = ()
    attempts: int = 0
    acceptance: float | None = None

    def as_markdown(self) -> str:
        """A readable summary for clients that only show reasoning text."""

        lines = [
            "### Verification",
            "",
            f"- Guaranteed: {'yes' if self.guaranteed else 'no'}",
        ]
        if self.reason:
            lines.append(f"- Reason: {self.reason}")
        if self.threshold is not None:
            lines.append(f"- Threshold: p >= {self.threshold}")
        lines.append(f"- Attempts: {self.attempts}")
        if self.acceptance is not None:
            lines.append(f"- Accepted as the reply: p={self.acceptance:.3f}")
        if self.items:
            lines.append("")
        for item in self.items:
            mark = "PASS" if item.passed else "FAIL"
            lines.append(f"- [{item.id}] {mark} p={item.p:.3f} — {item.proposition}")
        return "\n".join(lines)

    def as_dict(self) -> dict[str, object]:
        return {
            "guaranteed": self.guaranteed,
            "reason": self.reason,
            "threshold": self.threshold,
            "attempts": self.attempts,
            **({"acceptance": round(self.acceptance, 6)} if self.acceptance is not None else {}),
            "requirements": [item.as_dict() for item in self.items],
        }


class _TemplateValues(dict):
    def __missing__(self, key: str) -> str:
        return ""


def _validate_template(label: str, template: str) -> None:
    try:
        list(Formatter().parse(template))
    except ValueError as error:
        raise ValueError(f"{label} is not a valid template ({error})") from error


def _display(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _display(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return ", ".join(str(_display(inner)) for inner in value)
    return value


def render(template: str, values: Mapping[str, object]) -> str:
    """Render a checklist template; lists read as comma-separated text."""

    prepared = _TemplateValues({key: _display(value) for key, value in values.items()})
    try:
        return template.format_map(prepared)
    except (KeyError, IndexError, AttributeError, TypeError, ValueError) as error:
        raise TemplateError(f"template cannot be rendered: {error}") from error


def _source_items(source: ItemSource, outputs: Mapping[str, str]) -> list[Mapping[str, object]]:
    raw = outputs.get(source.role)
    if raw is None:
        raise ChecklistUnavailable("checklist_unavailable", f"{source.role!r} has no output")
    try:
        items = json_path(parse_json_output(raw), source.path)
    except (ValueError, KeyError) as error:
        raise ChecklistUnavailable(
            "checklist_unavailable",
            f"{source.role!r} output has no JSON list at {source.path!r}",
        ) from error
    if not isinstance(items, list):
        raise ChecklistUnavailable(
            "checklist_unavailable", f"{source.role!r} output at {source.path!r} is not a list"
        )
    return [
        item
        for item in items
        if isinstance(item, Mapping)
        and all(item.get(key) == value for key, value in source.where.items())
    ]


def _bindings(source: ItemSource | None, outputs: Mapping[str, str]) -> list[dict[str, object]]:
    if source is None:
        return [{}]
    return [{"item": item} for item in _source_items(source, outputs)]


@dataclass(frozen=True)
class _PendingQuestion:
    item_id: str
    proposition: str
    group: str
    # The Jev ``noul`` question object (instructions and criteria).
    payload: Mapping[str, object]
    threshold: float
    tags: Mapping[str, str] = field(default_factory=dict)


def requirement_question(
    subject: str,
    proposition: str,
    context: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """A Jev ``noul`` question asking whether ``subject`` meets a requirement."""

    return {
        "type": "noul",
        "instructions": {
            "question": f"Does {subject} satisfy the requirement below?",
            "requirement": proposition,
            **dict(context or {}),
        },
        "criteria": {
            "true": f"{subject} fully satisfies the requirement",
            "false": f"{subject} does not satisfy the requirement, or satisfies it only partly",
        },
    }


def _question_payload(
    config: ChecklistConfig,
    question: ChecklistQuestion,
    binding: Mapping[str, object],
    proposition: str,
) -> dict[str, object]:
    context = {key: render(value, binding) for key, value in question.context.items()}
    if not question.ask:
        subject = render(question.subject, binding) if question.subject else config.subject
        return requirement_question(subject, proposition, context)
    payload: dict[str, object] = {
        "type": "noul",
        "instructions": (
            {"question": render(question.ask, binding), **context}
            if context
            else render(question.ask, binding)
        ),
    }
    if question.criteria_true or question.criteria_false:
        payload["criteria"] = {
            "true": render(question.criteria_true, binding),
            "false": render(question.criteria_false, binding),
        }
    return payload


def _pending_questions(
    config: ChecklistConfig,
    outputs: Mapping[str, str],
) -> list[_PendingQuestion]:
    pending = []
    for question in config.questions:
        for binding in _bindings(question.foreach, outputs):
            proposition = render(question.proposition, binding)
            pending.append(
                _PendingQuestion(
                    item_id=render(question.id, binding),
                    proposition=proposition,
                    group=question.group,
                    payload=_question_payload(config, question, binding, proposition),
                    tags={key: render(value, binding) for key, value in question.tags.items()},
                    threshold=(
                        config.threshold if question.threshold is None else question.threshold
                    ),
                )
            )
    return pending


def _read_probabilities(body: bytes, keys: Sequence[str]) -> dict[str, float]:
    try:
        answers = json.loads(body)["answers"]
    except (ValueError, KeyError, TypeError) as error:
        raise ChecklistUnavailable("judge_unavailable", "malformed System One reply") from error
    probabilities = {}
    for key in keys:
        answer = answers.get(key) if isinstance(answers, Mapping) else None
        value = answer.get("noul") if isinstance(answer, Mapping) else None
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0.0 <= value <= 1.0
        ):
            raise ChecklistUnavailable("judge_unavailable", f"no probability for {key!r}")
        probabilities[key] = float(value)
    return probabilities


async def _decide(
    backend: DecisionBackend,
    config: ChecklistConfig,
    state: dict[str, object],
    questions: Sequence[_PendingQuestion],
    on_read: Callable[[tuple[int, int]], None] | None = None,
) -> tuple[list[float], tuple[int, int]]:
    """Ask every question in one System One request; P(yes) per question and usage.

    With ``on_read``, the usage of a read that returned is reported there as
    soon as it returns, and never again on an error.
    """

    from kairyu.engine.systemone import (
        SystemOneCapacityError,
        SystemOneUnavailableError,
    )

    keys = [f"q{index}" for index in range(len(questions))]
    body: dict[str, object] = {
        "model": "",
        "state": state,
        "questions": {
            key: dict(question.payload) for key, question in zip(keys, questions, strict=True)
        },
    }
    for name in ("samples", "think", "steps"):
        value = getattr(config, name)
        if value is not None:
            body[name] = value
    try:
        reply = await backend.decide(body)
    except (SystemOneCapacityError, SystemOneUnavailableError) as error:
        raise ChecklistUnavailable("judge_unavailable", str(error)) from error
    if reply.status != 200:
        raise ChecklistUnavailable("judge_unavailable", f"System One status {reply.status}")
    usage = (reply.input_tokens or 0, reply.output_tokens or 0)
    if on_read is not None:
        on_read(usage)
    try:
        probabilities = _read_probabilities(reply.body, keys)
    except ChecklistUnavailable as error:
        billed = (0, 0) if on_read is not None else usage
        raise ChecklistUnavailable(error.reason, error.detail, billed) from error
    return [probabilities[key] for key in keys], usage


def _aggregate(
    questions: Sequence[_PendingQuestion],
    yes_probabilities: Sequence[float],
) -> list[ChecklistItem]:
    grouped: dict[tuple[str, str], list[tuple[_PendingQuestion, float]]] = {}
    for question, p in zip(questions, yes_probabilities, strict=True):
        grouped.setdefault((question.group, question.item_id), []).append((question, p))
    items = []
    for (group, item_id), members in grouped.items():
        first = members[0][0]
        p = min(value for _question, value in members)
        items.append(
            ChecklistItem(
                id=item_id,
                proposition=first.proposition,
                group=group,
                p=p,
                passed=p >= first.threshold,
                tags=first.tags,
            )
        )
    return items


def feedback_text(
    config: ChecklistConfig,
    items: Sequence[ChecklistItem],
    acceptance: float | None = None,
) -> str:
    failing = [item for item in items if not item.passed]
    if acceptance is not None:
        if acceptance >= config.acceptance.threshold:
            return "PASS"
        if not failing:
            return f"FAIL\nThe answer was not accepted as the reply (p={acceptance:.2f})."
    elif not failing:
        return "PASS"
    lines = ["FAIL", config.feedback_header]
    for item in failing:
        lines.append(
            render(
                config.feedback_item,
                {
                    "id": item.id,
                    "proposition": item.proposition,
                    "p": f"{item.p:.2f}",
                    "group": item.group,
                },
            )
        )
    return "\n".join(lines)


async def judge(
    config: ChecklistConfig,
    backend: DecisionBackend | None,
    outputs: Mapping[str, str],
    query: str,
    tools: Sequence[Mapping[str, object]] = (),
    tool_choice: object = None,
    published: str | None = None,
    on_read: Callable[[tuple[int, int]], None] | None = None,
) -> ChecklistVerdict:
    """One verdict: every question in one System One request.

    ``published`` names the output the API publishes (the judged attempt):
    its tool calls enter the state as the API sends them. ``on_read``
    receives each returned read's usage as it returns (see ``_decide``).
    """

    try:
        pending = _pending_questions(config, outputs)
    except TemplateError as error:
        raise ChecklistUnavailable("checklist_unavailable", str(error)) from error
    if not pending:
        if config.on_empty == "pass":
            # A curation read over empty lists has nothing to drop.
            return ChecklistVerdict(passed=True, items=(), text="PASS", usage=(0, 0), reads=0)
        # A verdict over nothing vouches for nothing: no item, no pass.
        raise ChecklistUnavailable("checklist_unavailable", "no items to judge")
    if backend is None:
        raise ChecklistUnavailable("judge_unavailable", "no decision backend")
    if len(pending) > config.max_questions:
        raise ChecklistUnavailable(
            "checklist_unavailable",
            f"{len(pending)} questions exceed the limit of {config.max_questions}",
        )
    state = build_state(config.state, outputs, query, tools, tool_choice, published)
    size = len(json.dumps(state, ensure_ascii=False))
    if config.max_state_chars is not None and size > config.max_state_chars:
        raise ChecklistUnavailable(
            "checklist_unavailable",
            f"state of {size} characters exceeds {config.max_state_chars}",
        )
    yes, usage = await _decide(backend, config, state, pending, on_read)
    items = tuple(_aggregate(pending, yes))
    if config.acceptance is None:
        return ChecklistVerdict(
            passed=all(item.passed for item in items),
            items=items,
            text=feedback_text(config, items),
            usage=usage,
            reads=1,
        )
    try:
        acceptance, accept_usage = await _accept(
            backend,
            config,
            config.acceptance,
            items,
            outputs,
            query,
            tools,
            tool_choice,
            published,
            on_read,
        )
    except ChecklistUnavailable as unavailable:
        if on_read is None:
            # The coverage read completed and billed its tokens.
            unavailable.usage = (
                usage[0] + unavailable.usage[0],
                usage[1] + unavailable.usage[1],
            )
        raise
    usage = (usage[0] + accept_usage[0], usage[1] + accept_usage[1])
    return ChecklistVerdict(
        passed=acceptance >= config.acceptance.threshold,
        items=items,
        text=feedback_text(config, items, acceptance),
        usage=usage,
        reads=2,
        acceptance=acceptance,
    )


async def _accept(
    backend: DecisionBackend,
    config: ChecklistConfig,
    acceptance: AcceptanceConfig,
    items: Sequence[ChecklistItem],
    outputs: Mapping[str, str],
    query: str,
    tools: Sequence[Mapping[str, object]],
    tool_choice: object,
    published: str | None,
    on_read: Callable[[tuple[int, int]], None] | None,
) -> tuple[float, tuple[int, int]]:
    """The acceptance read: one question over its state and the item results."""

    state = build_state(acceptance.state, outputs, query, tools, tool_choice, published)
    state[acceptance.results_key] = [
        {"id": item.id, "point": item.proposition, "p": round(item.p, 4), "passed": item.passed}
        for item in items
    ]
    size = len(json.dumps(state, ensure_ascii=False))
    if config.max_state_chars is not None and size > config.max_state_chars:
        raise ChecklistUnavailable(
            "checklist_unavailable",
            f"acceptance state of {size} characters exceeds {config.max_state_chars}",
        )
    question: dict[str, object] = {"type": "noul", "instructions": acceptance.ask}
    if acceptance.criteria_true or acceptance.criteria_false:
        question["criteria"] = {
            "true": acceptance.criteria_true,
            "false": acceptance.criteria_false,
        }
    pending = [
        _PendingQuestion(
            item_id="acceptance",
            proposition=acceptance.ask,
            group="acceptance",
            payload=question,
            threshold=acceptance.threshold,
        )
    ]
    (p,), usage = await _decide(backend, config, state, pending, on_read)
    return p, usage


def _cut(text: str, limit: int | None) -> str:
    if limit is None or len(text) <= limit:
        return text
    return f"{text[:limit]}\n[... {len(text) - limit} more characters cut ...]"


def _with_tool_calls(
    text: str,
    max_chars: int | None,
    selection: tuple[str, frozenset[str], str | None] | None,
) -> dict[str, object] | None:
    """A role output holding tool calls as ``{text, tool_calls}``, else None.

    Only calls the public API would publish become calls (the shared rules in
    :mod:`kairyu.tool_call_markup`, including the caller's tool choice), so the
    judge reads them as calls the caller will execute. Like the API, a reply
    that publishes calls carries no text. ``max_chars`` still bounds the
    result: output too long falls back to the cut raw text.
    """

    _remaining, calls = generic_tool_calls(text, selection)
    if not calls:
        return None
    # The public API drops the content of a reply that publishes calls.
    structured: dict[str, object] = {"text": "", "tool_calls": calls}
    if max_chars is None:
        return structured
    if len(json.dumps(structured, ensure_ascii=False)) <= max_chars:
        return structured
    return None


def _request_messages(messages: list[object]) -> list[object]:
    """The system and developer messages plus the latest user message, in order."""

    latest_user = max(
        (
            index
            for index, message in enumerate(messages)
            if isinstance(message, Mapping) and message.get("role") == "user"
        ),
        default=None,
    )
    return [
        message
        for index, message in enumerate(messages)
        if index == latest_user
        or (isinstance(message, Mapping) and message.get("role") in {"system", "developer"})
    ]


def build_state(
    sections: Sequence[StateSection],
    outputs: Mapping[str, str],
    query: str,
    tools: Sequence[Mapping[str, object]] = (),
    tool_choice: object = None,
    published: str | None = None,
) -> dict[str, object]:
    """The System One state object: one field per section.

    The request is unwrapped to its role-tagged messages when the query is
    Kairyu's chat transcript; role outputs that are JSON are embedded as JSON
    values, so the judge reads structure instead of escaped text and the
    material it judges cannot pose as instructions.
    """

    state: dict[str, object] = {}
    for section in sections:
        if section.source == _TOOLS_SOURCE:
            state[section.key] = [dict(tool) for tool in tools] or "none"
            continue
        if section.source in _REQUEST_SOURCES:
            messages = conversation_messages(query)
            if messages is not None:
                if section.source == "request":
                    messages = _request_messages(messages)
                cut = [
                    {**message, "content": _cut(message["content"], section.max_chars)}
                    if isinstance(message, dict) and isinstance(message.get("content"), str)
                    else message
                    for message in messages
                ]
                if section.max_total_chars is not None:
                    cut, omitted = bounded_conversation(cut, section.max_total_chars)
                    if omitted:
                        state[f"{section.key}_omitted_messages"] = omitted
                state[section.key] = cut
                continue
            raw = query
        else:
            raw = outputs.get(section.source, "")
            try:
                state[section.key] = parse_json_output(raw)
                continue
            except ValueError:
                pass
            if section.source == published:
                structured = _with_tool_calls(
                    raw, section.max_chars, tool_selection(tools, tool_choice)
                )
                if structured is not None:
                    state[section.key] = structured
                    continue
        state[section.key] = _cut(raw, section.max_chars)
    return state


def curate(
    config: CurationConfig,
    outputs: Mapping[str, str],
    items: Sequence[ChecklistItem],
) -> dict[str, str]:
    """Drop low-probability items from each target list.

    Returns the new output text of every target whose list changed; a target
    whose output is not a JSON object holding a list at its path is left as is.
    """

    dropped = {
        item.id for item in items if item.group == config.drop_group and item.p < config.drop_below
    }
    changed: dict[str, str] = {}
    for target in config.targets:
        text = changed.get(target.role, outputs.get(target.role))
        if text is None:
            continue
        try:
            document = parse_json_output(text)
            entries = json_path(document, target.items_path)
        except (ValueError, KeyError):
            continue
        if not isinstance(document, dict) or not isinstance(entries, list):
            continue
        kept = [
            entry
            for entry in entries
            if not (isinstance(entry, Mapping) and str(entry.get(config.id_key)) in dropped)
        ]
        if len(kept) == len(entries):
            continue
        _set_path(document, target.items_path, kept)
        changed[target.role] = json.dumps(document, ensure_ascii=False)
    return changed


def _set_path(document: dict, path: str, value: object) -> None:
    keys = [key for key in path.split(".") if key]
    current: object = document
    for key in keys[:-1]:
        assert isinstance(current, dict)
        current = current[key]
    assert isinstance(current, dict) and keys
    current[keys[-1]] = value


def unverified_report(reason: str, attempts: int, threshold: float | None) -> VerificationReport:
    return VerificationReport(
        guaranteed=False,
        reason=reason,
        threshold=threshold,
        attempts=attempts,
    )


def verdict_report(
    verdict: ChecklistVerdict,
    *,
    threshold: float,
    attempts: int,
) -> VerificationReport:
    return VerificationReport(
        guaranteed=verdict.passed,
        reason=None if verdict.passed else "refinement_limit",
        threshold=threshold,
        items=verdict.items,
        attempts=attempts,
        acceptance=verdict.acceptance,
    )


__all__ = [
    "AcceptanceConfig",
    "ChecklistConfig",
    "ChecklistItem",
    "ChecklistQuestion",
    "ChecklistUnavailable",
    "ChecklistVerdict",
    "CurationConfig",
    "CurationTarget",
    "DecisionBackend",
    "ItemSource",
    "StateSection",
    "build_state",
    "curate",
    "feedback_text",
    "judge",
    "json_path",
    "parse_json_output",
    "requirement_question",
    "unverified_report",
    "VerificationReport",
    "verdict_report",
]
