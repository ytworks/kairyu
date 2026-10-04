"""Codex live-matrix scenarios (M20 WP-05, D1/D22).

A scenario is one real ``codex exec`` turn against a Kairyu server whose model
behavior is a ``ScenarioBackend`` script (``launcher.py``), through the
recording proxy (``record_proxy.py --exchange-log``). ``run_matrix.py`` checks
the turn's outcome from Codex's ``--json`` events and the wire expectation
from the exchange log; every run also requires that the turn completes and
that no HTTP response is >= 400 except those the provider shape expects.

Scripts are selected per generation call: ``rules`` match the pending user
text of the legacy-rendered prompt; tool loops use index-based ``turns``
(each run gets a fresh server). AUTO prompts are role-preserving JSON
envelopes, so AUTO scripts use ``turns`` and ``default`` only.

A scenario a later work package delivers carries ``xfail`` (gap, WP): it is
expected to fail, and an unexpected pass fails the matrix so that WP flips it.
``LIVE_SCENARIOS`` replace ``scripts/codex_responses_smoke.sh`` for a real
deployment (``run_matrix --live``): natural-language prompts, no script.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from scripts.codex_gate.codex_cli import CUSTOM, OPENAI_BASE_URL
from tests.support.scenario_script import Scenario, ToolCall, Turn, numbered_text

ENGINE_MODEL = "kairyu-scenario"
AUTO_MODEL = "kairyu-auto-scenario"
SHAPES = (CUSTOM, OPENAI_BASE_URL)
DEFAULT_MAX_MODEL_LEN = 131_072
LONG_OUTPUT_TOKENS = 1_500
# AUTO first-token latency vs Codex's idle timer: Kairyu repeats
# response.in_progress after 15 s of data silence, so a 20 s idle timeout only
# survives a 45 s silent generation if those data heartbeats reset it (the
# 300 s production default is scaled down to keep the gate short).
AUTO_FIRST_TOKEN_DELAY_S = 45.0
AUTO_IDLE_TIMEOUT_MS = 20_000
# The overflow run: Codex's prompt (~5-6k toy tokens) fits, then one tool
# output of 30k space-separated numbers (a toy token each, also inside AUTO's
# JSON envelope) pushes the history past the window. Codex compacts mid-turn;
# local compaction sends that history, meets the in-band
# context_length_exceeded, drops the oldest items and retries. Remote v2
# compaction (openai_base_url shape) trims to the catalog window first.
OVERFLOW_MAX_MODEL_LEN = 24_000
OVERFLOW_TOOL_OUTPUT_TOKENS = 100_000
CONTEXT_LENGTH_EXCEEDED = "context_length_exceeded"

ModelKind = Literal["engine", "auto"]


@dataclass(frozen=True)
class Outcome:
    """What Codex reports for the (always completing) turn."""

    final_text: str = "PASS"  # contained in the last agent message
    min_final_words: int = 0
    command: str | None = None  # a completed command execution (exit 0) contains it


@dataclass(frozen=True)
class WireExpectation:
    """What the exchange log must show."""

    max_posts: int | None = None  # POST .../responses bound: no retried turns
    min_heartbeats: int = 0  # repeated response.in_progress within one stream
    in_band_codes: tuple[str, ...] = ()  # response.failed codes allowed in-band
    in_band_required: tuple[str, ...] = ()  # shapes on which they must occur
    compaction: bool = False  # local (tools: []) or remote (compaction_trigger)
    live_web_search: bool = False  # a request declared web_search with live access


@dataclass(frozen=True)
class XFail:
    gap: str
    wp: str
    reason: str


@dataclass(frozen=True)
class MatrixScenario:
    name: str
    model: ModelKind
    prompt: str
    script: Scenario | None  # None: a live deployment answers
    outcome: Outcome = Outcome()
    wire: WireExpectation = WireExpectation()
    codex_args: tuple[str, ...] = ()
    codex_config: tuple[tuple[str, str], ...] = ()  # -c KEY=TOML-VALUE
    attach_image: bool = False  # codex exec --image (a 1x1 PNG)
    idle_timeout_ms: int | None = None  # custom provider stream_idle_timeout_ms
    shapes: tuple[str, ...] = SHAPES
    max_model_len: int = DEFAULT_MAX_MODEL_LEN
    tenant_limits: tuple[tuple[str, int], ...] = ()  # the default tenant's limits
    xfail: XFail | None = None

    @property
    def served_model(self) -> str:
        return AUTO_MODEL if self.model == "auto" else ENGINE_MODEL


def _pwd() -> ToolCall:
    return ToolCall.of("exec_command", cmd="pwd")


def _overflow(name: str, model: ModelKind) -> MatrixScenario:
    call = ToolCall.of(
        "exec_command", cmd="seq -s ' ' 1 30000", max_output_tokens=OVERFLOW_TOOL_OUTPUT_TOKENS
    )
    return MatrixScenario(
        name=name,
        model=model,
        prompt="Print the numbers from 1 to 30000, then reply PASS.",
        script=Scenario(turns=(Turn(call),), default=Turn("PASS")),
        outcome=Outcome(command="seq"),
        wire=WireExpectation(
            in_band_codes=(CONTEXT_LENGTH_EXCEEDED,), in_band_required=(CUSTOM,), compaction=True
        ),
        codex_config=(("tool_output_token_limit", str(OVERFLOW_TOOL_OUTPUT_TOKENS)),),
        max_model_len=OVERFLOW_MAX_MODEL_LEN,
    )


SCENARIOS: tuple[MatrixScenario, ...] = (
    MatrixScenario(
        name="default-turn",
        model="engine",
        prompt="Reply with exactly PASS.",
        script=Scenario(default=Turn("PASS")),
        wire=WireExpectation(max_posts=1),
    ),
    MatrixScenario(
        name="namespace-tool-loop",
        model="engine",
        prompt="Close agent matrix-agent, run pwd, then reply PASS.",
        script=Scenario(
            turns=(
                Turn(ToolCall.of("multi_agent_v1__close_agent", target="matrix-agent")),
                Turn(_pwd()),
            ),
            default=Turn("PASS"),
        ),
        outcome=Outcome(command="pwd"),
        wire=WireExpectation(max_posts=3),
    ),
    MatrixScenario(
        name="full-access-web-search",
        model="engine",
        prompt="Reply with exactly PASS.",
        script=Scenario(default=Turn("PASS")),
        wire=WireExpectation(max_posts=1, live_web_search=True),
        codex_args=("--dangerously-bypass-approvals-and-sandbox",),
    ),
    MatrixScenario(
        name="long-output",
        model="engine",
        prompt="Write 1500 words.",
        script=Scenario(default=Turn(numbered_text(LONG_OUTPUT_TOKENS), chunk_tokens=50)),
        outcome=Outcome(final_text="w1499", min_final_words=LONG_OUTPUT_TOKENS),
        wire=WireExpectation(max_posts=1),
    ),
    MatrixScenario(
        name="auto-long-turn",
        model="auto",
        prompt="Think for a long time, then reply PASS.",
        script=Scenario(default=Turn("PASS", first_token_delay_s=AUTO_FIRST_TOKEN_DELAY_S)),
        wire=WireExpectation(max_posts=1, min_heartbeats=2),
        idle_timeout_ms=AUTO_IDLE_TIMEOUT_MS,
        shapes=(CUSTOM,),  # the built-in openai provider's idle timeout is not settable
    ),
    _overflow("overflow-compaction", "engine"),
    _overflow("overflow-compaction-auto", "auto"),
    MatrixScenario(
        name="local-compaction",
        model="engine",
        prompt="Run pwd, then reply PASS.",
        script=Scenario(turns=(Turn(_pwd()),), default=Turn("PASS")),
        outcome=Outcome(command="pwd"),
        wire=WireExpectation(compaction=True),
        codex_config=(("model_auto_compact_token_limit", "1000"),),
    ),
    # G-input-items-1 P0 mitigation: the text-only catalog makes Codex drop the
    # attachment instead of sending an input_image Kairyu rejects (WP-25).
    MatrixScenario(
        name="image-attached",
        model="engine",
        prompt="Describe the attached image, then reply PASS.",
        script=Scenario(default=Turn("PASS")),
        attach_image=True,
        wire=WireExpectation(max_posts=1),
    ),
    MatrixScenario(
        name="backpressure-retry",
        model="engine",
        prompt="Run pwd, then reply PASS.",
        script=Scenario(turns=(Turn(_pwd()),), default=Turn("PASS")),
        outcome=Outcome(command="pwd"),
        # A burst of two (the matrix's catalog fetch, Codex's first request)
        # refilled once per 10 s: the tool loop's follow-up is refused.
        tenant_limits=(("requests_per_minute", 6), ("request_burst", 2)),
        xfail=XFail(
            gap="G-errors-5",
            wp="WP-07",
            reason="a transient tenant refusal is a 429, which Codex never retries; "
            "503 slow_down with Retry-After is (O-2)",
        ),
    ),
)

LIVE_SCENARIOS: tuple[MatrixScenario, ...] = (
    MatrixScenario(
        name="live-text",
        model="engine",
        prompt="Reply with exactly PASS. Do not call tools.",
        script=None,
    ),
    MatrixScenario(
        name="live-tool",
        model="engine",
        prompt="Run exactly one shell command: pwd. After receiving its output, "
        "reply with exactly PASS.",
        script=None,
        outcome=Outcome(command="pwd"),
    ),
    MatrixScenario(
        name="live-full-access",
        model="engine",
        prompt="Reply with exactly PASS. Do not call tools.",
        script=None,
        wire=WireExpectation(live_web_search=True),
        codex_args=("--dangerously-bypass-approvals-and-sandbox",),
    ),
)


def scenario_named(name: str) -> MatrixScenario:
    for scenario in (*SCENARIOS, *LIVE_SCENARIOS):
        if scenario.name == name:
            return scenario
    raise KeyError(f"unknown Codex matrix scenario {name!r}")
