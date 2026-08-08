"""SpeculativeRunner: deterministic draft + target verification (m8 D4).

A ``ModelRunner`` wrapper (composition — the scheduler is untouched beyond its
D3 reservation contract). For a speculative decode chunk it proposes a draft
from the committed context, then sends every target-verification position to
the wrapped runner as one multi-token chunk.  An immutable overlay state exposes
the committed tokens plus the complete draft (draft tokens are not in scheduler
state).  All compatible requests in the scheduler step share that one wrapped
runner call before greedy or point-mass rejection verification returns each
accepted prefix plus its bonus/correction token.

Multi-token chunks are an explicit optional ``ModelRunner`` capability.
Targets must declare ``supports_batched_verification is True``; an undeclared
custom runner keeps the pre-#215 one-position-at-a-time contract. The branch is
chosen before execution, so a malformed capable-runner result is never retried
after model/KV side effects.

Per-request gating: stochastic sampling and penalties are supported. Grammar
and forced-token requests still bypass because matcher/continuation rollback is
outside the deterministic-draft verification contract. Bypassed chunks return
a single token; the D3 shortfall accounting absorbs the difference.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from kairyu.engine.core.draft import DraftSource, NGramDraftSource
from kairyu.engine.core.engine_core import ModelRunner, StepOutput
from kairyu.engine.core.scheduler import ScheduledChunk
from kairyu.engine.core.spec_decode import verify_greedy, verify_rejection


class _OverlayState:
    """Immutable view of a request state with ``outputs`` overridden."""

    __slots__ = ("_base", "outputs", "outputs_override", "output_epoch")

    def __init__(
        self,
        base: object,
        outputs: tuple[int, ...],
        *,
        output_epoch: int | None = None,
    ) -> None:
        self._base = base
        self.outputs = list(outputs)
        self.outputs_override = True
        self.output_epoch = (
            getattr(base, "output_epoch", 0)
            if output_epoch is None
            else output_epoch
        )

    def __getattr__(self, name: str):
        return getattr(object.__getattribute__(self, "_base"), name)


class SpeculativeRunner:
    """Wraps a ModelRunner; speculative chunks (num_tokens > 1) get draft+verify."""

    def __init__(self, runner: ModelRunner, draft_source: DraftSource | None = None) -> None:
        # default n-gram keeps m8 behavior byte-identical (m17 D3)
        self._draft_source = draft_source or NGramDraftSource()
        self._runner = runner
        self.draft_proposed = 0
        self.draft_accepted = 0
        self._overlay_epoch = 0

    @property
    def draft_source_name(self) -> str:
        return str(getattr(self._draft_source, "source_name", "custom"))

    def _overlay(self, state: object, outputs: tuple[int, ...]) -> _OverlayState:
        """Give every replaceable draft prefix a distinct sync epoch."""

        self._overlay_epoch += 1
        return _OverlayState(
            state,
            outputs,
            output_epoch=self._overlay_epoch,
        )

    @property
    def mean_accepted(self) -> float:
        """Accepted draft tokens per proposed draft token (G4 M-A4 lineage)."""
        if self.draft_proposed == 0:
            return 0.0
        return self.draft_accepted / self.draft_proposed

    def speculative_execution_stats(self) -> dict[str, object]:
        """Machine-readable acceptance evidence for benchmark artifacts."""

        return {
            "draft_source": self.draft_source_name,
            "draft_proposed": self.draft_proposed,
            "draft_accepted": self.draft_accepted,
            "mean_accepted": self.mean_accepted,
        }

    def release(self, request_id: str) -> None:
        """Forward finish cleanup to the wrapped runner (E2)."""
        draft_release = getattr(self._draft_source, "release", None)
        if callable(draft_release):
            draft_release(request_id)
        release = getattr(self._runner, "release", None)
        if release is not None:
            release(request_id)

    def decode_graph_metadata(self) -> object:
        """Forward live graph counters from the wrapped target runner."""

        getter = getattr(self._runner, "decode_graph_metadata", None)
        if not callable(getter):
            raise RuntimeError(
                f"{type(self._runner).__name__} exposes no decode-graph metadata"
            )
        return getter()

    def set_batched_verification_enabled(self, enabled: bool) -> None:
        """Forward the matched-A/B and rollback switch to the target runner."""
        setter = getattr(self._runner, "set_batched_verification_enabled", None)
        if not callable(setter):
            raise RuntimeError(
                f"{type(self._runner).__name__} does not expose a batched "
                "verification switch"
            )
        setter(enabled)

    def verification_execution_stats(
        self, *, reset: bool = False
    ) -> object:
        """Expose target-runner structural counters through the wrapper."""
        getter = getattr(self._runner, "verification_execution_stats", None)
        if not callable(getter):
            raise RuntimeError(
                f"{type(self._runner).__name__} does not expose verification "
                "execution stats"
            )
        return getter(reset=reset)

    def set_decode_page_table_cache_enabled(self, enabled: bool) -> None:
        """Forward the page-table allocation/copy rollback switch."""
        setter = getattr(
            self._runner, "set_decode_page_table_cache_enabled", None
        )
        if not callable(setter):
            raise RuntimeError(
                f"{type(self._runner).__name__} does not expose a decode "
                "page-table cache switch"
            )
        setter(enabled)

    def decode_page_table_cache_stats(
        self, *, reset: bool = False
    ) -> object:
        """Expose rank-local or gathered page-table evidence."""
        getter = getattr(self._runner, "decode_page_table_cache_stats", None)
        if not callable(getter):
            raise RuntimeError(
                f"{type(self._runner).__name__} does not expose decode "
                "page-table cache stats"
            )
        return getter(reset=reset)

    def execute(
        self, scheduled: tuple[ScheduledChunk, ...], states: Mapping[str, object]
    ) -> StepOutput:
        if getattr(self._runner, "supports_batched_verification", False) is True:
            return self._execute_batched(scheduled, states)
        return self._execute_sequential(scheduled, states)

    def _propose(self, chunk: ScheduledChunk, state: object) -> tuple[int, ...]:
        committed = tuple(state.outputs)
        context = state.request.prompt_token_ids + committed
        propose_for_request = getattr(
            self._draft_source, "propose_for_request", None
        )
        if callable(propose_for_request):
            proposed = propose_for_request(
                chunk.request_id,
                context,
                chunk.num_tokens - 1,
            )
        else:
            proposed = self._draft_source.propose(
                context,
                max_draft=chunk.num_tokens - 1,
            )
        return tuple(
            proposed
        )[: chunk.num_tokens - 1]

    def _finish_verification(
        self,
        request_id: str,
        draft: tuple[int, ...],
        target: tuple,
        *,
        stochastic: bool,
    ) -> tuple:
        expected = len(draft) + 1
        if len(target) != expected:
            raise RuntimeError(
                "target runner returned the wrong number of speculative "
                f"verification positions for {request_id!r}: "
                f"got {len(target)}, expected {expected}"
            )
        verify = verify_rejection if stochastic else verify_greedy
        result = verify(
            draft,
            tuple(token.token_id for token in target),
        )
        self.draft_proposed += len(draft)
        self.draft_accepted += result.accepted
        commit = getattr(self._draft_source, "commit_verification", None)
        if callable(commit):
            commit(request_id, result.accepted)
        return tuple(target[: result.accepted + 1])

    def _execute_batched(
        self,
        scheduled: tuple[ScheduledChunk, ...],
        states: Mapping[str, object],
    ) -> StepOutput:
        runner_chunks: list[ScheduledChunk] = []
        runner_states = dict(states)
        drafts: dict[str, tuple[int, ...]] = {}
        for chunk in scheduled:
            if chunk.is_prefill or chunk.num_tokens == 1:
                runner_chunks.append(chunk)
            elif not states[
                chunk.request_id
            ].request.sampling.is_speculative_eligible:
                # bypass: score exactly one token; D3 shortfall releases the rest
                runner_chunks.append(replace(chunk, num_tokens=1))
            else:
                state = states[chunk.request_id]
                committed = tuple(state.outputs)
                draft = self._propose(chunk, state)
                drafts[chunk.request_id] = draft
                runner_states[chunk.request_id] = self._overlay(
                    state, (*committed, *draft)
                )
                # The target emits one score for each draft token and one
                # bonus/correction score.  Keeping one chunk per request lets
                # the production runner pack every compatible row together.
                runner_chunks.append(
                    replace(chunk, num_tokens=len(draft) + 1)
                )

        if not runner_chunks:
            return {}
        target_output = self._runner.execute(tuple(runner_chunks), runner_states)
        sampled = dict(target_output)
        for request_id, draft in drafts.items():
            target = target_output[request_id]
            # target[i] is the model's own next token given the draft prefix
            # of length i (verify_greedy's contract; walkthrough in m8 §6).
            sampled[request_id] = self._finish_verification(
                request_id,
                draft,
                target,
                stochastic=(
                    states[request_id].request.sampling.temperature != 0.0
                ),
            )
        return sampled

    def _execute_sequential(
        self,
        scheduled: tuple[ScheduledChunk, ...],
        states: Mapping[str, object],
    ) -> StepOutput:
        """Pre-#215 compatibility path for one-token custom ModelRunners."""
        plain: list[ScheduledChunk] = []
        speculative: list[ScheduledChunk] = []
        for chunk in scheduled:
            if chunk.is_prefill or chunk.num_tokens == 1:
                plain.append(chunk)
            elif not states[
                chunk.request_id
            ].request.sampling.is_speculative_eligible:
                plain.append(replace(chunk, num_tokens=1))
            else:
                speculative.append(chunk)

        sampled: StepOutput = {}
        if plain:
            sampled.update(self._runner.execute(tuple(plain), states))
        for chunk in speculative:
            state = states[chunk.request_id]
            committed = tuple(state.outputs)
            draft = self._propose(chunk, state)
            target = []
            for index in range(len(draft) + 1):
                sub_chunk = ScheduledChunk(
                    request_id=chunk.request_id,
                    num_tokens=1,
                    is_prefill=False,
                    position=chunk.position + index,
                )
                result = self._runner.execute(
                    (sub_chunk,),
                    {
                        chunk.request_id: self._overlay(
                            state,
                            (*committed, *draft[:index]),
                        )
                    },
                )
                records = result[chunk.request_id]
                if len(records) != 1:
                    raise RuntimeError(
                        "legacy target runner must return exactly one "
                        "speculative verification position for "
                        f"{chunk.request_id!r}; got {len(records)}"
                    )
                target.append(records[0])
            sampled[chunk.request_id] = self._finish_verification(
                chunk.request_id,
                draft,
                tuple(target),
                stochastic=(
                    state.request.sampling.temperature != 0.0
                ),
            )
        return sampled
