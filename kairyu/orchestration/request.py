"""Transport-neutral, per-call orchestration intent."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from kairyu.sampling_params import (
    PARALLEL_TOOL_CALLS_EXTRA_ARG,
    SamplingParams,
    resolve_parallel_tool_calls,
)


@dataclass(frozen=True)
class OrchestrationRequest:
    """One immutable orchestration call.

    The orchestrator instance is shared by concurrent HTTP requests, so public
    sampling and output intent must travel with the call instead of mutating
    constructor defaults.  Tool schemas remain structured until the final
    backend boundary; native backends may also receive a prompt rendered from
    the same schemas by the chat-template layer.
    """

    prompt: str
    sampling_params: SamplingParams
    tools: tuple[Mapping[str, object], ...] = ()
    tool_choice: str | Mapping[str, object] | None = None
    tools_in_prompt: bool = False
    response_format: Mapping[str, object] | None = None
    parallel_tool_calls: bool | None = None
    tool_call_protocol: str = "generic"
    reasoning_effort: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "tools", tuple(self.tools))
        sampling_format = self.sampling_params.extra_args.get("response_format")
        if self.response_format != sampling_format:
            raise ValueError("response_format intent must match sampling_params.extra_args")
        resolve_parallel_tool_calls(
            self.parallel_tool_calls,
            self.sampling_params.extra_args,
        )
        if self.tool_call_protocol not in {"generic", "llama", "qwen", "deepseek_v4"}:
            raise ValueError(
                "tool_call_protocol must be generic, llama, qwen or deepseek_v4"
            )
        if self.reasoning_effort not in {None, "low", "high", "max"}:
            raise ValueError("reasoning_effort must be low, high, max, or null")

    def internal_sampling_params(
        self,
        *,
        max_tokens_cap: int | None = None,
    ) -> SamplingParams:
        """Sampling for non-final planning/proposal/verification stages.

        Intermediate alternatives are consumed as one control-flow value, so
        generating ``n`` copies or retaining token logprobs only wastes work.
        A final-output grammar must likewise apply only at the answer boundary;
        constraining planner/verifier text can corrupt the orchestration DAG.
        Their token ceiling is an orchestration policy: a large public final
        allowance must not let private control text run past backend timeouts.
        """

        extra_args = dict(self.sampling_params.extra_args)
        extra_args.pop("response_format", None)
        extra_args.pop(PARALLEL_TOOL_CALLS_EXTRA_ARG, None)
        max_tokens = self.sampling_params.max_tokens
        if max_tokens_cap is not None:
            max_tokens = max_tokens_cap if max_tokens is None else min(max_tokens, max_tokens_cap)
        return self.sampling_params.clone(
            n=1,
            best_of=None,
            logprobs=None,
            max_tokens=max_tokens,
            extra_args=extra_args,
        )


def default_orchestration_request(
    prompt: str,
    sampling_params: SamplingParams,
) -> OrchestrationRequest:
    """Build the backwards-compatible request used by the Python facade."""

    return OrchestrationRequest(
        prompt=prompt,
        sampling_params=sampling_params,
        response_format=sampling_params.extra_args.get("response_format"),
    )
