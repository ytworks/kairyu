"""Direct-engine Responses execution through one output assembler.

Streams pull backend deltas live; unary turns keep ``execute_chat`` (and its
usage recording) but rebuild the output from the raw completion. Both feed
the same reasoning split, protocol tool scanner, and assembler, so a stream's
final snapshot and the unary response carry the same items.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import AsyncIterator

from fastapi import Request

from kairyu.engine.backend import GenerationResult, backend_max_model_len
from kairyu.entrypoints.server.chat_service import (
    ChatRequestError,
    ExecutedChat,
    ValidatedChatRequest,
    execute_chat,
)
from kairyu.entrypoints.server.metering import record_state_usage, stream_usage_owner_from_state
from kairyu.entrypoints.server.responses_codec import (
    _compaction_output_from_message,
    _CompactionCodec,
)
from kairyu.entrypoints.server.responses_events import (
    OutputAssembler,
    ReasoningSplit,
    ResponseEmitter,
    heartbeat_tick,
    prompt_opens_reasoning,
)
from kairyu.entrypoints.server.responses_protocol import (
    ResponsesRequest,
    _BufferedFailure,
    _usage_payload,
    context_overflow_error,
    is_context_overflow,
    responses_error_payload,
)
from kairyu.entrypoints.server.responses_store import PendingSave
from kairyu.entrypoints.server.responses_tools import _namespace_names
from kairyu.entrypoints.server.sse_keepalive import iter_with_idle_markers
from kairyu.entrypoints.server.tool_stream import tool_stream_scanner_for

logger = logging.getLogger(__name__)


def _record_execution(
    http_request: Request,
    request: ResponsesRequest,
    execution: ExecutedChat,
) -> None:
    usage = _usage_payload(
        execution.result.prompt,
        execution.result.completions,
        execution.result.usage,
    )
    owner = getattr(http_request.state, "tenant", None) or "default"
    record_state_usage(
        http_request.app.state,
        tenant=owner,
        model=request.model,
        prompt_tokens=usage["input_tokens"],
        completion_tokens=usage["output_tokens"],
        cached_tokens=usage["input_tokens_details"]["cached_tokens"],
        reservation=getattr(http_request.state, "tenant_admission", None),
        usage_exact=execution.result.usage is not None,
    )


def terminal_status(
    finish_reason: str | None, *, cap_omitted: bool, context_bounded: bool
) -> tuple[str, dict | None]:
    """``completed``, ``incomplete``, or ``context_exhausted``.

    With no requested cap the engine generates up to the remaining context
    (#496), so a length stop on a backend with a known window means the
    context ran out: reporting ``incomplete(max_output_tokens)`` would name a
    cap the client never set (and Codex retries it), while
    ``context_length_exceeded`` lets the client compact.
    """

    if finish_reason in {"length", "max_tokens"}:
        if cap_omitted and context_bounded:
            return "context_exhausted", None
        return "incomplete", {"reason": "max_output_tokens"}
    return "completed", None


def _status(request: ResponsesRequest, validated: ValidatedChatRequest, finish_reason):
    return terminal_status(
        finish_reason,
        cap_omitted=request.max_output_tokens is None,
        context_bounded=backend_max_model_len(validated.engine) is not None,
    )


def _assembler(
    request: ResponsesRequest,
    validated: ValidatedChatRequest,
    emitter: ResponseEmitter,
) -> OutputAssembler:
    chat_request = validated.input.request
    choice = validated.input.normalized_tool_choice
    scanner = (
        tool_stream_scanner_for(validated.input.tool_call_protocol, chat_request.tools, choice)
        if chat_request.tools and choice.mode != "none"
        else None
    )
    return OutputAssembler(
        emitter,
        scanner=scanner,
        tool_choice=choice,
        parallel=validated.input.parallel_tool_calls,
        namespaces=_namespace_names(request.tools),
    )


def _failure_payload(error: BaseException) -> dict:
    if is_context_overflow(error):
        return context_overflow_error().payload()
    if isinstance(error, ChatRequestError):
        return responses_error_payload(error)
    return {
        "message": f"upstream backend error ({type(error).__name__})",
        "type": "upstream_error",
        "param": None,
        "code": "server_error",
    }


async def engine_stream(
    request: ResponsesRequest,
    validated: ValidatedChatRequest,
    *,
    emitter: ResponseEmitter,
    saver: PendingSave,
    owner: str,
    http_request: Request,
) -> AsyncIterator[str | bytes]:
    """Stream a direct-engine turn: reasoning, prose, and calls as they decode."""

    assembler = _assembler(request, validated, emitter)
    split = ReasoningSplit(opened=prompt_opens_reasoning(validated.input.prompt))
    usage_owner = stream_usage_owner_from_state(
        http_request.app.state,
        tenant=owner,
        model=request.model,
        prompt=validated.generation_request.prompt,
        reservation=getattr(http_request.state, "tenant_admission", None),
    )
    last: GenerationResult | None = None
    sent = 0
    saw_final = False

    def usage() -> dict:
        completions = last.completions if last is not None else ()
        return _usage_payload(
            validated.generation_request.prompt, completions, usage_owner.latest_usage
        )

    try:
        for frame in emitter.start():
            yield frame
        try:
            usage_owner.mark_dispatched()
            partials = iter_with_idle_markers(
                validated.engine.stream(validated.generation_request), heartbeat_tick()
            )
            async with contextlib.aclosing(partials):
                async for partial in partials:
                    if partial is not None:
                        last = partial
                        usage_owner.observe(partial.usage, partial.completions)
                        completion = min(
                            partial.completions, key=lambda item: item.index, default=None
                        )
                        if completion is not None:
                            delta, sent = completion.delta_after(sent)
                            reasoning, content = split.feed(
                                completion,
                                delta if type(delta) is str else "",
                                final=partial.finished,
                            )
                            for frame in assembler.reasoning(reasoning):
                                yield frame
                            for frame in assembler.content(content, final=partial.finished):
                                yield frame
                        saw_final = saw_final or partial.finished
                        if assembler.failure is not None:
                            break
                    if emitter.heartbeat_due():
                        yield emitter.heartbeat()
        except Exception as error:
            if not is_context_overflow(error):
                logger.exception("Responses API upstream stream failed")
            _envelope, frames = emitter.fail(_failure_payload(error), usage())
            for frame in frames:
                yield frame
            return
        if assembler.failure is None and not saw_final:
            reasoning, content = split.feed(None, "", final=True)
            for frame in assembler.reasoning(reasoning) + assembler.content(content, final=True):
                yield frame
        completions = last.completions if last is not None else ()
        completion = min(completions, key=lambda item: item.index, default=None)
        status, details = _status(
            request, validated, completion.finish_reason if completion is not None else None
        )
        if status == "context_exhausted":
            usage_owner.mark_completed()
            _envelope, frames = emitter.fail(
                context_overflow_error(exhausted=True).payload(), usage()
            )
            for frame in frames:
                yield frame
            return
        frames = assembler.finish(incomplete=status == "incomplete")
        if assembler.failure is not None:
            # Like the Messages adapter: an abandoned gate settles inexactly.
            _envelope, frames = emitter.fail(assembler.failure, usage())
            for frame in frames:
                yield frame
            return
        usage_owner.mark_completed()
        envelope, terminal = emitter.complete(status, usage(), details)
        for frame in frames + terminal:
            yield frame
        saver.commit(envelope)
    finally:
        usage_owner.finalize()


async def _execute(
    request: ResponsesRequest,
    validated: ValidatedChatRequest,
    http_request: Request,
) -> ExecutedChat:
    admission = getattr(http_request.state, "tenant_admission", None)
    try:
        if admission is not None:
            admission.mark_dispatched()
        execution = await execute_chat(validated)
    except ChatRequestError as error:
        if error.execution is None:
            if is_context_overflow(error):
                raise _BufferedFailure(context_overflow_error().payload(), 400) from error
            raise _BufferedFailure.from_chat_error(error) from error
        # Chat's own tool gates judged its parse of the raw text; the
        # Responses assembler re-judges the same text with reasoning removed.
        execution = error.execution
    except Exception as error:
        if is_context_overflow(error):
            raise _BufferedFailure(context_overflow_error().payload(), 400) from error
        logger.exception("Responses API upstream generation failed")
        raise _BufferedFailure(
            {
                "message": f"upstream backend error ({type(error).__name__})",
                "type": "upstream_error",
                "code": "backend_error",
            },
            502,
        ) from error
    _record_execution(http_request, request, execution)
    return execution


def _first_completion(execution: ExecutedChat):
    return min(execution.result.completions, key=lambda item: item.index, default=None)


async def engine_unary(
    request: ResponsesRequest,
    validated: ValidatedChatRequest,
    *,
    emitter: ResponseEmitter,
    http_request: Request,
) -> dict:
    """Run a direct-engine turn to completion and return its envelope."""

    execution = await _execute(request, validated, http_request)
    completion = _first_completion(execution)
    result = execution.result
    usage = _usage_payload(result.prompt, result.completions, result.usage)
    status, details = _status(
        request, validated, completion.finish_reason if completion is not None else None
    )
    if status == "context_exhausted":
        raise _BufferedFailure(context_overflow_error(exhausted=True).payload(), 400)
    assembler = _assembler(request, validated, emitter)
    split = ReasoningSplit(opened=prompt_opens_reasoning(validated.input.prompt))
    reasoning, content = split.feed(
        completion, completion.text if completion is not None else "", final=True
    )
    assembler.reasoning(reasoning)
    assembler.content(content, final=True)
    assembler.finish(incomplete=status == "incomplete")
    if assembler.failure is not None:
        raise _BufferedFailure(assembler.failure, 502)
    envelope, _frames = emitter.complete(status, usage, details)
    return envelope


async def engine_compaction(
    request: ResponsesRequest,
    validated: ValidatedChatRequest,
    *,
    http_request: Request,
    compaction_codec: _CompactionCodec,
    owner: str,
) -> tuple[list[dict], dict, str, dict | None]:
    """A tool-free summarization turn sealed into one compaction item."""

    execution = await _execute(request, validated, http_request)
    completion = _first_completion(execution)
    result = execution.result
    usage = _usage_payload(result.prompt, result.completions, result.usage)
    status, details = terminal_status(
        completion.finish_reason if completion is not None else None,
        cap_omitted=False,
        context_bounded=False,
    )
    if status != "completed":
        return [], usage, status, details
    split = ReasoningSplit(opened=prompt_opens_reasoning(validated.input.prompt))
    _reasoning, summary = split.feed(
        completion, completion.text if completion is not None else "", final=True
    )
    output = _compaction_output_from_message(
        {"content": summary}, compaction_codec=compaction_codec, owner=owner
    )
    return output, usage, status, None
