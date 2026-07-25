"""OpenAI chat-completions wire schema (request/response/chunk models)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ContentPart(BaseModel):
    """OpenAI vision content part (m11 D5): text or image_url."""

    type: str
    text: str | None = None
    image_url: dict | None = None


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    content: str | list[ContentPart] | None = None
    name: str | None = None
    tool_call_id: str | None = None


class RouteFeatures(BaseModel):
    char_len: int
    word_count: int
    has_code_fence: bool
    math_symbol_count: int
    reasoning_keyword_count: int
    multi_step_marker_count: int
    question_count: int


class RouteDecisionPayload(BaseModel):
    target: str
    confidence: float
    reason: str
    features: RouteFeatures


class RoutePreviewRequest(BaseModel):
    model: str
    messages: list[ChatMessage] = Field(min_length=1)


class RoutePreviewResponse(BaseModel):
    model: str
    orchestrated: bool
    binding: bool = False
    router_type: str | None = None
    target: str | None = None
    confidence: float | None = None
    reason: str | None = None
    features: RouteFeatures | None = None


class RouterDescriptorPayload(BaseModel):
    router_type: str
    thresholds: dict[str, int] | None = None
    min_confidence: float | None = None
    fallback_type: str | None = None
    epsilon: float | None = None
    is_warm: bool | None = None
    min_updates_per_arm: int | None = None


class EngineDescriptorPayload(BaseModel):
    backend_type: str
    model: str | None = None


class EngineResolutionPayload(BaseModel):
    configured: bool
    engine: str
    fallback: bool


class TargetResolutionPayload(BaseModel):
    configured: bool | None = None
    engine: str | None = None
    fallback: bool | None = None
    mode: str | None = None
    engines: list[EngineResolutionPayload] = Field(default_factory=list)


class RoleDescriptorPayload(BaseModel):
    name: str
    worker: str
    role_type: str
    depends_on: list[str]
    verifies: str | None = None


class BudgetDescriptorPayload(BaseModel):
    max_steps: int
    max_refine_depth: int
    max_cost_usd: float | None = None


class RoutingModelDescriptorPayload(BaseModel):
    router: RouterDescriptorPayload
    targets: list[str]
    configured_engines: dict[str, EngineDescriptorPayload]
    target_resolution: dict[str, TargetResolutionPayload]
    roles: list[RoleDescriptorPayload]
    budget: BudgetDescriptorPayload
    moa_samples: int


class RoutingResponse(BaseModel):
    models: dict[str, RoutingModelDescriptorPayload]


class StreamOptions(BaseModel):
    include_usage: bool = False


class TopLogprobEntry(BaseModel):
    token: str
    logprob: float
    bytes: list[int] | None = None


class LogprobEntry(BaseModel):
    token: str
    logprob: float
    bytes: list[int] | None = None
    top_logprobs: list[TopLogprobEntry] = Field(default_factory=list)


class ChoiceLogprobs(BaseModel):
    content: list[LogprobEntry] | None = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[ChatMessage] = Field(min_length=1)
    temperature: float = 1.0
    top_p: float = 1.0
    n: int = 1
    logprobs: bool = False
    top_logprobs: int | None = None
    stream: bool = False
    stream_options: StreamOptions | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None  # modern-SDK alias of max_tokens
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    stop: str | list[str] | None = None
    seed: int | None = None
    tools: list[dict] | None = None
    tool_choice: str | dict | None = None
    response_format: dict | None = None
    user: str | None = None


class FunctionCall(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str
    type: str = "function"
    function: FunctionCall


class ResponseMessage(BaseModel):
    role: str = "assistant"
    content: str | list[ContentPart] | None = None
    tool_calls: list[ToolCall] | None = None


class Choice(BaseModel):
    index: int
    message: ResponseMessage
    finish_reason: str | None = None
    logprobs: ChoiceLogprobs | None = None


class PromptTokensDetails(BaseModel):
    cached_tokens: int = 0


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    prompt_tokens_details: PromptTokensDetails | None = None


class KairyuTraceTiming(BaseModel):
    queued_at: str | None = None
    started_at: str | None = None
    first_token_at: str | None = None
    completed_at: str | None = None


class KairyuTraceUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0


class KairyuTraceBudget(BaseModel):
    max_steps: int
    steps_before: int
    steps_consumed: int
    steps_remaining: int
    max_cost_usd: float | None = None
    cost_before_usd: float
    cost_consumed_usd: float
    cost_remaining_usd: float | None = None


class KairyuTraceError(BaseModel):
    type: str
    retryable: bool = False


class KairyuTraceEvent(BaseModel):
    seq: int
    node: str
    role: str | None = None
    kind: str
    status: str
    attempt: int = 0
    worker: str | None = None
    engine: str | None = None
    model: str | None = None
    timing: KairyuTraceTiming | None = None
    usage: KairyuTraceUsage | None = None
    budget: KairyuTraceBudget | None = None
    detail: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
    error: KairyuTraceError | None = None


class KairyuTraceV2(BaseModel):
    trace_version: str
    request_id: str
    started_at: str
    completed_at: str
    events: list[KairyuTraceEvent]


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage = Usage()
    # m11 D1: explicit opt-in trace (X-Kairyu-Trace: 1); excluded when None
    kairyu_trace: list[str] | None = None
    # Additive structured trace for evaluation tooling. Like the legacy field,
    # it is populated only when X-Kairyu-Trace: 1 is requested.
    kairyu_trace_v2: KairyuTraceV2 | None = None
    # Actual route uses the same schema as route preview. It is populated only
    # for traced orchestrated responses.
    kairyu_route: RouteDecisionPayload | None = None


class ChunkToolCall(BaseModel):
    # streamed tool-call deltas require an `index` so SDK accumulators can merge
    # fragments across chunks (S6); the non-streaming ToolCall has no index
    index: int
    id: str
    type: str = "function"
    function: FunctionCall


class ChunkDelta(BaseModel):
    role: str | None = None
    content: str | list[ContentPart] | None = None
    tool_calls: list[ChunkToolCall] | None = None


class ChunkChoice(BaseModel):
    index: int
    delta: ChunkDelta
    finish_reason: str | None = None
    # OpenAI: logprobs sits on the chunk CHOICE (sibling of delta), never inside it
    logprobs: ChoiceLogprobs | None = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChunkChoice]
    # OpenAI contract (m9 D1): key OMITTED unless stream_options.include_usage,
    # then null on every chunk except the final usage chunk (choices: [])
    usage: Usage | None = None


class CompletionRequest(BaseModel):
    """Legacy /v1/completions (m9 D3). echo/suffix/best_of are rejected."""

    model_config = ConfigDict(extra="allow")

    model: str
    prompt: str | list[str]
    max_tokens: int | None = 16
    temperature: float = 1.0
    top_p: float = 1.0
    n: int = 1
    logprobs: int | None = None  # legacy top-k int, capped at 5
    stream: bool = False
    stream_options: StreamOptions | None = None
    stop: str | list[str] | None = None
    seed: int | None = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    user: str | None = None


class CompletionLogprobs(BaseModel):
    """Legacy four-parallel-array shape; offsets from 0 within `text` (echo
    is rejected, so there is no prompt segment to offset past)."""

    tokens: list[str] = Field(default_factory=list)
    token_logprobs: list[float] = Field(default_factory=list)
    top_logprobs: list[dict[str, float]] | None = None
    text_offset: list[int] = Field(default_factory=list)


class CompletionChoice(BaseModel):
    index: int
    text: str
    logprobs: CompletionLogprobs | None = None
    finish_reason: str | None = None


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: Usage = Usage()


class CompletionChunk(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: list[CompletionChoice]
    usage: Usage | None = None


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "kairyu"


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelCard]
