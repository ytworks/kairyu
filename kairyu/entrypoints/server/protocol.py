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


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage = Usage()
    # m11 D1: explicit opt-in trace (X-Kairyu-Trace: 1); excluded when None
    kairyu_trace: list[str] | None = Field(default=None, exclude=False)


class ChunkDelta(BaseModel):
    role: str | None = None
    content: str | list[ContentPart] | None = None
    tool_calls: list[ToolCall] | None = None


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
