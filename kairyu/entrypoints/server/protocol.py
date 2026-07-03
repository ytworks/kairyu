"""OpenAI chat-completions wire schema (request/response/chunk models)."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: str
    content: str | None = None
    name: str | None = None
    tool_call_id: str | None = None


class StreamOptions(BaseModel):
    include_usage: bool = False


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[ChatMessage] = Field(min_length=1)
    temperature: float = 1.0
    top_p: float = 1.0
    n: int = 1
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
    content: str | None = None
    tool_calls: list[ToolCall] | None = None


class Choice(BaseModel):
    index: int
    message: ResponseMessage
    finish_reason: str | None = None


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


class ChunkDelta(BaseModel):
    role: str | None = None
    content: str | None = None
    tool_calls: list[ToolCall] | None = None


class ChunkChoice(BaseModel):
    index: int
    delta: ChunkDelta
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChunkChoice]
    # OpenAI contract (m9 D1): key OMITTED unless stream_options.include_usage,
    # then null on every chunk except the final usage chunk (choices: [])
    usage: Usage | None = None


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "kairyu"


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelCard]
