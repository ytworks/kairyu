"""Schemas for the Fugu benchmark suite: config, items, per-pair results.

Everything is a frozen pydantic model (repo convention, m7 D3): configs are
loaded from YAML/CLI once and never mutated; results are written atomically
and re-read for resume, so round-tripping through JSON must be lossless.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = 1


class BenchTarget(BaseModel):
    """One scoreboard column: a model name on an OpenAI-compatible endpoint.

    Single models and orchestrations are both just model names ("llama-70b"
    vs "kairyu-auto-max") — usually on the same gateway base_url.
    """

    model_config = ConfigDict(frozen=True)

    name: str = ""  # scoreboard label; defaults to model
    base_url: str
    model: str
    api_key_env: str | None = None  # env var NAME, never the key itself
    max_context_tokens: int | None = None  # gate for long-context items
    max_output_tokens: int = 8192
    supports_vision: bool = True

    def label(self) -> str:
        return self.name or self.model


class JudgeConfig(BaseModel):
    """LLM judge endpoint (any OpenAI-compatible server, incl. kairyu itself)."""

    model_config = ConfigDict(frozen=True)

    base_url: str | None = None
    model: str | None = None
    api_key_env: str = "KAIRYU_JUDGE_API_KEY"
    concurrency: int = Field(default=4, ge=1)
    max_retries: int = Field(default=3, ge=0)

    @property
    def enabled(self) -> bool:
        return self.base_url is not None and self.model is not None


class BenchConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    suite: str = "fugu"
    targets: tuple[BenchTarget, ...] = Field(min_length=1)
    judge: JudgeConfig = JudgeConfig()
    limit: int | None = Field(default=None, ge=1)  # None = full dataset
    smoke: bool = False  # preset: limit<=SMOKE_LIMIT, halved output budget
    offline_fixtures: bool = False  # read committed fixtures, no cache/network
    only: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()
    seed: int = 0
    concurrency: int = Field(default=8, ge=1)  # in-flight requests per pair
    request_timeout_s: float = Field(default=600.0, gt=0)
    retries: int = Field(default=2, ge=0)
    cache_dir: str | None = None
    results_dir: str = "bench/results/fugu"
    run_id: str | None = None  # reuse an id to resume
    rerun: bool = False  # ignore existing pair results
    download: bool = True  # auto-download missing datasets before running


SMOKE_LIMIT = 20


class BenchItem(BaseModel):
    """One dataset row; payload is adapter-private normalized fields."""

    model_config = ConfigDict(frozen=True)

    id: str
    payload: dict


class ChatRequestSpec(BaseModel):
    """OpenAI wire-format request an adapter built for one item."""

    model_config = ConfigDict(frozen=True)

    messages: tuple[dict, ...]
    max_tokens: int | None = None
    temperature: float = 0.0
    est_prompt_tokens: int | None = None  # chars/4 heuristic, for context gating


class SkipItem(BaseModel):
    """build_request() verdict: this item cannot run against this target."""

    model_config = ConfigDict(frozen=True)

    reason: str


ItemStatus = Literal["completed", "failed", "unjudged", "skipped"]


class ItemResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    item_id: str
    status: ItemStatus
    score: float | None = None  # 0..1
    response_excerpt: str | None = None  # capped, evidence only
    error: str | None = None
    judge: dict | None = None  # {model, verdict, raw_excerpt} when judged
    latency_s: float | None = None


PairStatus = Literal["completed", "partial", "skipped", "failed"]


class PairResult(BaseModel):
    """One scoreboard cell: one benchmark run against one target."""

    model_config = ConfigDict(frozen=True)

    schema_version: int = SCHEMA_VERSION
    benchmark: str
    target: str
    status: PairStatus
    reason: str | None = None  # "docker unavailable", "dataset unavailable", ...
    # {"score":…, "n_total":…, "n_scored":…, "n_unjudged":…, "n_skipped":…, "n_failed":…}
    metrics: dict[str, float | int | None] = Field(default_factory=dict)
    items: tuple[ItemResult, ...] = ()  # per-item evidence (roadmap §6)
    methodology: dict = Field(default_factory=dict)
    annotations: tuple[str, ...] = ()
    started_at: str = ""
    finished_at: str = ""

    @property
    def score(self) -> float | None:
        value = self.metrics.get("score")
        return float(value) if value is not None else None


class DownloadReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    adapter: str
    status: Literal["ok", "cached", "gated", "unavailable", "extras_missing"]
    detail: str = ""


class BenchExtrasMissing(RuntimeError):
    """Raised when an optional dependency group is required but not installed."""

    def __init__(self, extra: str, purpose: str) -> None:
        super().__init__(
            f"{purpose} requires the [{extra}] extra: pip install 'kairyu[{extra}]'"
        )
        self.extra = extra


class DatasetGated(RuntimeError):
    """Raised when a HF dataset needs license acceptance + token."""

    def __init__(self, dataset: str) -> None:
        super().__init__(
            f"dataset {dataset!r} is gated: accept the license at "
            f"https://huggingface.co/datasets/{dataset} and set HF_TOKEN"
        )
        self.dataset = dataset


class DatasetUnavailable(RuntimeError):
    """Raised when a dataset cannot be fetched (missing repo, network, ...)."""
