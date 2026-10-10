#!/usr/bin/env python3
"""System One (Jev wire API) for Quyet-1.0-Large, read through the vLLM that serves chat.

Quyet's own runtime (``quyet`` 1.0.2) answers a decision with one forward pass per
question: the options are lettered A..J in a fixed chat prompt, and the answer is
the softmax of those letters' next-token logits at a calibrated temperature. The
package has no server, and its transformers forward pass would need a second copy
of the 62.5 GB weights next to the vLLM that serves chat on the same GPU.

This server keeps the package's own code for everything but that forward pass:
``QuyetOnVllm`` subclasses ``quyet.llm.runtime.LLMModel`` and replaces only
``__init__`` (tokenizer and config, no torch model) and ``_letter_logits``. vLLM's
``/v1/completions`` reads the letters' logprobs for the exact prompt token IDs the
package built. A softmax over logprobs equals a softmax over logits (the
vocabulary's normaliser cancels), so the package's calibration applies unchanged.
OpenJev serves JevK5 the same way.

The wire contract follows OpenJev's Jev-compatible server: 422 with a validation
list for a body of the wrong shape, 400 for a question the model cannot ask or an
option it does not have (images, think, samples, steps, sequential), 529 when the
queue is full, 503 when vLLM is unavailable, and a ``Server-Timing`` header.

Each answered read logs one ``read`` line (questions, input tokens, whether the
state was cut at Quyet's 6,000 tokens): the example's gates count cut states from it.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, Literal

import httpx
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

log = logging.getLogger("quyet-systemone")

# The adapter relies on LLMModel's attribute layout and on _run calling
# _letter_logits once per request; both are this exact release's.
SUPPORTED_QUYET = "1.0.2"
SDK_ALIASES = ("jev-latest", "jev-preview")
WARMUP_QUESTIONS = {
    "n": {"type": "noul", "instructions": "The message is a greeting."},
    "c": {"type": "choice", "instructions": "Which word is it?", "criteria": ["hello", "bye"]},
    "s": {"type": "score", "instructions": "How formal is it?", "criteria": ["casual", "formal"]},
}


def _env(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value or default


class Settings:
    def __init__(self) -> None:
        self.model_dir = Path(_env("QUYET_MODEL_DIR", "/models/quyet-1.0-large"))
        self.upstream = _env("QUYET_UPSTREAM", "http://vllm:8000").rstrip("/")
        self.upstream_model = _env("QUYET_UPSTREAM_MODEL", "quyet-1.0-large")
        self.aliases = tuple(a for a in _env("QUYET_ALIASES", "quyet-latest").split(",") if a)
        self.max_inflight = int(_env("QUYET_MAX_INFLIGHT", "16"))
        self.max_queue = int(_env("QUYET_MAX_QUEUE", "64"))
        self.read_workers = int(_env("QUYET_READ_WORKERS", "32"))
        self.max_questions = int(_env("QUYET_MAX_QUESTIONS", "32"))
        self.max_body_bytes = int(_env("QUYET_MAX_BODY_BYTES", str(8 * 1024 * 1024)))
        self.timeout_s = float(_env("QUYET_TIMEOUT_S", "300"))
        self.startup_timeout_s = float(_env("QUYET_STARTUP_TIMEOUT_S", "3600"))
        if self.max_inflight < 1 or self.max_queue < 0 or self.read_workers < 1:
            raise ValueError(
                "QUYET_MAX_INFLIGHT and QUYET_READ_WORKERS must be >= 1, QUYET_MAX_QUEUE >= 0"
            )


class UpstreamRefused(RuntimeError):
    """vLLM answered a read with a 4xx: the request cannot be read as asked."""


class Overloaded(RuntimeError):
    pass


class VllmReader:
    """The letters' logprobs at the answer position, one vLLM completion per question."""

    def __init__(self, settings: Settings, transport: httpx.BaseTransport | None = None) -> None:
        self.model = settings.upstream_model
        self.http = httpx.Client(
            base_url=settings.upstream,
            timeout=httpx.Timeout(settings.timeout_s, connect=5.0),
            limits=httpx.Limits(max_connections=settings.read_workers),
            transport=transport,
        )
        self.pool = ThreadPoolExecutor(settings.read_workers, thread_name_prefix="quyet-read")

    def close(self) -> None:
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.http.close()

    def letter_logprobs(self, ids: list[int], letter_ids: list[int]) -> list[float]:
        response = self.http.post(
            "/v1/completions",
            json={
                "model": self.model,
                "prompt": ids,
                "max_tokens": 1,
                "temperature": 0,
                "logprobs": 1,
                "logprob_token_ids": letter_ids,
                "return_tokens_as_token_ids": True,
            },
        )
        if 400 <= response.status_code < 500:
            raise UpstreamRefused(response.text[:500])
        response.raise_for_status()
        try:
            body = response.json()
            usage = body["usage"]["prompt_tokens"]
            step = body["choices"][0]["logprobs"]["top_logprobs"][0]
            by_id = {int(key.rsplit(":", 1)[1]): value for key, value in step.items()}
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise RuntimeError(f"unexpected vLLM completion: {response.text[:300]}") from error
        if usage != len(ids):
            # The package bills and truncates by these IDs; vLLM must read exactly them.
            raise RuntimeError(f"vLLM read {usage} prompt tokens, the package built {len(ids)}")
        missing = [i for i in letter_ids if i not in by_id]
        if missing:
            raise RuntimeError(f"vLLM returned no logprob for letter token IDs {missing}")
        return [float(by_id[i]) for i in letter_ids]


@functools.cache
def quyet_on_vllm_class():
    """The package's LLMModel with the transformers forward pass replaced by vLLM."""

    from quyet.llm import prompt as P
    from quyet.llm.runtime import LLMModel

    class QuyetOnVllm(LLMModel):
        def __init__(self, path: Path, config: dict, reader: VllmReader, tokenizer=None) -> None:
            # The fields LLMModel.__init__ sets and _run/_prompt_ids read, minus the torch model.
            if tokenizer is None:
                from transformers import AutoTokenizer

                tokenizer = AutoTokenizer.from_pretrained(str(path))
            self.path, self.config = Path(path), config
            self.name = config["name"]
            self.tok = tokenizer
            self.letter_ids = P.letter_ids(self.tok)
            self.temps = dict(config.get("temperatures") or {})
            self.prompt_version = P.check_version(int(config.get("prompt_version", 1)))
            limits = config["limits"]
            self.max_state_tokens = int(limits["max_state_tokens"])
            self.max_prompt_tokens = int(limits["max_prompt_tokens"])
            self.min_state_tokens = int(limits["min_state_tokens"])
            self.reader = reader
            self._timing = threading.local()

        def _letter_logits(self, seqs):
            """seqs: [(prompt IDs, option count)] -> letter logprobs, read in parallel."""

            started = time.perf_counter()
            out = list(
                self.reader.pool.map(
                    lambda seq: self.reader.letter_logprobs(seq[0], self.letter_ids[: seq[1]]), seqs
                )
            )
            self._timing.model_s = time.perf_counter() - started
            return out

        def predict_timed(self, state, questions) -> tuple[dict, float]:
            self._timing.model_s = 0.0
            return self.predict(state, questions), self._timing.model_s

    return QuyetOnVllm


class QuyetRuntime:
    """The installed quyet package: its version, error type, model config and model."""

    def __init__(self, model_dir: Path) -> None:
        import quyet
        from quyet.errors import QuestionError
        from quyet.hub import read_config

        if quyet.__version__ != SUPPORTED_QUYET:
            raise RuntimeError(
                f"quyet {quyet.__version__} is installed; this adapter is for {SUPPORTED_QUYET}"
            )
        self.version = quyet.__version__
        self.question_error = QuestionError
        self.config = read_config(model_dir)
        if self.config.get("kind") != "llm":
            raise RuntimeError(f"{model_dir} is a {self.config.get('kind')!r} model, not an LLM")

    def build(self, settings: Settings):
        reader = VllmReader(settings)
        return quyet_on_vllm_class()(settings.model_dir, self.config, reader), reader


# --- Jev request shape (as OpenJev and Jev accept it) ------------------------------------

JSONContent = str | dict[str, Any] | list[Any]
Described = str | dict[str, Any] | list[Any] | None


class NoulCriteria(BaseModel):
    true: Described = None
    false: Described = None


class NoulQuestion(BaseModel):
    type: Literal["noul"]
    instructions: Described = None
    criteria: NoulCriteria | None = None


class ChoiceQuestion(BaseModel):
    type: Literal["choice"]
    instructions: Described = None
    criteria: dict[str, Described] | list[JSONContent]


class ScoreQuestion(BaseModel):
    type: Literal["score"]
    instructions: Described = None
    criteria: list[JSONContent] = Field(min_length=1)


Question = Annotated[NoulQuestion | ChoiceQuestion | ScoreQuestion, Field(discriminator="type")]


class SystemOneRequest(BaseModel):
    state: JSONContent
    model: str
    questions: dict[str, Question] = Field(min_length=1)
    images: list[Any] | None = None
    steps: int | None = Field(default=None, ge=1, le=8)
    samples: int | None = Field(default=None, ge=1, le=32)
    think: int | None = Field(default=None, ge=0, le=4096)
    sequential: bool | None = None


def unsupported_option(req: SystemOneRequest) -> str | None:
    """The first Jev option this model cannot honour; their neutral values are accepted."""

    for name, used in (
        ("images", bool(req.images)),
        ("steps", (req.steps or 1) > 1),
        ("samples", (req.samples or 1) > 1),
        ("think", bool(req.think)),
        ("sequential", bool(req.sequential)),
    ):
        if used:
            return name
    return None


def _jev_error(status: int, error_type: str, message: str, headers: dict | None = None):
    return JSONResponse(
        {"detail": {"error_type": error_type, "message": message}},
        status_code=status,
        headers=headers,
    )


def _semantic_error(message: str) -> JSONResponse:
    # Jev's shape for a request whose shape is fine but whose meaning is not.
    return JSONResponse({"detail": message}, status_code=400)


def create_app(settings: Settings | None = None, *, runtime=None) -> FastAPI:
    """``runtime`` provides version, question_error, config and build(settings) -> (model,
    reader); it defaults to the installed quyet package, and tests pass a fake one."""

    settings = settings or Settings()
    runtime = runtime or QuyetRuntime(settings.model_dir)
    config = runtime.config
    model_version = config["name"]
    names = {model_version, *settings.aliases, *SDK_ALIASES}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        model, reader = await asyncio.to_thread(runtime.build, settings)
        app.state.model, app.state.reader = model, reader
        app.state.requests = ThreadPoolExecutor(
            settings.max_inflight, thread_name_prefix="quyet-req"
        )
        app.state.slots = asyncio.Semaphore(settings.max_inflight)
        app.state.waiting = 0
        await asyncio.to_thread(_wait_for_upstream, settings, reader)
        # The first read proves the whole path before /health answers.
        answer = await asyncio.to_thread(model.predict, "Hello there!", WARMUP_QUESTIONS)
        if set(answer["answers"]) != set(WARMUP_QUESTIONS):
            raise RuntimeError(f"warmup read returned {answer!r}")
        log.info("ready: %s on %s (%s)", model_version, settings.upstream, settings.upstream_model)
        yield
        app.state.requests.shutdown(wait=False, cancel_futures=True)
        reader.close()

    app = FastAPI(title="Quyet System One", version=runtime.version, lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def invalid_body(request: Request, exc: RequestValidationError):
        if any(e.get("type") == "union_tag_invalid" for e in exc.errors()):
            # an unknown question type, which Jev answers generically
            return _jev_error(400, "api_usage_error", "Invalid request.")
        errors = [
            {"type": e.get("type"), "loc": list(e.get("loc", ())), "msg": e.get("msg")}
            for e in exc.errors()
        ]
        return JSONResponse({"detail": errors}, status_code=422)

    @app.middleware("http")
    async def body_limit_and_timing(request: Request, call_next):
        if request.url.path.startswith("/v1/"):
            declared = request.headers.get("content-length")
            if (
                declared is not None
                and declared.isdigit()
                and int(declared) > settings.max_body_bytes
            ):
                return _jev_error(
                    413, "request_too_large", f"body over {settings.max_body_bytes} bytes"
                )
        request.state.model_s = 0.0
        started = time.perf_counter()
        response = await call_next(request)
        total_ms = (time.perf_counter() - started) * 1000
        model_ms = request.state.model_s * 1000
        response.headers["server-timing"] = (
            f"model;dur={model_ms:.1f}, server;dur={max(0.0, total_ms - model_ms):.1f}, "
            f"total;dur={total_ms:.1f}"
        )
        return response

    @app.get("/health")
    async def health():
        try:
            response = await asyncio.to_thread(app.state.reader.http.get, "/health", timeout=3.0)
            healthy = response.status_code == 200
        except httpx.HTTPError:
            healthy = False
        if not healthy:
            return _jev_error(503, "api_error", "vLLM is unavailable", {"retry-after": "2"})
        model = app.state.model
        return {
            "status": "ok",
            "model": model_version,
            "quyet": runtime.version,
            "prompt_version": model.prompt_version,
            "temperatures": model.temps,
            "limits": config["limits"],
            "letter_ids": model.letter_ids,
            "upstream_model": settings.upstream_model,
        }

    @app.get("/v1/models")
    async def models():
        return {
            "models": [
                {
                    "name": model_version,
                    "description": "Quyet-1.0-Large by Chinh Nguyen (Apache-2.0): Gemma-4-31B-it "
                    "with a merged decision LoRA, read through vLLM with the quyet package's "
                    "prompt and calibration. "
                    "Text only, 6,000 state tokens, up to 10 options.",
                }
            ]
        }

    @app.post("/v1/systemone")
    async def systemone(req: SystemOneRequest, request: Request):
        if req.model not in names:
            return _jev_error(400, "api_usage_error", f"Unknown model: {req.model}")
        if len(req.questions) > settings.max_questions:
            return _semantic_error(f"at most {settings.max_questions} questions per request")
        option = unsupported_option(req)
        if option is not None:
            return _semantic_error(f"{model_version} does not support {option}")
        questions = {qid: q.model_dump() for qid, q in req.questions.items()}
        try:
            body, model_s = await _admit_and_read(app, settings, req.state, questions)
        except runtime.question_error as error:
            return _semantic_error(str(error))
        except UpstreamRefused as error:
            return _semantic_error(f"the model rejected this request: {error}")
        except Overloaded as error:
            return _jev_error(529, "overloaded_error", str(error), {"retry-after": "1"})
        except (httpx.HTTPError, RuntimeError) as error:
            log.warning("read failed: %s", error)
            return _jev_error(
                503,
                "api_error",
                f"inference backend unavailable: {type(error).__name__}",
                {"retry-after": "2"},
            )
        request.state.model_s = model_s
        cut = [w for w in body.get("warnings") or [] if w.get("code") == "state_truncated"]
        log.info(
            "read questions=%d input_tokens=%d truncated=%s state_tokens=%s model_ms=%.0f",
            len(questions),
            body["usage"]["input_tokens"],
            bool(cut),
            cut[0]["state_tokens"] if cut else "-",
            model_s * 1000,
        )
        return body

    return app


async def _admit_and_read(app: FastAPI, settings: Settings, state, questions) -> tuple[dict, float]:
    if app.state.slots.locked() and app.state.waiting >= settings.max_queue:
        raise Overloaded(f"{settings.max_inflight} reads running and {settings.max_queue} waiting")
    app.state.waiting += 1
    try:
        await app.state.slots.acquire()
    finally:
        app.state.waiting -= 1
    try:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            app.state.requests, app.state.model.predict_timed, state, questions
        )
    finally:
        app.state.slots.release()


def _wait_for_upstream(settings: Settings, reader: VllmReader) -> None:
    deadline = time.monotonic() + settings.startup_timeout_s
    while True:
        try:
            response = reader.http.get("/v1/models", timeout=5.0)
            if response.status_code == 200:
                served = {row["id"] for row in response.json().get("data", [])}
                if settings.upstream_model not in served:
                    raise RuntimeError(
                        f"vLLM serves {sorted(served)}, not {settings.upstream_model!r}"
                    )
                return
        except httpx.HTTPError:
            pass
        if time.monotonic() > deadline:
            raise RuntimeError(f"vLLM at {settings.upstream} did not become ready")
        time.sleep(5)


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    uvicorn.run(
        create_app(), host="0.0.0.0", port=int(_env("QUYET_PORT", "8092")), log_level="info"
    )


if __name__ == "__main__":
    main()
