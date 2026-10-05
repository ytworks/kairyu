"""Contract gate for Kairyu's ``upstream: llamacpp`` profile (LCP-D2..D4).

llama.cpp's server HTTP API carries no stability guarantee (its semver covers
only the C API), so every pinned ``llama-server`` build is checked against the
profile before an example or deployment relies on it. The gate drives a
running server through ``OpenAICompatBackend(upstream="llamacpp")`` and, as
negative controls, sends the raw forms the adapter exists to avoid.

Start the server with explicit slots, per-slot context and Jinja, then run::

    uv run --frozen python -m verification run \
      l1.correctness.llamacpp_upstream_contract -- \
      --base-url http://127.0.0.1:8080 --model <alias> \
      --output llamacpp-contract.json --assert-gate

Without network access, ``synthetic-model`` writes a tiny random-weight GGUF
from a llama.cpp checkout's bundled Gemma 4 vocabulary and chat template. A
random model never chooses a tool call on its own, so pass
``--tool-call-bias 48:40`` (the ``<|tool_call>`` token) with it; the tool
rows then check which function the server's grammar admits, not model
quality. ``--record-fixtures DIR`` stores the exact wire request and the
server's raw response of selected rows for Kairyu's unit tests.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path

import httpx

from kairyu.engine.backend import GenerationRequest, UpstreamClientError
from kairyu.engine.openai_backend import OpenAICompatBackend
from kairyu.engine.prompt import (
    MultimodalItem,
    MultimodalMessage,
    MultimodalMessagePart,
    MultimodalPrompt,
)
from kairyu.sampling_params import GENERATION_CONFIG_SAMPLING_FIELDS, SamplingParams

SCHEMA_VERSION = 1
GATE_ID = "l1.correctness.llamacpp_upstream_contract"
_PROMPT = "Write one short sentence about rivers."
_TOOLS = (
    {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Look up a word.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "weather",
            "description": "Get the weather.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
)


class _RecordingTransport(httpx.AsyncBaseTransport):
    """Keep the last wire request and raw response body per label."""

    def __init__(self) -> None:
        self._inner = httpx.AsyncHTTPTransport()
        self.label: str | None = None
        self.records: dict[str, dict[str, object]] = {}

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        response = await self._inner.handle_async_request(request)
        if self.label is None:
            return response
        body = await response.aread()
        self.records[self.label] = {
            "path": request.url.path,
            "request": json.loads(request.content),
            "status": response.status_code,
            "response": body.decode(),
        }
        headers = [
            (name, value)
            for name, value in response.headers.multi_items()
            if name.lower() not in {"content-length", "transfer-encoding"}
        ]
        return httpx.Response(
            response.status_code,
            headers=headers,
            content=body,
            request=request,
        )

    async def aclose(self) -> None:
        await self._inner.aclose()


def _request(
    params: SamplingParams,
    *,
    explicit: frozenset[str] = frozenset(),
    **fields: object,
) -> GenerationRequest:
    """A request whose unset generation-config fields stay server defaults."""

    omitted = set(GENERATION_CONFIG_SAMPLING_FIELDS) - set(explicit)
    return GenerationRequest(
        request_id=f"contract-{time.monotonic_ns()}",
        prompt=fields.pop("prompt", _PROMPT),
        sampling_params=params.with_generation_config_omitted(omitted),
        **fields,
    )


def _webp_data_url() -> str:
    from PIL import Image

    output = BytesIO()
    Image.new("RGB", (64, 64), (200, 30, 30)).save(output, format="WEBP")
    return "data:image/webp;base64," + base64.b64encode(output.getvalue()).decode()


class Gate:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.root = args.base_url.rstrip("/")
        self.raw = httpx.AsyncClient(timeout=args.timeout_s, trust_env=False)
        self.recorder = _RecordingTransport()
        self.props: dict[str, object] = {}
        # Built from /props: llamacpp requires the per-slot context.
        self.backend: OpenAICompatBackend | None = None
        self.rows: list[dict[str, object]] = []

    def _backend(self, **options: object) -> OpenAICompatBackend:
        return OpenAICompatBackend(
            base_url=f"{self.root}/v1",
            model=self.args.model,
            api_key_env=None,
            timeout_s=self.args.timeout_s,
            upstream="llamacpp",
            max_model_len=int(self.props["default_generation_settings"]["n_ctx"]),
            **options,
        )

    def _tool_params(self, **fields: object) -> SamplingParams:
        extra: dict[str, object] = {}
        if self.args.tool_call_bias:
            token, bias = self.args.tool_call_bias.split(":")
            extra["logit_bias"] = [[int(token), float(bias)]]
        return SamplingParams(
            temperature=0.0,
            max_tokens=self.args.tool_max_tokens,
            extra_args=extra,
            **fields,
        )

    async def _raw_chat(self, body: dict[str, object]) -> httpx.Response:
        return await self.raw.post(
            f"{self.root}/v1/chat/completions",
            json={
                "model": self.args.model,
                "messages": [{"role": "user", "content": _PROMPT}],
                **body,
            },
        )

    async def row(self, row_id: str, check: Callable[[], Awaitable[dict]]) -> None:
        try:
            detail = await check()
            passed = bool(detail.pop("passed"))
        except Exception as error:  # a failed row must not hide later rows
            passed = False
            detail = {"error": f"{type(error).__name__}: {error}"}
        self.rows.append({"id": row_id, "passed": passed, "detail": detail})
        print(f"{'PASS' if passed else 'FAIL'} {row_id} {json.dumps(detail)[:240]}")

    async def server_identity(self) -> dict:
        health = await self.raw.get(f"{self.root}/health")
        props = (await self.raw.get(f"{self.root}/props")).json()
        self.props = props
        settings = props.get("default_generation_settings", {})
        detail = {
            "health": health.status_code,
            "build_info": props.get("build_info"),
            "total_slots": props.get("total_slots"),
            "slot_n_ctx": settings.get("n_ctx"),
            "supports_tool_calls": props.get("chat_template_caps", {}).get(
                "supports_tool_calls"
            ),
            "vision": props.get("modalities", {}).get("vision"),
        }
        commit = str(props.get("build_info", "")).rpartition("-")[2]
        detail["passed"] = (
            health.status_code == 200
            and detail["supports_tool_calls"] is True
            and (self.args.expect_commit is None or commit.startswith(self.args.expect_commit))
            and (self.args.expect_slots is None or detail["total_slots"] == self.args.expect_slots)
            and (self.args.expect_ctx is None or detail["slot_n_ctx"] == self.args.expect_ctx)
        )
        return detail

    async def repeat_penalty_executed(self) -> dict:
        explicit = frozenset({"temperature", "repetition_penalty"})
        base = SamplingParams(temperature=0.0, max_tokens=32, seed=1)
        baseline = await self.backend.generate(_request(base, explicit=explicit))
        self.recorder.label = "repeat_penalty"
        try:
            penalized = await self.backend.generate(
                _request(base.clone(repetition_penalty=3.0), explicit=explicit)
            )
        finally:
            self.recorder.label = None
        window = self.recorder.records["repeat_penalty"]["request"].get("repeat_last_n")
        raw = await self._raw_chat(
            {"temperature": 0.0, "max_tokens": 32, "seed": 1, "repetition_penalty": 3.0}
        )
        vllm_spelling = raw.json()["choices"][0]["message"]["content"]
        n_ctx = self.props["default_generation_settings"]["n_ctx"]
        return {
            # Executed, and over the whole sequence like Kairyu's sampler.
            "passed": penalized.text != baseline.text and window == n_ctx,
            "repeat_last_n": window,
            "vllm_spelling_ignored": vllm_spelling == baseline.text,
        }

    async def top_k_disabled_accepted(self) -> dict:
        result = await self.backend.generate(
            _request(
                SamplingParams(temperature=0.0, top_k=-1, max_tokens=4),
                explicit=frozenset({"temperature", "top_k"}),
            )
        )
        return {"passed": result.completions[0].finish_reason in {"stop", "length"}}

    async def named_tool_choice(self) -> dict:
        self.recorder.label = "named_tool_choice"
        try:
            result = await self.backend.generate(
                _request(
                    self._tool_params(),
                    explicit=frozenset({"temperature"}),
                    tools=_TOOLS,
                    tool_choice={"type": "function", "function": {"name": "weather"}},
                )
            )
        finally:
            self.recorder.label = None
        raw = await self._raw_chat(
            {
                "temperature": 0.0,
                "max_tokens": self.args.tool_max_tokens,
                "tools": list(_TOOLS),
                "tool_choice": {"type": "function", "function": {"name": "weather"}},
                **self._tool_params().extra_args,
            }
        )
        raw_calls = raw.json()["choices"][0]["message"].get("tool_calls") or []
        return {
            "passed": '<tool_call>{"name":"weather"' in result.text,
            "raw_object_called": [call["function"]["name"] for call in raw_calls],
        }

    async def logprobs_zero(self) -> dict:
        result = await self.backend.generate(
            _request(
                SamplingParams(temperature=0.0, max_tokens=3, logprobs=0),
                explicit=frozenset({"temperature"}),
            )
        )
        content = result.completions[0].logprob_content or ()
        raw = await self._raw_chat(
            {"temperature": 0.0, "max_tokens": 3, "logprobs": True, "top_logprobs": 0}
        )
        return {
            "passed": len(content) > 0 and all(item.top == () for item in content),
            "raw_top_logprobs_0_returned": raw.json()["choices"][0].get("logprobs") is not None,
        }

    async def assistant_prefill(self) -> dict:
        prefill = "The river"
        template = await self.raw.post(
            f"{self.root}/apply-template",
            json={
                "messages": [
                    {"role": "user", "content": _PROMPT},
                    {"role": "assistant", "content": prefill},
                ],
                "continue_final_message": True,
                "add_generation_prompt": False,
            },
        )
        result = await self.backend.generate(
            _request(
                SamplingParams(temperature=0.0, max_tokens=4),
                explicit=frozenset({"temperature"}),
                assistant_prefill=prefill,
            )
        )
        prompt = template.json().get("prompt", "")
        return {
            "passed": prompt.endswith(prefill) and result.completions[0].finish_reason is not None,
            "rendered_tail": prompt[-40:],
        }

    async def context_overflow_is_400(self) -> dict:
        n_ctx = int(self.props["default_generation_settings"]["n_ctx"])
        try:
            await self.backend.generate(
                _request(SamplingParams(max_tokens=2), prompt="river " * (2 * n_ctx))
            )
        except UpstreamClientError as error:
            return {"passed": error.status_code == 400 and "exceed" in str(error)}
        return {"passed": False}

    async def cached_tokens_reported(self) -> dict:
        prompt = "Rivers carry water to the sea. " * 32
        params = SamplingParams(temperature=0.0, max_tokens=2)
        explicit = frozenset({"temperature"})
        await self.backend.generate(_request(params, explicit=explicit, prompt=prompt))
        second = await self.backend.generate(_request(params, explicit=explicit, prompt=prompt))
        usage = second.usage
        return {
            "passed": usage is not None and usage.cached_tokens > 0,
            "cached_tokens": None if usage is None else usage.cached_tokens,
        }

    async def stream_usage_and_done(self) -> dict:
        self.recorder.label = "stream_tool_call"
        final = None
        try:
            async for partial in self.backend.stream(
                _request(
                    self._tool_params(),
                    explicit=frozenset({"temperature"}),
                    tools=_TOOLS[1:],
                    tool_choice="required",
                )
            ):
                final = partial
        finally:
            self.recorder.label = None
        response = str(self.recorder.records["stream_tool_call"]["response"])
        usage = None if final is None else final.usage
        return {
            "passed": final is not None
            and final.finished
            and usage is not None
            and usage.prompt_tokens > 0
            and response.rstrip().endswith("data: [DONE]")
            and '<tool_call>{"name":"weather"' in final.text,
            "finish_reason": None if final is None else final.completions[0].finish_reason,
        }

    async def cancel_on_disconnect(self) -> dict:
        request = _request(
            SamplingParams(temperature=0.8, max_tokens=4096, ignore_eos=True),
        )
        stream = self.backend.stream(request)
        seen = 0
        async for _partial in stream:
            seen += 1
            if seen >= 3:
                break
        await stream.aclose()
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            slots = (await self.raw.get(f"{self.root}/slots")).json()
            if not any(slot.get("is_processing") for slot in slots):
                return {"passed": True, "chunks_before_close": seen}
            await asyncio.sleep(0.25)
        return {"passed": False, "chunks_before_close": seen}

    async def webp_image(self) -> dict:
        if not self.props.get("modalities", {}).get("vision"):
            return {"passed": True, "skipped": "server has no vision projector"}
        url = _webp_data_url()
        prompt = MultimodalPrompt(
            base="describe",
            items=(MultimodalItem("image", "uri", url),),
            messages=(
                MultimodalMessage(
                    "user",
                    (
                        MultimodalMessagePart("item", item_index=0),
                        MultimodalMessagePart("text", text="Describe the image."),
                    ),
                ),
            ),
        )
        backend = self._backend(
            capabilities={"allow_prompt_kinds": ["multimodal"]},
            image_input_policy={"max_processed_prompt_tokens": 8192},
        )
        try:
            result = await backend.generate(
                _request(SamplingParams(temperature=0.0, max_tokens=4), prompt=prompt)
            )
        finally:
            await backend.shutdown()
        raw = await self.raw.post(
            f"{self.root}/v1/chat/completions",
            json={
                "model": self.args.model,
                "max_tokens": 4,
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "image_url", "image_url": {"url": url}}],
                    }
                ],
            },
        )
        return {
            "passed": result.usage is not None and result.usage.prompt_tokens > 0,
            "raw_webp_status": raw.status_code,
        }

    async def run(self) -> dict[str, object]:
        started = datetime.now(UTC)
        try:
            await self.row("server_identity", self.server_identity)
            if not self.props:
                raise RuntimeError("server /props is unavailable")
            extra = ["logit_bias"] if self.args.tool_call_bias else []
            self.backend = self._backend(
                transport=self.recorder,
                capabilities={"allow_extra_args": extra},
            )
            for row_id, check in (
                ("repeat_penalty_executed", self.repeat_penalty_executed),
                ("top_k_disabled_accepted", self.top_k_disabled_accepted),
                ("named_tool_choice", self.named_tool_choice),
                ("logprobs_zero", self.logprobs_zero),
                ("assistant_prefill", self.assistant_prefill),
                ("context_overflow_is_400", self.context_overflow_is_400),
                ("cached_tokens_reported", self.cached_tokens_reported),
                ("stream_usage_and_done", self.stream_usage_and_done),
                ("cancel_on_disconnect", self.cancel_on_disconnect),
                ("webp_image", self.webp_image),
            ):
                await self.row(row_id, check)
        finally:
            if self.backend is not None:
                await self.backend.shutdown()
            await self.recorder.aclose()
            await self.raw.aclose()
        if self.args.record_fixtures:
            self._write_fixtures(Path(self.args.record_fixtures))
        return {
            "schema_version": SCHEMA_VERSION,
            "gate": GATE_ID,
            "started_at": started.isoformat(),
            "completed_at": datetime.now(UTC).isoformat(),
            "server": {
                "build_info": self.props.get("build_info"),
                "total_slots": self.props.get("total_slots"),
                "slot_n_ctx": self.props.get("default_generation_settings", {}).get("n_ctx"),
                "model_path": self.props.get("model_path"),
            },
            "rows": self.rows,
            "passed": all(row["passed"] for row in self.rows),
        }

    def _write_fixtures(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        for label, record in sorted(self.recorder.records.items()):
            payload = {"build_info": self.props.get("build_info"), **record}
            path = directory / f"{label}.json"
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
            print(f"recorded {path}")


def _synthetic_model(llama_cpp: Path, output: Path) -> None:
    """Write a tiny random-weight qwen2-architecture GGUF with Gemma 4 vocab."""

    import numpy as np

    sys.path.insert(0, str(llama_cpp / "gguf-py"))
    from gguf import GGUFReader, GGUFValueType, GGUFWriter

    vocab = GGUFReader(llama_cpp / "models/ggml-vocab-gemma-4.gguf")
    n_vocab = len(vocab.fields["tokenizer.ggml.tokens"].data)
    n_embd, n_head, n_head_kv, n_ff, n_layer = 64, 4, 2, 128, 2
    kv_dim = n_head_kv * (n_embd // n_head)
    writer = GGUFWriter(str(output), "qwen2")
    writer.add_name("kairyu-contract-tiny")
    writer.add_block_count(n_layer)
    writer.add_context_length(4096)
    writer.add_embedding_length(n_embd)
    writer.add_feed_forward_length(n_ff)
    writer.add_head_count(n_head)
    writer.add_head_count_kv(n_head_kv)
    writer.add_rope_freq_base(1000000.0)
    writer.add_layer_norm_rms_eps(1e-6)
    writer.add_file_type(0)
    for key, field in vocab.fields.items():
        if not key.startswith("tokenizer."):
            continue
        value_type = field.types[0]
        if value_type == GGUFValueType.ARRAY:
            writer.add_array(key, field.contents())
        elif value_type == GGUFValueType.STRING:
            writer.add_string(key, field.contents())
        elif value_type == GGUFValueType.BOOL:
            writer.add_bool(key, field.contents())
        else:
            writer.add_uint32(key, int(field.contents()))
    rng = np.random.default_rng(0)

    def random(name: str, *shape: int) -> None:
        writer.add_tensor(name, (rng.standard_normal(shape) * 0.02).astype(np.float32))

    random("token_embd.weight", n_vocab, n_embd)
    writer.add_tensor("output_norm.weight", np.ones(n_embd, dtype=np.float32))
    for layer in range(n_layer):
        prefix = f"blk.{layer}"
        writer.add_tensor(f"{prefix}.attn_norm.weight", np.ones(n_embd, dtype=np.float32))
        writer.add_tensor(f"{prefix}.ffn_norm.weight", np.ones(n_embd, dtype=np.float32))
        random(f"{prefix}.attn_q.weight", n_embd, n_embd)
        random(f"{prefix}.attn_q.bias", n_embd)
        random(f"{prefix}.attn_k.weight", kv_dim, n_embd)
        random(f"{prefix}.attn_k.bias", kv_dim)
        random(f"{prefix}.attn_v.weight", kv_dim, n_embd)
        random(f"{prefix}.attn_v.bias", kv_dim)
        random(f"{prefix}.attn_output.weight", n_embd, n_embd)
        random(f"{prefix}.ffn_gate.weight", n_ff, n_embd)
        random(f"{prefix}.ffn_up.weight", n_ff, n_embd)
        random(f"{prefix}.ffn_down.weight", n_embd, n_ff)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command")
    synthetic = commands.add_parser("synthetic-model", help="write a tiny offline GGUF")
    synthetic.add_argument("--llama-cpp", type=Path, required=True)
    synthetic.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default="model")
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--tool-call-bias", help="TOKEN_ID:BIAS for a random-weight model")
    parser.add_argument(
        "--tool-max-tokens",
        type=int,
        default=24,
        help="output budget of the tool rows (a real model may write text first)",
    )
    parser.add_argument("--expect-commit", help="build_info commit prefix the server must report")
    parser.add_argument("--expect-slots", type=int)
    parser.add_argument("--expect-ctx", type=int, help="per-slot context the server must report")
    parser.add_argument("--record-fixtures", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--assert-gate", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "synthetic-model":
        _synthetic_model(args.llama_cpp, args.output)
        print(args.output)
        return 0
    report = asyncio.run(Gate(args).run())
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    print(f"{'PASS' if report['passed'] else 'FAIL'} {GATE_ID}")
    return 1 if args.assert_gate and not report["passed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
