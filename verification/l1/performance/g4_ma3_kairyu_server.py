#!/usr/bin/env python3
"""Pinned production Kairyu server for the G4 M-A3 formal comparison.

This launcher intentionally exposes only the diagnostic pipeline-depth choice
and the HTTP bind address.  Model geometry, parallelism, CUDA-graph bounds,
cache capacity, and scheduling limits are part of the formal operator and
cannot be changed from the command line.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]
if __package__ in {None, ""}:
    sys.path.insert(0, str(_REPO_ROOT))

import uvicorn  # noqa: E402
from fastapi import FastAPI  # noqa: E402

from kairyu.engine.kairyu_backend import KairyuBackend  # noqa: E402
from kairyu.entrypoints.server.app import create_app  # noqa: E402
from kairyu.entrypoints.server.settings import ServerSettings  # noqa: E402

SCHEMA_VERSION = "kairyu.g4.ma3.kairyu-server.v1"
MODEL_PATH = "/models/qwen3-235b-nvfp4"
SERVED_MODEL_NAME = MODEL_PATH
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 30_000
PIPELINE_DEPTHS = (1, 5)


def _port(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("port must be an integer") from error
    if not 1 <= parsed <= 65_535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return parsed


def _host(value: str) -> str:
    if not value or value.strip() != value or any(character.isspace() for character in value):
        raise argparse.ArgumentTypeError("host must be one non-empty value without whitespace")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="g4-ma3-kairyu-server",
        description="Run the pinned Kairyu G4 M-A3 comparison server.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--pipeline-depth",
        type=int,
        choices=PIPELINE_DEPTHS,
        required=True,
        help="Diagnostic depth; the formal comparison is pinned to depth 5.",
    )
    parser.add_argument("--host", type=_host, default=DEFAULT_HOST)
    parser.add_argument("--port", type=_port, default=DEFAULT_PORT)
    return parser


def backend_kwargs(pipeline_depth: int) -> dict[str, Any]:
    """Return the complete immutable engine contract for one diagnostic arm."""

    if pipeline_depth not in PIPELINE_DEPTHS:
        raise ValueError(f"pipeline_depth must be one of {PIPELINE_DEPTHS}")
    return {
        "model_path": MODEL_PATH,
        "tokenizer": MODEL_PATH,
        "tensor_parallel_size": 1,
        "expert_parallel_size": 4,
        "expert_parallel_attention_dp": True,
        "pipeline_depth": pipeline_depth,
        "decode_mode": "cuda_graph",
        "cuda_graph_max_batch": 32,
        "cuda_graph_max_pages": 128,
        "cuda_graph_warmup_iters": 3,
        "num_pages": 4096,
        "page_size": 16,
        "max_num_batched_tokens": 8192,
        "max_num_seqs": 256,
        "max_model_len": 2048,
        "priority_age_s": None,
        "kv_cache_dtype": "bfloat16",
    }


def resolved_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "served_model_name": SERVED_MODEL_NAME,
        "host": args.host,
        "port": args.port,
        "access_log": False,
        **backend_kwargs(args.pipeline_depth),
    }


def build_application(args: argparse.Namespace) -> FastAPI:
    """Construct the GPU-backed app only after CLI validation has completed."""

    backend = KairyuBackend(**backend_kwargs(args.pipeline_depth))

    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            await backend.shutdown()

    return create_app(
        engines={SERVED_MODEL_NAME: backend},
        settings=ServerSettings(access_log=False),
        lifespan=lifespan,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    print(
        json.dumps(
            resolved_config(args),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ),
        flush=True,
    )
    app = build_application(args)
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        loop="uvloop" if sys.platform == "linux" else "auto",
        http="httptools" if sys.platform == "linux" else "auto",
        log_config=None,
        access_log=False,
    )


if __name__ == "__main__":
    main()
