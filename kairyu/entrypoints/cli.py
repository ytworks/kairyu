"""`kairyu` console entrypoint: `serve` (design m7 D3) and `bench` (goal G6 P-C1)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from kairyu.bench.cli import add_bench_parser
from kairyu.deploy.builder import build_app_from_config
from kairyu.deploy.spec import load_deployment_spec
from kairyu.entrypoints.server.middleware import configure_json_logging


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kairyu", description="Kairyu serving CLI"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    serve = subparsers.add_parser(
        "serve",
        help="Run the OpenAI-compatible server from a DeploymentSpec YAML "
        "(gateway or replica role — the config decides).",
    )
    serve.add_argument("config", type=Path, help="Path to a DeploymentSpec YAML")
    serve.add_argument("--host", default=None, help="Override server.host")
    serve.add_argument("--port", type=int, default=None, help="Override server.port")
    add_bench_parser(subparsers)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.command == "serve":
        import uvicorn

        configure_json_logging()
        spec = load_deployment_spec(args.config)
        app = build_app_from_config(args.config)
        uvicorn.run(
            app,
            host=args.host or spec.server.host,
            port=args.port or spec.server.port,
            log_config=None,  # keep the JSON root logger
        )
    elif args.command == "bench":
        from kairyu.bench.cli import handle

        sys.exit(handle(args))


if __name__ == "__main__":
    main()
