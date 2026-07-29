"""``kairyu`` console entrypoint: serve, validate, and benchmark commands."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from kairyu.bench.cli import add_bench_parser


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
    validate = subparsers.add_parser(
        "validate",
        help="Validate a DeploymentSpec and its local linked artifacts without "
        "starting a server.",
    )
    validate.add_argument("config", type=Path, help="Path to a DeploymentSpec YAML")
    add_bench_parser(subparsers)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.command == "serve":
        import uvicorn

        from kairyu.deploy.builder import build_app_from_config
        from kairyu.deploy.spec import load_deployment_spec
        from kairyu.entrypoints.server.middleware import configure_json_logging

        configure_json_logging()
        spec = load_deployment_spec(args.config)
        app = build_app_from_config(args.config)
        uvicorn.run(
            app,
            host=args.host or spec.server.host,
            port=args.port or spec.server.port,
            log_config=None,  # keep the JSON root logger
            # AccessLogMiddleware is the single structured access-log owner.
            # Leaving Uvicorn's logger enabled duplicates every request and
            # makes ServerSettings.access_log=False ineffective.
            access_log=False,
        )
    elif args.command == "validate":
        from kairyu.deploy.validation import validate_deployment

        report = validate_deployment(args.config)
        print(report.render_text())
        if not report.valid:
            sys.exit(1)
    elif args.command == "bench":
        from kairyu.bench.cli import handle

        sys.exit(handle(args))


if __name__ == "__main__":
    main()
