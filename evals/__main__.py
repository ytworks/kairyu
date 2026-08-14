"""Checkout-only model-evaluation command line."""

from __future__ import annotations

import argparse
import sys

from evals.cli import add_evals_parser, handle


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evals",
        description="Run Kairyu repository model evaluations.",
    )
    add_evals_parser(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = _build_parser().parse_args(arguments)
    return handle(args)


if __name__ == "__main__":
    raise SystemExit(main())
