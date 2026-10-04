"""Kairyu server for the Open Responses compliance suite (M20 WP-14, D1).

``python -m scripts.open_responses_gate.launcher --port PORT --workdir DIR``
serves ``tests/contracts/openresponses/kairyu-mock.yaml`` with ``SCRIPT`` as
its ScenarioBackend, so every HTTP scenario of the pinned suite gets a
deterministic answer: text for the conversation scenarios and a
``get_weather`` function call for ``tool-calling``. The suite runs its
scenarios concurrently, so turns are chosen by the scenario's user text
(``rules``), never by call order.

No turn carries reasoning. Kairyu emits OpenAI's ``response.reasoning_text.*``
events (Codex and the SDKs read them), which the pinned suite's event union
names ``response.reasoning.*`` (openresponses PR #42), so the streaming
scenario runs with reasoning off; ``docs/design/m20-open-responses-ci.md``
records the divergence. Importing this module registers nothing.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import yaml

from tests.support.scenario_script import Scenario, ToolCall, Turn

ROOT = Path(__file__).resolve().parents[2]
DEPLOYMENT = ROOT / "tests" / "contracts" / "openresponses" / "kairyu-mock.yaml"
BACKEND = "open-responses-scenario"
MODEL = "kairyu-mock"

# Keys are distinctive substrings of the suite's user messages
# (src/lib/compliance-tests.ts at the pin); everything else gets ``default``.
SCRIPT = Scenario(
    rules={
        "weather like in San Francisco": Turn(
            ToolCall.of("get_weather", location="San Francisco, CA")
        ),
        "Count from 1 to 5.": Turn("1, 2, 3, 4, 5"),
        "Count from 1 to 3.": Turn("1, 2, 3"),
        "Repeat only the number.": Turn("Four."),
        "What is my name?": Turn("Your name is Alice."),
        "Say hello.": Turn("Ahoy, matey!"),
    },
    default=Turn("Hello there, friend."),
)


def deployment(host: str, port: int) -> dict:
    """The gate's DeploymentSpec mapping, bound to ``host:port``."""

    spec = yaml.safe_load(DEPLOYMENT.read_text(encoding="utf-8"))
    server = {**spec["server"], "host": host, "port": port}
    return {**spec, "server": server}


def build_app(host: str, port: int, workdir: Path):
    from kairyu.deploy.builder import build_app_from_spec
    from kairyu.deploy.spec import DeploymentSpec
    from tests.support.scenario_backend import register_scenario_backend

    register_scenario_backend(SCRIPT, name=BACKEND)
    spec = DeploymentSpec.model_validate(deployment(host, port))
    if MODEL not in spec.engines:
        raise SystemExit(f"{DEPLOYMENT} must serve engine {MODEL!r}")
    return build_app_from_spec(spec, base_dir=workdir)


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    import uvicorn

    from kairyu.entrypoints.cli import uvicorn_options

    args = _parse_args(sys.argv[1:] if argv is None else argv)
    args.workdir.mkdir(parents=True, exist_ok=True)
    app = build_app(args.host, args.port, args.workdir)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning", **uvicorn_options())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
