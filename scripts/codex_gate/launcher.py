"""Kairyu server for one Codex matrix scenario (M20 WP-05, D1).

``python -m scripts.codex_gate.launcher --scenario NAME --port PORT --workdir DIR``
serves the scenario's ``ScenarioBackend`` script as engine ``kairyu-scenario``
and, for AUTO scenarios, an orchestration ``kairyu-auto-scenario`` whose rule
router always takes a direct tier over that engine. The deployment is an
ordinary ``DeploymentSpec`` built by ``build_app_from_spec`` and served by
uvicorn with ``ws="none"`` (WebSocket upgrades are plain 426 GETs, D-e), so
Codex meets the production HTTP stack. The scenario backend is registered here
only; importing this module registers nothing.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import yaml

from scripts.codex_gate.scenarios import (
    AUTO_MODEL,
    ENGINE_MODEL,
    MatrixScenario,
    scenario_named,
)

BACKEND = "codex-gate-scenario"
# Thresholds no Codex prompt reaches: AUTO always routes to one direct tier.
DIRECT_ROUTE = {
    "multi_step_markers": 10**6,
    "multi_agent_min_chars": 10**9,
    "reasoning_keywords": 10**6,
    "math_symbols": 10**6,
    "tier2_min_chars": 10**9,
}


def orchestrator_spec() -> dict:
    """Direct-route AUTO over the scenario engine (both tiers are that engine)."""

    return {
        "workers": [
            {"name": "tier1", "engine_ref": ENGINE_MODEL},
            {"name": "tier2", "engine_ref": ENGINE_MODEL},
        ],
        "router": {"kind": "rules", "thresholds": DIRECT_ROUTE},
    }


def deployment(scenario: MatrixScenario, host: str, port: int, workdir: Path) -> dict:
    """The DeploymentSpec mapping that serves ``scenario``."""

    spec: dict = {
        "server": {"host": host, "port": port, "access_log": False},
        "engines": {
            ENGINE_MODEL: {
                "backend": BACKEND,
                "options": {"max_model_len": scenario.max_model_len},
            }
        },
        # ScenarioBackend scripts read the legacy "role: text" rendering.
        "legacy_chat_models": [ENGINE_MODEL],
    }
    if scenario.tenant_limits:
        spec["tenants"] = {"limits": {"default": dict(scenario.tenant_limits)}}
    if scenario.model == "auto":
        path = workdir / "auto-orchestrator.yaml"
        path.write_text(yaml.safe_dump(orchestrator_spec(), sort_keys=False), encoding="utf-8")
        spec["orchestrators"] = {AUTO_MODEL: {"spec": str(path)}}
    return spec


def build_app(scenario: MatrixScenario, host: str, port: int, workdir: Path):
    from kairyu.deploy.builder import build_app_from_spec
    from kairyu.deploy.spec import DeploymentSpec
    from tests.support.scenario_backend import register_scenario_backend

    if scenario.script is None:
        raise SystemExit(f"scenario {scenario.name!r} targets a live deployment (--live)")
    register_scenario_backend(scenario.script, name=BACKEND)
    spec = DeploymentSpec.model_validate(deployment(scenario, host, port, workdir))
    return build_app_from_spec(spec, base_dir=workdir)


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    import uvicorn

    args = _parse_args(sys.argv[1:] if argv is None else argv)
    args.workdir.mkdir(parents=True, exist_ok=True)
    app = build_app(scenario_named(args.scenario), args.host, args.port, args.workdir)
    uvicorn.run(app, host=args.host, port=args.port, ws="none", log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
