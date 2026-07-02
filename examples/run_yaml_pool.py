"""Load the YAML agent pool and run one orchestrated query."""

from pathlib import Path

from kairyu.dsl.loader import build_orchestrator, load_spec

spec = load_spec(Path(__file__).parent / "agent_pool.yaml")
orchestrator = build_orchestrator(spec)
result = orchestrator.run_sync(
    "First, outline a caching strategy for a chat service. "
    "Then pick one approach. Finally, list its risks."
)
print(f"route: {result.route.target}")
print(result.text)
