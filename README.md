# Kairyu

vLLM-compatible LLM inference framework with **native orchestration**: a learned-router-ready
Router, a Planner/Worker/Verifier/Synthesizer Conductor (role DAG), and Mixture-of-Agents —
all behind one Python API and one OpenAI-compatible endpoint.

Status: **M1** — orchestration (L2) + interface (L3) layers, running on pluggable engine
backends (`mock` for dev/CI, `vllm` on GPU hosts, `openai` for external APIs). The custom
engine (overlap scheduler + Radix-Paged KV) is milestone M2; see `docs/design/`.

## Quick start

```bash
uv sync
uv run pytest                      # 85+ tests, coverage gate 80%
uv run python examples/basic_offline_inference.py
uv run python examples/run_yaml_pool.py
uv run python examples/serve.py    # OpenAI-compatible server on :8000
```

## vLLM drop-in

```python
from kairyu import LLM, SamplingParams   # was: from vllm import ...

llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")
outputs = llm.generate(["Hello, my name is"], SamplingParams(temperature=0.8))
print(outputs[0].outputs[0].text)
```

## Orchestration

```python
from kairyu import Orchestrator
from kairyu.engine.mock import MockBackend

orchestrator = Orchestrator(engines={"tier1": MockBackend(), "tier2": MockBackend()})
result = orchestrator.run_sync("First, plan X. Then do Y. Finally, verify.")
print(result.route.target, result.text)
```

Or declaratively — see `examples/agent_pool.yaml` (workers, role DAG, budget) and
`kairyu.dsl.decorators.AgentPool` for the decorator front-end.

## Layout

- `kairyu/engine/` — `EngineBackend` protocol + mock/vllm/openai backends
- `kairyu/orchestration/` — Router, Conductor, MoA, Budget, Orchestrator
- `kairyu/dsl/` — YAML/decorator agent-pool spec
- `kairyu/entrypoints/` — `LLM` API and OpenAI-compatible FastAPI server
- `bench/` — measurement scripts (framework overhead now; engine benches in M2)
- `docs/design/` — design docs per milestone
