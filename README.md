# Kairyu

vLLM-compatible LLM inference framework with **native orchestration**: a learned-router-ready
Router, a Planner/Worker/Verifier/Synthesizer Conductor (role DAG), and Mixture-of-Agents —
all behind one Python API and one OpenAI-compatible endpoint.

Status: **M1 complete**; GPU-independent halves of M2/M3/M4 implemented (see `docs/design/`):

- M1 — L2 orchestration + L3 interface on pluggable backends (`mock` / `vllm` / `openai`),
  incl. `LLM` and `AsyncLLMEngine` vLLM drop-in compatibility
- M2 (CPU half) — Radix-Paged KV manager, chunked-prefill scheduler, EngineCore step loop
  (`kairyu/engine/core/`); FlashInfer runner / FP8 / overlap pipelining need H100/A100
- M3 (CPU half) — n-gram draft speculative decoding policy with a tested
  greedy-equivalence invariant; EAGLE / CUDA graphs / P-D split are GPU-phase
- M4 — router learning pipeline: JSONL serving logs → labeled dataset → distilled
  classifier (`LearnedRouter`) → contextual bandit online refinement

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
