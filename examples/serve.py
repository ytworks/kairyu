"""Start the OpenAI-compatible server backed by mock engines.

Run: uv run python examples/serve.py
Then: curl localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model": "kairyu-auto", "messages": [{"role": "user", "content": "hi"}]}'
"""

import uvicorn

from kairyu.engine.mock import MockBackend
from kairyu.entrypoints.server.app import create_app
from kairyu.orchestration.orchestrator import Orchestrator

engines = {"kairyu-mock": MockBackend()}
orchestrator = Orchestrator(engines={"tier1": MockBackend(), "tier2": MockBackend()})
app = create_app(engines=engines, orchestrator=orchestrator)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
