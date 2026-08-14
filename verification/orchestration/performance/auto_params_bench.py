"""Real-model gate for OpenAI request intent on an AUTO model."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import httpx

_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "arithmetic_answer",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"answer": {"type": "integer"}},
            "required": ["answer"],
            "additionalProperties": False,
        },
    },
}
_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "add",
            "description": "Add two integers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"},
                },
                "required": ["a", "b"],
                "additionalProperties": False,
            },
        },
    }
]


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _post(client: httpx.Client, body: dict) -> dict:
    response = client.post(
        "/chat/completions",
        json=body,
        headers={"X-Kairyu-Trace": "1"},
    )
    if response.status_code != 200:
        raise RuntimeError(
            f"chat request returned HTTP {response.status_code}: {response.text[:500]}"
        )
    return response.json()


def run(base_url: str, model: str, timeout: float) -> dict:
    with httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout) as client:
        models = sorted(item["id"] for item in client.get("/models").json()["data"])
        structured = _post(
            client,
            {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": "/no_think Return an object whose answer is 4.",
                    }
                ],
                "temperature": 0.0,
                "n": 2,
                "max_tokens": 64,
                "logprobs": True,
                "top_logprobs": 1,
                "seed": 208,
                "response_format": _SCHEMA,
            },
        )
        choices = sorted(structured["choices"], key=lambda choice: choice["index"])
        structured_values = [json.loads(choice["message"]["content"]) for choice in choices]
        structured_passed = (
            [choice["index"] for choice in choices] == [0, 1]
            and all(value == {"answer": 4} for value in structured_values)
            and all((choice.get("logprobs") or {}).get("content") for choice in choices)
        )

        tools = _post(
            client,
            {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "/no_think Call the add function with a=2 and b=3. "
                            "Do not answer directly."
                        ),
                    }
                ],
                "temperature": 0.0,
                "max_tokens": 128,
                "seed": 208,
                "tools": _TOOLS,
                "tool_choice": {
                    "type": "function",
                    "function": {"name": "add"},
                },
            },
        )
        tool_calls = tools["choices"][0]["message"].get("tool_calls") or []
        tool_passed = (
            len(tool_calls) == 1
            and tool_calls[0]["function"]["name"] == "add"
            and json.loads(tool_calls[0]["function"]["arguments"]) == {"a": 2, "b": 3}
        )

    return {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "base_url": base_url,
        "model": model,
        "served_models": models,
        "structured_n_logprobs": {
            "passed": structured_passed,
            "choice_indices": [choice["index"] for choice in choices],
            "values": structured_values,
            "response_sha256": _sha256(json.dumps(structured_values, sort_keys=True)),
            "usage": structured["usage"],
            "route": structured.get("kairyu_route"),
        },
        "required_named_tool": {
            "passed": tool_passed,
            "name": (tool_calls[0]["function"]["name"] if len(tool_calls) == 1 else None),
            "arguments": (
                json.loads(tool_calls[0]["function"]["arguments"]) if len(tool_calls) == 1 else None
            ),
            "usage": tools["usage"],
            "route": tools.get("kairyu_route"),
        },
        "passed": structured_passed and tool_passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8002/v1")
    parser.add_argument("--model", default="kairyu-auto")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    result = run(args.base_url, args.model, args.timeout)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.result is not None:
        args.result.parent.mkdir(parents=True, exist_ok=True)
        args.result.write_text(payload)
    print(payload, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
