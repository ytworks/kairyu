#!/usr/bin/env python3
"""Exercise real image chat and exact usage through Kairyu's OpenAI API."""

from __future__ import annotations

import argparse
import base64
import json
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

PROMPT = (
    "Read the single large word shown in the image. Reply with exactly that "
    "uppercase word, RED or BLUE, and no other text."
)
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class GateFailure(RuntimeError):
    """A binding VLM smoke assertion failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateFailure(message)


def data_url(path: Path) -> str:
    payload = path.read_bytes()
    require(payload.startswith(PNG_SIGNATURE), f"{path} is not a PNG fixture")
    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def request_body(model: str, image_url: str, *, stream: bool) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url, "detail": "high"},
                    },
                    {"type": "text", "text": PROMPT},
                ],
            }
        ],
        "temperature": 0,
        "seed": 203,
        "max_tokens": 16,
        "stream": stream,
    }
    if stream:
        body["stream_options"] = {"include_usage": True}
    return body


def open_request(
    endpoint: str,
    body: dict[str, Any],
    *,
    timeout: float,
) -> urllib.response.addinfourl:
    encoded = json.dumps(body, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=encoded,
        method="POST",
        headers={
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
        },
    )
    return urllib.request.urlopen(request, timeout=timeout)


def json_completion(
    endpoint: str,
    body: dict[str, Any],
    *,
    timeout: float,
) -> dict[str, Any]:
    with open_request(endpoint, body, timeout=timeout) as response:
        require(response.status == 200, f"completion returned HTTP {response.status}")
        content_type = response.headers.get("Content-Type", "").lower()
        require("application/json" in content_type, f"unexpected content type {content_type!r}")
        payload = json.load(response)
    require(isinstance(payload, dict), "completion response was not a JSON object")
    return payload


def answer_text(payload: dict[str, Any]) -> str:
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as error:
        raise GateFailure(f"completion response has no assistant content: {payload}") from error
    require(isinstance(content, str) and content.strip(), "assistant content was empty")
    return content.strip()


def assert_color(text: str, expected: str) -> None:
    other = "BLUE" if expected == "RED" else "RED"
    words = re.findall(r"[A-Z]+", text.upper())
    require(expected in words, f"image expected {expected}, got {text!r}")
    require(other not in words, f"image answer was ambiguous for {expected}: {text!r}")


def assert_usage(payload: dict[str, Any], *, max_prompt_tokens: int) -> dict[str, int]:
    usage = payload.get("usage")
    require(isinstance(usage, dict), f"response omitted exact usage: {payload}")
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    require(
        isinstance(prompt_tokens, int)
        and not isinstance(prompt_tokens, bool)
        and prompt_tokens > 0,
        f"invalid prompt token usage: {usage}",
    )
    require(
        isinstance(completion_tokens, int)
        and not isinstance(completion_tokens, bool)
        and completion_tokens > 0,
        f"invalid completion token usage: {usage}",
    )
    require(
        prompt_tokens <= max_prompt_tokens,
        f"high-detail image used {prompt_tokens} prompt tokens, above {max_prompt_tokens}",
    )
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }


def stream_completion(
    endpoint: str,
    body: dict[str, Any],
    *,
    timeout: float,
) -> tuple[str, dict[str, Any]]:
    content_parts: list[str] = []
    usage: dict[str, Any] | None = None
    saw_done = False
    with open_request(endpoint, body, timeout=timeout) as response:
        require(response.status == 200, f"stream returned HTTP {response.status}")
        content_type = response.headers.get("Content-Type", "").lower()
        require("text/event-stream" in content_type, f"unexpected stream type {content_type!r}")
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line.removeprefix("data:").strip()
            if data == "[DONE]":
                saw_done = True
                continue
            if not data:
                continue
            event = json.loads(data)
            require("error" not in event, f"stream returned an error event: {event}")
            if isinstance(event.get("usage"), dict):
                usage = event["usage"]
            for choice in event.get("choices", []):
                content = choice.get("delta", {}).get("content")
                if isinstance(content, str):
                    content_parts.append(content)
    require(saw_done, "stream did not terminate with data: [DONE]")
    require(usage is not None, "stream omitted the requested exact usage chunk")
    return "".join(content_parts).strip(), {"usage": usage}


def assert_remote_url_rejected(
    endpoint: str,
    model: str,
    *,
    timeout: float,
) -> None:
    body = request_body(
        model,
        "http://169.254.169.254/latest/meta-data/iam/security-credentials/",
        stream=False,
    )
    try:
        response = open_request(endpoint, body, timeout=timeout)
    except urllib.error.HTTPError as error:
        payload = json.loads(error.read())
        require(error.code == 400, f"remote image URL returned HTTP {error.code}: {payload}")
        require(
            payload.get("error", {}).get("code") == "invalid_image",
            f"remote image URL did not fail closed as invalid_image: {payload}",
        )
        return
    response.close()
    raise GateFailure("remote image URL was accepted")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="qwen3-vl-32b")
    parser.add_argument("--red", type=Path, required=True)
    parser.add_argument("--blue", type=Path, required=True)
    parser.add_argument("--max-prompt-tokens", type=int, default=8192)
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()
    require(args.max_prompt_tokens > 0, "--max-prompt-tokens must be positive")
    require(args.timeout > 0, "--timeout must be positive")

    endpoint = f"{args.base_url.rstrip('/')}/chat/completions"
    results: dict[str, dict[str, Any]] = {}
    answers: list[str] = []
    urls = {"RED": data_url(args.red), "BLUE": data_url(args.blue)}
    for expected, image_url in urls.items():
        payload = json_completion(
            endpoint,
            request_body(args.model, image_url, stream=False),
            timeout=args.timeout,
        )
        answer = answer_text(payload)
        assert_color(answer, expected)
        answers.append(answer)
        results[expected.lower()] = {
            "answer": answer,
            "usage": assert_usage(payload, max_prompt_tokens=args.max_prompt_tokens),
        }
    require(answers[0] != answers[1], f"red and blue images produced one answer: {answers[0]!r}")

    streamed_answer, streamed_payload = stream_completion(
        endpoint,
        request_body(args.model, urls["RED"], stream=True),
        timeout=args.timeout,
    )
    assert_color(streamed_answer, "RED")
    results["stream_red"] = {
        "answer": streamed_answer,
        "usage": assert_usage(
            streamed_payload,
            max_prompt_tokens=args.max_prompt_tokens,
        ),
    }
    assert_remote_url_rejected(endpoint, args.model, timeout=args.timeout)
    results["remote_url"] = {"status": 400, "code": "invalid_image"}

    print(
        json.dumps(
            {
                "model": args.model,
                "prompt": PROMPT,
                "detail": "high",
                "max_prompt_tokens": args.max_prompt_tokens,
                "results": results,
                "passed": True,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (GateFailure, OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(f"VLM API FAIL: {error}") from error
