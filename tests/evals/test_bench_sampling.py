"""Endpoint sampling policy for retained checkout-only eval suites."""

import argparse
import json

import httpx
import pytest
from conftest import make_target

from evals.adapters.base import call_chat
from evals.config import build_config, parse_target_flag
from evals.types import (
    BenchTarget,
    ChatRequestSpec,
)


def _run_args(**overrides) -> argparse.Namespace:
    defaults = dict(
        config=None,
        base_url="http://gw/v1",
        model=["m"],
        target=None,
        api_key_env=None,
        reasoning_effort=None,
        top_p=None,
        sampling_seed=None,
        extra_body=None,
        suite="core",
        only=None,
        exclude=None,
        limit=None,
        smoke=False,
        offline_fixtures=False,
        seed=None,
        concurrency=None,
        results_dir=None,
        run_id=None,
        rerun=False,
        cache_dir=None,
        no_download=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _capture_body() -> tuple[httpx.AsyncClient, list[dict]]:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={"choices": [{"index": 0, "message": {"role": "a", "content": "ok"}}]},
        )

    return httpx.AsyncClient(transport=httpx.MockTransport(handler)), seen


# -- wire body -----------------------------------------------------------------


async def test_target_sampling_reaches_the_request_body():
    client, seen = _capture_body()
    target = make_target(
        reasoning_effort="high",
        top_p=0.95,
        seed=1234,
        extra_body_json='{"chat_template_kwargs": {"enable_thinking": true}}',
    )
    async with client:
        await call_chat(
            client,
            target,
            ChatRequestSpec(messages=({"role": "user", "content": "q"},), max_tokens=16),
            retries=0,
            timeout_s=5,
        )
    body = seen[0]
    assert body["reasoning_effort"] == "high"
    assert body["top_p"] == 0.95
    assert body["seed"] == 1234
    assert body["chat_template_kwargs"] == {"enable_thinking": True}
    # the request still carries what the adapter asked for
    assert body["max_tokens"] == 16 and body["temperature"] == 0.0
    assert body["stream"] is False and body["model"] == target.model


async def test_unset_sampling_leaves_the_body_untouched():
    client, seen = _capture_body()
    async with client:
        await call_chat(
            client,
            make_target(),
            ChatRequestSpec(messages=({"role": "user", "content": "q"},)),
            retries=0,
            timeout_s=5,
        )
    assert list(seen[0]) == ["model", "messages", "temperature", "stream"]


# -- validation ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("not json", "not valid JSON"),
        ("[1, 2]", "must be a JSON object"),
        ('{"model": "sneaky"}', "may not override model"),
        ('{"messages": [], "stream": true}', "may not override messages, stream"),
        # merged last, so these would silently beat the adapter's request...
        ('{"max_tokens": 1}', "may not override max_tokens"),
        ('{"temperature": 1.5}', "may not override temperature"),
        # ...and these would beat the typed policy the fingerprint records
        ('{"reasoning_effort": "low"}', "may not override reasoning_effort"),
        ('{"top_p": 0.1}', "may not override top_p"),
        ('{"seed": 9}', "may not override seed"),
    ],
)
def test_extra_body_is_validated_at_load_time(value, message):
    with pytest.raises(ValueError, match=message):
        BenchTarget(base_url="http://gw", model="m", extra_body_json=value)


async def test_extra_body_cannot_silently_replace_the_adapter_request():
    """Whatever extra_body carries, the adapter's own fields survive."""
    client, seen = _capture_body()
    target = make_target(
        reasoning_effort="high",
        extra_body_json='{"chat_template_kwargs": {"enable_thinking": false}}',
    )
    async with client:
        await call_chat(
            client,
            target,
            ChatRequestSpec(
                messages=({"role": "user", "content": "q"},), max_tokens=64, temperature=0.0
            ),
            retries=0,
            timeout_s=5,
        )
    body = seen[0]
    assert body["max_tokens"] == 64
    assert body["temperature"] == 0.0
    assert body["reasoning_effort"] == "high"
    assert body["chat_template_kwargs"] == {"enable_thinking": False}


def test_top_p_bounds_are_enforced():
    for bad in (0.0, 1.5, -1):
        with pytest.raises(ValueError):
            BenchTarget(base_url="http://gw", model="m", top_p=bad)
    assert BenchTarget(base_url="http://gw", model="m", top_p=1.0).top_p == 1.0


# -- CLI / config wiring -------------------------------------------------------


def test_cli_sampling_applies_to_model_targets():
    config = build_config(
        _run_args(reasoning_effort="high", top_p=0.9, sampling_seed=7, extra_body='{"a": 1}')
    )
    target = config.targets[0]
    assert target.reasoning_effort == "high"
    assert target.top_p == 0.9
    assert target.seed == 7
    assert target.extra_body() == {"a": 1}


def test_cli_sampling_applies_to_explicit_targets():
    target = parse_target_flag("n=http://gw/v1=m", reasoning_effort="minimal")
    assert target.reasoning_effort == "minimal"
    config = build_config(
        _run_args(model=None, target=["n=http://gw/v1=m"], reasoning_effort="high")
    )
    assert config.targets[0].reasoning_effort == "high"


def test_cli_sampling_overrides_yaml_targets(tmp_path):
    path = tmp_path / "bench.yaml"
    path.write_text(
        "targets:\n"
        "  - name: a\n"
        "    base_url: http://gw/v1\n"
        "    model: m\n"
        "    reasoning_effort: low\n",
        encoding="utf-8",
    )
    config = build_config(
        _run_args(config=str(path), model=None, base_url=None, reasoning_effort="high")
    )
    assert config.targets[0].reasoning_effort == "high"

    kept = build_config(_run_args(config=str(path), model=None, base_url=None))
    assert kept.targets[0].reasoning_effort == "low"
