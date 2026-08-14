"""Replayable paired-arm evidence for structured-output conformance.

The constrained and unconstrained requests are repeated measures of one source
task.  Raw response text, HTTP outcome, usage, and latency remain beneath that
task; every displayed rate and cost delta is recomputed from those bytes.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from importlib import resources
from statistics import mean
from typing import Any

from evals.types import (
    JSON_SAFE_INTEGER_MAX,
    SMOKE_LIMIT,
    BenchExtrasMissing,
    ItemResult,
    PairResult,
    StructuredArmEvidence,
)

PROTOCOL_ID = "paired-json-schema-v1"
CORPUS_PACKAGE = "evals.fixtures"
CORPUS_RESOURCE = "structured-output.jsonl"
CORPUS_ID = f"package:{CORPUS_PACKAGE}/{CORPUS_RESOURCE}"
CORPUS_REVISION = (
    "sha256:a40b41e1f91f6f33803a55ab4967c75d610dc82d2c14fc11d03c19ace45b05be"
)
CATEGORY_ORDER = ("nested", "recursive", "enum", "pattern", "union")
CORPUS_ITEM_IDS = (
    "nested-object",
    "recursive-tree",
    "enum-signal",
    "pattern-code",
    "union-value",
)
SCHEMA_DRAFT_URI = "https://json-schema.org/draft/2020-12/schema"

_ROW_FIELDS = frozenset(
    {"id", "category", "schema_name", "prompt", "schema", "expected", "max_tokens"}
)
_SCHEMA_NAME_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,63}")


class StrictJsonError(ValueError):
    """JSON is malformed under the benchmark's strict, duplicate-free policy."""


def strict_json_loads(text: str) -> object:
    """Parse exactly one RFC JSON value, rejecting duplicates and NaN/Infinity."""

    def reject_constant(value: str) -> None:
        raise StrictJsonError(f"non-finite JSON constant {value!r}")

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise StrictJsonError(f"duplicate JSON object key {key!r}")
            result[key] = value
        return result

    try:
        parsed = json.loads(
            text,
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except (RecursionError, UnicodeError, ValueError) as error:
        raise StrictJsonError(str(error)) from error
    stack = [(parsed, 0)]
    while stack:
        value, depth = stack.pop()
        if depth > 256:
            raise StrictJsonError("JSON nesting exceeds 256 levels")
        if isinstance(value, dict):
            stack.extend((child, depth + 1) for child in value.values())
        elif isinstance(value, list):
            stack.extend((child, depth + 1) for child in value)
        elif isinstance(value, float) and not math.isfinite(value):
            raise StrictJsonError("JSON contains a non-finite number")
    return parsed


def _jsonschema_types():
    try:
        from jsonschema import Draft202012Validator, exceptions
    except ImportError as error:
        raise BenchExtrasMissing("bench", "structured-output scoring") from error
    return Draft202012Validator, exceptions.SchemaError


def jsonschema_dependency_error() -> str | None:
    try:
        _jsonschema_types()
    except BenchExtrasMissing as error:
        return str(error)
    return None


def _external_reference(schema: object) -> str | None:
    if isinstance(schema, dict):
        reference = schema.get("$ref")
        if isinstance(reference, str) and not reference.startswith("#"):
            return reference
        for value in schema.values():
            found = _external_reference(value)
            if found is not None:
                return found
    elif isinstance(schema, list):
        for value in schema:
            found = _external_reference(value)
            if found is not None:
                return found
    return None


def _contains_keyword(schema: object, keyword: str) -> bool:
    if isinstance(schema, dict):
        return keyword in schema or any(
            _contains_keyword(value, keyword) for value in schema.values()
        )
    if isinstance(schema, list):
        return any(_contains_keyword(value, keyword) for value in schema)
    return False


def _category_feature_present(category: str, schema: dict) -> bool:
    if category == "nested":
        properties = schema.get("properties")
        return isinstance(properties, dict) and any(
            isinstance(value, dict)
            and ("properties" in value or value.get("type") == "array")
            for value in properties.values()
        )
    if category == "recursive":
        return "$defs" in schema and _contains_keyword(schema, "$ref")
    if category == "enum":
        return _contains_keyword(schema, "enum")
    if category == "pattern":
        return _contains_keyword(schema, "pattern")
    if category == "union":
        return _contains_keyword(schema, "anyOf") or _contains_keyword(schema, "oneOf")
    return False


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def canonical_corpus_payload(rows: object) -> bytes:
    """Return a type-sensitive canonical form for fixed-corpus comparisons."""

    return _canonical_json(rows).encode("utf-8")


def structured_cache_sha256() -> str:
    """Reproduce ``BenchCache.write_rows`` for the package-owned corpus."""

    payload = b"".join(
        (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")
        for row in _load_packaged_corpus_rows()
    )
    return hashlib.sha256(payload).hexdigest()


def structured_run_expectation(
    *,
    limit: object,
    smoke: object,
    selection_seed: object,
    extra_body_present: object,
) -> dict[str, object]:
    """Derive the exact fixed-corpus selection and paired-run eligibility."""

    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("structured selection limit must be null or a positive integer")
    if type(smoke) is not bool:
        raise ValueError("structured smoke selection flag must be a boolean")
    if type(selection_seed) is not int:
        raise ValueError("structured selection seed must be an integer")
    if type(extra_body_present) is not bool:
        raise ValueError("structured extra-body presence must be a boolean")
    effective_limit = min(limit or SMOKE_LIMIT, SMOKE_LIMIT) if smoke else limit
    selected_ids = list(CORPUS_ITEM_IDS)
    if effective_limit is not None and effective_limit < len(selected_ids):
        picked = random.Random(selection_seed).sample(
            range(len(selected_ids)), effective_limit
        )
        selected_ids = [selected_ids[index] for index in sorted(picked)]
    return {
        "selected_ids": selected_ids,
        "extra_body_present": extra_body_present,
    }


def validate_corpus_rows(rows: list[dict]) -> list[dict]:
    """Validate the fixed corpus and every expected answer independently."""

    Draft202012Validator, SchemaError = _jsonschema_types()
    if len(rows) != len(CATEGORY_ORDER):
        raise ValueError(
            f"structured-output corpus must contain exactly {len(CATEGORY_ORDER)} rows"
        )
    ids: list[str] = []
    names: list[str] = []
    categories: list[str] = []
    schema_digests: list[str] = []
    validated: list[dict] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != _ROW_FIELDS:
            raise ValueError(f"structured-output corpus row {index} has invalid fields")
        item_id = row["id"]
        category = row["category"]
        schema_name = row["schema_name"]
        prompt = row["prompt"]
        schema = row["schema"]
        max_tokens = row["max_tokens"]
        if not isinstance(item_id, str) or not item_id or len(item_id) > 80:
            raise ValueError(f"structured-output corpus row {index} has an invalid id")
        if category not in CATEGORY_ORDER:
            raise ValueError(f"structured-output corpus row {item_id!r} has an invalid category")
        if not isinstance(schema_name, str) or _SCHEMA_NAME_RE.fullmatch(schema_name) is None:
            raise ValueError(f"structured-output corpus row {item_id!r} has an invalid schema_name")
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 4_000:
            raise ValueError(f"structured-output corpus row {item_id!r} has an invalid prompt")
        if not isinstance(schema, dict):
            raise ValueError(f"structured-output corpus row {item_id!r} schema is not an object")
        if schema.get("$schema") != SCHEMA_DRAFT_URI:
            raise ValueError(
                f"structured-output corpus row {item_id!r} must declare Draft 2020-12"
            )
        if not _category_feature_present(category, schema):
            raise ValueError(
                f"structured-output corpus row {item_id!r} does not exercise {category}"
            )
        external = _external_reference(schema)
        if external is not None:
            raise ValueError(
                f"structured-output corpus row {item_id!r} uses external $ref {external!r}"
            )
        if type(max_tokens) is not int or not 1 <= max_tokens <= 1_024:
            raise ValueError(f"structured-output corpus row {item_id!r} has invalid max_tokens")
        try:
            Draft202012Validator.check_schema(schema)
        except SchemaError as error:
            raise ValueError(
                f"structured-output corpus row {item_id!r} has an invalid schema"
            ) from error
        validator = Draft202012Validator(schema)
        if not validator.is_valid(row["expected"]):
            raise ValueError(
                f"structured-output corpus row {item_id!r} expected answer violates its schema"
            )
        # Prove the expected value is canonical JSON without non-finite numbers.
        _canonical_json(row["expected"])
        ids.append(item_id)
        names.append(schema_name)
        categories.append(category)
        schema_digests.append(hashlib.sha256(_canonical_json(schema).encode()).hexdigest())
        validated.append(row)
    if len(ids) != len(set(ids)):
        raise ValueError("structured-output corpus item ids must be unique")
    if tuple(ids) != CORPUS_ITEM_IDS:
        raise ValueError("structured-output corpus item ids do not match the fixed protocol")
    if len(names) != len(set(names)):
        raise ValueError("structured-output corpus schema names must be unique")
    if len(schema_digests) != len(set(schema_digests)):
        raise ValueError("structured-output corpus schemas must be unique")
    if categories != list(CATEGORY_ORDER):
        raise ValueError(
            "structured-output corpus must contain nested/recursive/enum/pattern/union "
            "in protocol order"
        )
    return validated


def _load_packaged_corpus_rows() -> list[dict]:
    data = resources.files(CORPUS_PACKAGE).joinpath(CORPUS_RESOURCE).read_bytes()
    actual_revision = f"sha256:{hashlib.sha256(data).hexdigest()}"
    if actual_revision != CORPUS_REVISION:
        raise ValueError("structured-output packaged corpus digest does not match its revision")
    text = data.decode("utf-8")
    rows: list[dict] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        value = strict_json_loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"structured-output corpus line {line_number} is not an object")
        rows.append(value)
    return rows


def load_packaged_corpus() -> list[dict]:
    return validate_corpus_rows(_load_packaged_corpus_rows())


def corpus_by_id() -> dict[str, dict]:
    return {row["id"]: row for row in load_packaged_corpus()}


def response_format(row: dict) -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": row["schema_name"],
            "strict": True,
            "schema": row["schema"],
        },
    }


def request_prompt(row: dict) -> str:
    """Give both arms the same schema information before applying enforcement."""

    return (
        f"{row['prompt']}\n\n"
        "The response must satisfy this JSON Schema (Draft 2020-12):\n"
        f"{_canonical_json(row['schema'])}"
    )


def arm_order(observation_index: int) -> tuple[str, str]:
    """Alternate the selected item-major/seed-minor matrix exactly."""

    if type(observation_index) is not int or observation_index < 0:
        raise ValueError("structured observation index must be a non-negative integer")
    if observation_index % 2:
        return ("unconstrained", "constrained")
    return ("constrained", "unconstrained")


def methodology_claim(
    *,
    selected_ids: list[str],
    seeds: list[int | None],
) -> dict:
    return {
        "protocol": PROTOCOL_ID,
        "corpus": CORPUS_ID,
        "corpus_revision": CORPUS_REVISION,
        "schema_draft": "2020-12",
        "corpus_rows": len(CATEGORY_ORDER),
        "selected_ids": selected_ids,
        "seeds": seeds,
        "arms": ["constrained", "unconstrained"],
        "arm_difference": "response_format only",
        "arm_order": "alternating selected-item-major/seed-minor matrix",
        "primary_metric": "constrained schema-valid exact-task success rate",
        "rate_denominators": {
            "acceptance/schema/task": "all scheduled observations",
            "json-valid/malformed": "accepted HTTP 200 completions",
        },
        "json_parser": "strict RFC JSON; duplicate keys and non-finite constants rejected",
        "task_oracle": "type-sensitive canonical JSON equality after Draft 2020-12 validation",
        "token_cost": "endpoint-reported token counts; presence is measured, not attested",
        "latency": "client-observed retry-inclusive diagnostic, counterbalanced by arm order",
        "currency_cost": "not calculated",
    }


def evaluate_arm(arm: StructuredArmEvidence, row: dict) -> dict[str, Any]:
    completed = arm.outcome == "completed"
    parsed: object | None = None
    json_valid = False
    if (
        completed
        and arm.refusal is None
        and arm.message_auxiliary is None
        and isinstance(arm.content, str)
    ):
        try:
            parsed = strict_json_loads(arm.content)
        except StrictJsonError:
            pass
        else:
            json_valid = True
    schema_valid = False
    if json_valid:
        Draft202012Validator, _ = _jsonschema_types()
        try:
            schema_valid = Draft202012Validator(row["schema"]).is_valid(parsed)
        except (RecursionError, TypeError, ValueError):
            schema_valid = False
    task_correct = False
    if schema_valid:
        try:
            task_correct = _canonical_json(parsed) == _canonical_json(row["expected"])
        except (RecursionError, TypeError, ValueError):
            task_correct = False
    return {
        "accepted": completed,
        "schema_rejected": arm.outcome == "schema_rejected",
        "json_valid": json_valid,
        "schema_valid": schema_valid,
        "task_correct": task_correct,
        "malformed_json": completed and not json_valid,
    }


def _expected_leaf_error(
    item: ItemResult,
    *,
    row: dict,
    expected_seed: int | None,
    expected_order: tuple[str, str],
) -> str | None:
    evidence = item.structured_output
    if evidence is None:
        return "structured leaf omits paired arm evidence"
    if evidence.seed != expected_seed:
        return "structured leaf seed does not match its paired schedule"
    if evidence.order != expected_order:
        return "structured leaf arm order does not match counterbalancing"
    failed = [
        arm.arm
        for arm in (evidence.constrained, evidence.unconstrained)
        if arm.outcome == "failed"
    ]
    expected_latency = evidence.constrained.latency_s + evidence.unconstrained.latency_s
    if item.latency_s is None or not math.isclose(
        item.latency_s, expected_latency, rel_tol=0.0, abs_tol=1e-12
    ):
        return "structured item latency does not equal its paired arm latency"
    if item.response_excerpt != (
        evidence.constrained.content[:2_000]
        if evidence.constrained.content is not None
        else None
    ):
        return "structured item excerpt does not match constrained raw content"
    if failed:
        expected_error = "structured arm failure: " + ", ".join(failed)
        if item.status != "failed" or item.score is not None or item.error != expected_error:
            return "structured failed item status/error disagrees with its arms"
        return None
    expected_score = float(evaluate_arm(evidence.constrained, row)["task_correct"])
    if (
        item.status != "completed"
        or item.score != expected_score
        or item.error is not None
    ):
        return "structured item score/status disagrees with constrained task success"
    return None


def _claim_and_seeds(pair: PairResult) -> tuple[dict, list[int | None]] | None:
    claim = pair.methodology.get("structured_output")
    if not isinstance(claim, dict):
        return None
    seeds = claim.get("seeds")
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(seed is not None and type(seed) is not int for seed in seeds)
    ):
        return None
    sampling = pair.methodology.get("sampling")
    if isinstance(sampling, dict) and (
        sampling.get("attempts") != len(seeds) or sampling.get("seeds") != seeds
    ):
        return None
    if len(seeds) > 1 and (
        any(seed is None for seed in seeds) or len(set(seeds)) != len(seeds)
    ):
        return None
    return claim, seeds


def _leaf_schedule(pair: PairResult) -> list[tuple[ItemResult, int | None]] | None:
    claim_and_seeds = _claim_and_seeds(pair)
    if claim_and_seeds is None:
        return None
    _, seeds = claim_and_seeds
    scheduled: list[tuple[ItemResult, int | None]] = []
    for source in pair.items:
        if source.sampling_attempts is None:
            if len(seeds) != 1:
                return None
            scheduled.append((source, seeds[0]))
            continue
        if len(source.sampling_attempts) != len(seeds):
            return None
        for attempt, expected_seed in zip(source.sampling_attempts, seeds, strict=True):
            if expected_seed is None or attempt.seed != expected_seed:
                return None
            if attempt.result.item_id != source.item_id:
                return None
            scheduled.append((attempt.result, expected_seed))
    return scheduled


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _arm_summary(
    observations: list[tuple[StructuredArmEvidence, dict[str, Any]]],
) -> dict[str, Any]:
    total = len(observations)
    accepted = sum(result["accepted"] for _, result in observations)
    rejected = sum(result["schema_rejected"] for _, result in observations)
    json_valid = sum(result["json_valid"] for _, result in observations)
    schema_valid = sum(result["schema_valid"] for _, result in observations)
    correct = sum(result["task_correct"] for _, result in observations)
    malformed = sum(result["malformed_json"] for _, result in observations)
    usage = [
        arm.usage
        for arm, result in observations
        if result["accepted"] and arm.usage is not None
    ]
    latency = [arm.latency_s for arm, _ in observations]
    return {
        "observations": total,
        "accepted": accepted,
        "schema_rejected": rejected,
        "json_valid": json_valid,
        "schema_valid": schema_valid,
        "task_correct": correct,
        "malformed_json": malformed,
        "acceptance_rate": _ratio(accepted, total),
        "json_valid_rate": _ratio(json_valid, accepted),
        "schema_conformance_rate": _ratio(schema_valid, total),
        "task_accuracy": _ratio(correct, total),
        "malformed_json_rate": _ratio(malformed, accepted),
        "usage_observations": len(usage),
        "usage_coverage": _ratio(len(usage), accepted),
        "prompt_tokens_total": sum(value.prompt_tokens for value in usage),
        "completion_tokens_total": sum(value.completion_tokens for value in usage),
        "total_tokens_total": sum(value.total_tokens for value in usage),
        "mean_prompt_tokens": (
            mean(value.prompt_tokens for value in usage) if usage else None
        ),
        "mean_completion_tokens": (
            mean(value.completion_tokens for value in usage) if usage else None
        ),
        "mean_total_tokens": mean(value.total_tokens for value in usage) if usage else None,
        "mean_latency_s": mean(latency) if latency else None,
    }


def structured_summary(pair: PairResult) -> dict | None:
    """Recompute a complete paired matrix, or withhold the derived claim."""

    if pair.status != "completed" or not pair.items:
        return None
    rows = corpus_by_id()
    scheduled = _leaf_schedule(pair)
    if scheduled is None or not scheduled:
        return None
    leaves = [item for item, _ in scheduled]
    if any(item.status != "completed" for item in leaves):
        return None
    constrained: list[tuple[StructuredArmEvidence, dict[str, Any]]] = []
    unconstrained: list[tuple[StructuredArmEvidence, dict[str, Any]]] = []
    paired_usage = []
    latency_deltas: list[float] = []
    for item, _ in scheduled:
        row = rows.get(item.item_id)
        evidence = item.structured_output
        if row is None or evidence is None:
            return None
        constrained_result = evaluate_arm(evidence.constrained, row)
        unconstrained_result = evaluate_arm(evidence.unconstrained, row)
        constrained.append((evidence.constrained, constrained_result))
        unconstrained.append((evidence.unconstrained, unconstrained_result))
        latency_deltas.append(
            evidence.constrained.latency_s - evidence.unconstrained.latency_s
        )
        if evidence.constrained.usage is not None and evidence.unconstrained.usage is not None:
            paired_usage.append((evidence.constrained.usage, evidence.unconstrained.usage))
    constrained_summary = _arm_summary(constrained)
    unconstrained_summary = _arm_summary(unconstrained)
    claim_and_seeds = _claim_and_seeds(pair)
    if claim_and_seeds is None:
        return None
    _, seeds = claim_and_seeds
    attempts = len(seeds)
    source_categories = [rows[item.item_id]["category"] for item in pair.items]
    return {
        "protocol": PROTOCOL_ID,
        "categories": [
            category for category in CATEGORY_ORDER if category in source_categories
        ],
        "n_source_items": len(pair.items),
        "attempts": attempts,
        "observations": len(leaves),
        "arms": {
            "constrained": constrained_summary,
            "unconstrained": unconstrained_summary,
        },
        "quality_deltas": {
            "schema_conformance_rate": (
                constrained_summary["schema_conformance_rate"]
                - unconstrained_summary["schema_conformance_rate"]
            ),
            "task_accuracy": (
                constrained_summary["task_accuracy"]
                - unconstrained_summary["task_accuracy"]
            ),
        },
        "paired_cost": {
            "usage_observations": len(paired_usage),
            "usage_coverage": _ratio(len(paired_usage), len(leaves)),
            "mean_prompt_token_delta": (
                mean(left.prompt_tokens - right.prompt_tokens for left, right in paired_usage)
                if paired_usage
                else None
            ),
            "mean_completion_token_delta": (
                mean(
                    left.completion_tokens - right.completion_tokens
                    for left, right in paired_usage
                )
                if paired_usage
                else None
            ),
            "mean_total_token_delta": (
                mean(left.total_tokens - right.total_tokens for left, right in paired_usage)
                if paired_usage
                else None
            ),
            "currency_cost": None,
        },
        "latency_diagnostic": {
            "observations": len(latency_deltas),
            "mean_s_delta": mean(latency_deltas),
        },
    }


def structured_metrics(summary: dict) -> dict[str, float | int | None]:
    metrics: dict[str, float | int | None] = {
        "structured_complete": 1,
        "structured_observations": summary["observations"],
    }
    for arm_name, arm in summary["arms"].items():
        for name, value in arm.items():
            metrics[f"structured_{arm_name}_{name}"] = value
    for name, value in summary["quality_deltas"].items():
        metrics[f"structured_delta_{name}"] = value
    for name, value in summary["paired_cost"].items():
        if name != "currency_cost":
            metrics[f"structured_paired_cost_{name}"] = value
    for name, value in summary["latency_diagnostic"].items():
        metrics[f"structured_latency_diagnostic_{name}"] = value
    return metrics


def _summary_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if math.isfinite(number) else None


def _summary_ratio_matches(value: object, numerator: int, denominator: int) -> bool:
    expected = _ratio(numerator, denominator)
    if expected is None:
        return value is None
    number = _summary_number(value)
    return number is not None and math.isclose(
        number, expected, rel_tol=0.0, abs_tol=1e-12
    )


def structured_summary_error(
    value: object,
    *,
    expected_score: object,
    expected_total: object,
    expected_status: object,
) -> str | None:
    """Validate a detached derived claim before rendering its semantic labels."""

    if not isinstance(value, dict) or set(value) != {
        "protocol",
        "categories",
        "n_source_items",
        "attempts",
        "observations",
        "arms",
        "quality_deltas",
        "paired_cost",
        "latency_diagnostic",
    }:
        return "structured summary fields are invalid"
    categories = value["categories"]
    if (
        value["protocol"] != PROTOCOL_ID
        or not isinstance(categories, list)
        or not categories
        or categories != [category for category in CATEGORY_ORDER if category in categories]
        or len(categories) != len(set(categories))
    ):
        return "structured summary protocol/categories are invalid"
    source_items = value["n_source_items"]
    attempts = value["attempts"]
    observations = value["observations"]
    if (
        expected_status != "completed"
        or type(expected_total) is not int
        or not 1 <= expected_total <= JSON_SAFE_INTEGER_MAX
        or type(source_items) is not int
        or not 1 <= source_items <= JSON_SAFE_INTEGER_MAX
        or source_items != expected_total
        or type(attempts) is not int
        or not 1 <= attempts <= JSON_SAFE_INTEGER_MAX
        or type(observations) is not int
        or not 1 <= observations <= JSON_SAFE_INTEGER_MAX
        or observations != source_items * attempts
    ):
        return "structured summary dimensions disagree with the scoreboard cell"
    arms = value["arms"]
    if not isinstance(arms, dict) or set(arms) != {"constrained", "unconstrained"}:
        return "structured summary arms are invalid"
    arm_fields = {
        "observations",
        "accepted",
        "schema_rejected",
        "json_valid",
        "schema_valid",
        "task_correct",
        "malformed_json",
        "acceptance_rate",
        "json_valid_rate",
        "schema_conformance_rate",
        "task_accuracy",
        "malformed_json_rate",
        "usage_observations",
        "usage_coverage",
        "prompt_tokens_total",
        "completion_tokens_total",
        "total_tokens_total",
        "mean_prompt_tokens",
        "mean_completion_tokens",
        "mean_total_tokens",
        "mean_latency_s",
    }
    for arm_name, arm in arms.items():
        if not isinstance(arm, dict) or set(arm) != arm_fields:
            return f"structured summary {arm_name} arm fields are invalid"
        count_names = (
            "observations",
            "accepted",
            "schema_rejected",
            "json_valid",
            "schema_valid",
            "task_correct",
            "malformed_json",
            "usage_observations",
            "prompt_tokens_total",
            "completion_tokens_total",
            "total_tokens_total",
        )
        if any(
            type(arm[name]) is not int
            or not 0 <= arm[name] <= JSON_SAFE_INTEGER_MAX
            for name in count_names
        ):
            return f"structured summary {arm_name} counts are invalid"
        accepted = arm["accepted"]
        rejected = arm["schema_rejected"]
        json_valid = arm["json_valid"]
        schema_valid = arm["schema_valid"]
        task_correct = arm["task_correct"]
        malformed = arm["malformed_json"]
        usage = arm["usage_observations"]
        if (
            arm["observations"] != observations
            or accepted + rejected != observations
            or (arm_name == "unconstrained" and rejected != 0)
            or json_valid + malformed != accepted
            or not 0 <= task_correct <= schema_valid <= json_valid <= accepted
            or usage > accepted
            or arm["total_tokens_total"]
            != arm["prompt_tokens_total"] + arm["completion_tokens_total"]
        ):
            return f"structured summary {arm_name} count implications are invalid"
        rate_specs = (
            ("acceptance_rate", accepted, observations),
            ("json_valid_rate", json_valid, accepted),
            ("schema_conformance_rate", schema_valid, observations),
            ("task_accuracy", task_correct, observations),
            ("malformed_json_rate", malformed, accepted),
            ("usage_coverage", usage, accepted),
        )
        if any(
            not _summary_ratio_matches(arm[name], numerator, denominator)
            for name, numerator, denominator in rate_specs
        ):
            return f"structured summary {arm_name} rates disagree with counts"
        for mean_name, total_name in (
            ("mean_prompt_tokens", "prompt_tokens_total"),
            ("mean_completion_tokens", "completion_tokens_total"),
            ("mean_total_tokens", "total_tokens_total"),
        ):
            if not _summary_ratio_matches(arm[mean_name], arm[total_name], usage):
                return f"structured summary {arm_name} token mean disagrees with counts"
        latency = _summary_number(arm["mean_latency_s"])
        if latency is None or latency < 0:
            return f"structured summary {arm_name} latency is invalid"

    deltas = value["quality_deltas"]
    if not isinstance(deltas, dict) or set(deltas) != {
        "schema_conformance_rate",
        "task_accuracy",
    }:
        return "structured summary quality deltas are invalid"
    for name in ("schema_conformance_rate", "task_accuracy"):
        stored = _summary_number(deltas[name])
        expected = arms["constrained"][name] - arms["unconstrained"][name]
        if stored is None or not math.isclose(stored, expected, rel_tol=0.0, abs_tol=1e-12):
            return f"structured summary {name} delta disagrees with arm rates"
    score = _summary_number(expected_score)
    if score is None or not math.isclose(
        score,
        arms["constrained"]["task_accuracy"],
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        return "structured summary headline disagrees with constrained task accuracy"

    paired = value["paired_cost"]
    paired_fields = {
        "usage_observations",
        "usage_coverage",
        "mean_prompt_token_delta",
        "mean_completion_token_delta",
        "mean_total_token_delta",
        "currency_cost",
    }
    if not isinstance(paired, dict) or set(paired) != paired_fields:
        return "structured summary paired token-cost fields are invalid"
    paired_usage = paired["usage_observations"]
    if (
        type(paired_usage) is not int
        or not 0 <= paired_usage <= min(observations, JSON_SAFE_INTEGER_MAX)
        or not _summary_ratio_matches(
            paired["usage_coverage"], paired_usage, observations
        )
        or paired["currency_cost"] is not None
    ):
        return "structured summary paired token-cost coverage is invalid"
    for name in (
        "mean_prompt_token_delta",
        "mean_completion_token_delta",
        "mean_total_token_delta",
    ):
        number = _summary_number(paired[name])
        if (paired_usage == 0 and paired[name] is not None) or (
            paired_usage > 0 and number is None
        ):
            return f"structured summary paired token cost {name} is invalid"
    if paired_usage > 0 and not math.isclose(
        paired["mean_total_token_delta"],
        paired["mean_prompt_token_delta"] + paired["mean_completion_token_delta"],
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        return "structured summary paired total-token delta is inconsistent"
    latency = value["latency_diagnostic"]
    if (
        not isinstance(latency, dict)
        or set(latency) != {"observations", "mean_s_delta"}
        or type(latency["observations"]) is not int
        or not 0 <= latency["observations"] <= JSON_SAFE_INTEGER_MAX
        or latency["observations"] != observations
        or _summary_number(latency["mean_s_delta"]) is None
    ):
        return "structured summary latency diagnostic is invalid"
    expected_latency_delta = (
        arms["constrained"]["mean_latency_s"]
        - arms["unconstrained"]["mean_latency_s"]
    )
    if not math.isclose(
        latency["mean_s_delta"],
        expected_latency_delta,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        return "structured summary latency delta disagrees with arm means"
    return None


def _pair_envelope_error(pair: PairResult) -> str | None:
    n_total = len(pair.items)
    counts = {
        status: sum(item.status == status for item in pair.items)
        for status in ("completed", "failed", "unjudged", "skipped")
    }
    expected_counts = {
        "n_total": n_total,
        "n_scored": sum(
            item.status == "completed" and item.score is not None for item in pair.items
        ),
        "n_unjudged": counts["unjudged"],
        "n_skipped": counts["skipped"],
        "n_failed": counts["failed"],
    }
    for name, expected in expected_counts.items():
        if pair.metrics.get(name) != expected:
            return f"metrics.{name} does not match structured source-item evidence"
    scored = [
        float(item.score)
        for item in pair.items
        if item.status == "completed" and item.score is not None
    ]
    expected_score = mean(scored) if scored else None
    if expected_score is None:
        if pair.score is not None:
            return "metrics.score does not match structured source-item evidence"
    elif pair.score is None or not math.isclose(
        pair.score, expected_score, rel_tol=0.0, abs_tol=1e-12
    ):
        return "metrics.score does not match structured source-item evidence"

    reasons = []
    for key, label in (
        ("unjudged", "unjudgeable"),
        ("skipped", "skipped"),
        ("failed", "failed"),
    ):
        if counts[key]:
            reasons.append(f"{counts[key]}/{n_total} items {label}")
    if counts["failed"] > n_total / 2:
        expected_status = "failed"
    elif counts["completed"] == n_total:
        expected_status = "completed"
    elif not scored:
        expected_status = "skipped"
    else:
        expected_status = "partial"
    expected_reason = (
        None
        if expected_status == "completed"
        else "; ".join(reasons) or "no scoreable items"
    )
    if pair.status != expected_status or pair.reason != expected_reason:
        return "structured pair status/reason does not match source-item evidence"
    return None


def structured_pair_evidence_error(
    pair: PairResult,
    *,
    expected_sampling: dict[str, object] | None = None,
    expected_run: dict[str, object] | None = None,
) -> str | None:
    """Bind structured methodology, leaves, metrics, and primary score."""

    if pair.benchmark != "structured-output":
        return "structured evidence has the wrong benchmark identity"
    expected_seeds: list[int | None] | None = None
    stochastic_precondition_failed = False
    if expected_sampling is not None:
        from evals.sampling import seed_schedule

        attempts = expected_sampling.get("attempts")
        target_seed = expected_sampling.get("seed")
        mode = expected_sampling.get("mode")
        temperature = expected_sampling.get("temperature")
        if (
            type(attempts) is not int
            or attempts < 1
            or (target_seed is not None and type(target_seed) is not int)
            or mode not in {"adapter", "recommended"}
            or (
                temperature is not None
                and (
                    type(temperature) not in (int, float)
                    or not math.isfinite(temperature)
                    or temperature < 0
                )
            )
        ):
            return "structured expected sampling configuration is invalid"
        try:
            expected_seeds = list(seed_schedule(target_seed, attempts))
        except ValueError:
            return "structured expected sampling seed schedule is invalid"
        stochastic = mode == "recommended" or (
            temperature is not None and temperature != 0.0
        )
        stochastic_precondition_failed = (
            stochastic and target_seed is None and attempts == 1
        )
    expected_ids: list[str] | None = None
    extra_body_present = False
    if expected_run is not None:
        if not isinstance(expected_run, dict) or set(expected_run) != {
            "selected_ids",
            "extra_body_present",
        }:
            return "structured expected run configuration is invalid"
        selected_ids = expected_run["selected_ids"]
        extra_body = expected_run["extra_body_present"]
        if (
            not isinstance(selected_ids, list)
            or not selected_ids
            or any(type(item_id) is not str for item_id in selected_ids)
            or len(selected_ids) != len(set(selected_ids))
            or any(item_id not in CORPUS_ITEM_IDS for item_id in selected_ids)
            or type(extra_body) is not bool
        ):
            return "structured expected run configuration is invalid"
        expected_ids = selected_ids
        extra_body_present = extra_body
    structured_keys = {name for name in pair.metrics if name.startswith("structured_")}
    if not pair.items:
        if (
            pair.status not in {"skipped", "failed"}
            or not pair.reason
            or pair.metrics != {"score": None, "n_total": 0}
            or structured_keys
            or "structured_output" in pair.methodology
        ):
            return "empty structured pair is not a canonical skip/failure"
        return None
    if extra_body_present:
        return "structured completed evidence conflicts with rejected extra_body_json"
    if stochastic_precondition_failed:
        return "structured stochastic single-attempt evidence requires target.seed"
    envelope_error = _pair_envelope_error(pair)
    if envelope_error is not None:
        return envelope_error
    source_ids = [item.item_id for item in pair.items]
    claim_and_seeds = _claim_and_seeds(pair)
    if claim_and_seeds is None:
        return "structured methodology omits its paired seed schedule"
    claim, seeds = claim_and_seeds
    if expected_seeds is not None:
        if seeds != expected_seeds:
            return "structured paired seeds disagree with the target configuration"
    if claim != methodology_claim(selected_ids=source_ids, seeds=seeds):
        return "structured methodology does not match the paired corpus protocol"
    if len(source_ids) != len(set(source_ids)):
        return "structured source item ids must be unique"
    if expected_ids is not None and source_ids != expected_ids:
        return "structured selected item ids disagree with the run configuration"
    known = corpus_by_id()
    if any(item_id not in known for item_id in source_ids):
        return "structured source item id is outside the fixed corpus"
    scheduled = _leaf_schedule(pair)
    if scheduled is None or len(scheduled) != len(source_ids) * len(seeds):
        return "structured evidence does not form the declared paired seed matrix"
    for observation_index, (item, expected_seed) in enumerate(scheduled):
        row = known.get(item.item_id)
        if row is None:
            return "structured leaf item id is outside the fixed corpus"
        error = _expected_leaf_error(
            item,
            row=row,
            expected_seed=expected_seed,
            expected_order=arm_order(observation_index),
        )
        if error is not None:
            return f"structured item {item.item_id!r}: {error}"
    summary = structured_summary(pair)
    expected: dict[str, float | int | None] = {
        "structured_complete": int(summary is not None),
        "structured_observations": len(scheduled),
    }
    if summary is not None:
        expected.update(structured_metrics(summary))
        constrained_accuracy = summary["arms"]["constrained"]["task_accuracy"]
        if pair.score is None or not math.isclose(
            pair.score,
            constrained_accuracy,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            return "structured primary score does not equal constrained task success"
    if structured_keys != set(expected):
        return "structured metric names do not match the retained arm evidence"
    for name, value in expected.items():
        stored = pair.metrics.get(name)
        if value is None:
            if stored is not None:
                return f"metrics.{name} disagrees with structured arm evidence"
        elif isinstance(value, int):
            if type(stored) is not int or stored != value:
                return f"metrics.{name} disagrees with structured arm evidence"
        elif (
            isinstance(stored, bool)
            or not isinstance(stored, int | float)
            or not math.isclose(float(stored), value, rel_tol=0.0, abs_tol=1e-12)
        ):
            return f"metrics.{name} disagrees with structured arm evidence"
    return None
