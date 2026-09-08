"""Real-worker checklist quality gate, separate from successful serving traces."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
_STAGE = re.compile(r"(?m)^### (?P<node>[\w:-]+) — attempt (?P<attempt>\d+)\n")


def parse_checklist(text: str) -> list[dict[str, str]]:
    """Require a complete nonempty JSON checklist and stable, usable criteria."""
    values = json.loads(text)
    if not isinstance(values, list) or not values:
        raise ValueError("expected a nonempty JSON checklist array")
    entries = []
    for index, value in enumerate(values, 1):
        if not isinstance(value, dict) or set(value) != {
            "id",
            "priority",
            "requirement",
            "acceptance_criterion",
            "source",
        }:
            raise ValueError(f"incomplete requirement fields on item {index}")
        if value["id"] != f"R{index}":
            raise ValueError("requirement IDs must be consecutive and unique")
        if value["priority"] not in ("minimum", "optional"):
            raise ValueError("invalid requirement priority")
        for key in ("requirement", "acceptance_criterion", "source"):
            if not isinstance(value[key], str) or not re.search(r"\w", value[key]):
                raise ValueError(f"empty {key} on R{index}")
        entries.append(
            {
                "id": str(index),
                "priority": value["priority"],
                "requirement": value["requirement"],
                "acceptance": value["acceptance_criterion"],
                "source": value["source"],
            }
        )
    return entries


def stage_outputs(reasoning: str) -> list[dict]:
    matches = list(_STAGE.finditer(reasoning))
    outputs = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(reasoning)
        section = reasoning[match.end() : end].split("\n### Final answer attribution", 1)[0]
        if "#### Stage output\n" not in section:
            continue
        output = section.rsplit("#### Stage output\n", 1)[1].rsplit("\n---", 1)[0].strip()
        outputs.append(
            {"node": match["node"], "attempt": int(match["attempt"]) - 1, "output": output}
        )
    return outputs


def parse_audit(text: str) -> list[dict[str, str]]:
    lines = text.strip().splitlines()
    if not lines or lines[0].strip() not in ("PASS", "FAIL"):
        raise ValueError("audit has no explicit verdict")
    body = "\n".join(lines[1:]).strip()
    starts = list(re.finditer(r"(?m)^R[1-9]\d*\s*\|", body))
    if not starts or starts[0].start() != 0:
        raise ValueError("audit has no per-ID assessments")
    items = []
    for index, start in enumerate(starts):
        end = starts[index + 1].start() if index + 1 < len(starts) else len(body)
        match = re.fullmatch(
            r"R(?P<id>[1-9]\d*)\s*\|\s*"
            r"(?P<status>satisfied|unsatisfied|unverifiable|unsupported)\s*\|\s*"
            r"evidence:\s*(?P<evidence>.*?)\s*\|\s*correction:\s*(?P<correction>.*)",
            body[start.start() : end].strip(),
            re.DOTALL | re.IGNORECASE,
        )
        if match is None:
            # A satisfied item needs evidence, but no repair. Accept the
            # observed three-column form without guessing missing statuses
            # or accepting incomplete failure assessments.
            compact = re.fullmatch(
                r"R(?P<id>[1-9]\d*)\s*\|\s*satisfied\s*\|\s*(?P<evidence>[^|\n]+)",
                body[start.start() : end].strip(),
                re.IGNORECASE,
            )
            if compact is None:
                raise ValueError("audit item lacks status, evidence, or correction field")
            item = {**compact.groupdict(), "status": "satisfied", "correction": "none"}
        else:
            item = match.groupdict()
        item = {key: value.strip() for key, value in item.items()}
        item["status"] = item["status"].lower()
        evidence = re.sub(
            r"^(?:evidence|correction):\s*", "", item["evidence"], flags=re.IGNORECASE
        )
        evidence = evidence.strip(" .-'\"").lower()
        if not re.search(r"\w", evidence) or evidence in ("none", "n/a", "unknown", "missing"):
            raise ValueError("audit evidence is empty or a placeholder")
        correction = item["correction"].strip(" .-'\"").lower()
        if item["status"] != "satisfied" and (
            not re.search(r"\w", correction)
            or correction in ("none", "n/a", "unknown", "missing")
        ):
            raise ValueError("non-satisfied audit item lacks a concrete correction")
        items.append(item)
    return items


def _overlap(first: dict, second: dict) -> bool:
    try:
        starts = [
            datetime.fromisoformat(event["timing"]["started_at"]) for event in (first, second)
        ]
        ends = [
            datetime.fromisoformat(event["timing"]["completed_at"]) for event in (first, second)
        ]
        return max(starts) < min(ends)
    except (KeyError, TypeError, ValueError):
        return False


def validate_result(case: dict, result: dict, *, requirement_cap: int) -> dict:
    checks = {}
    failures = []
    events = (result.get("trace") or {}).get("events", [])
    outputs = stage_outputs(result.get("reasoning", ""))
    requirements = [item for item in outputs if item["node"] == "requirements"]
    req_events = [event for event in events if event.get("node") == "requirements"]
    audits = [item for item in outputs if item["node"] == "audit"]
    audit_events = [event for event in events if event.get("node") == "audit"]
    direct_nodes = {"qwen_answer", "qwen_think_answer", "deepseek_answer", "deepseek_think_answer"}
    checks["ensemble_executed"] = (
        any(event.get("node") == "profile_judge" for event in events)
        and any(
            event.get("node") == "synthesis" and event.get("status") == "success"
            for event in events
        )
        and not any(event.get("node") in direct_nodes for event in events)
    )
    checks["one_successful_requirements_stage"] = (
        len(req_events) == 1 and req_events[0].get("status") == "success" and len(requirements) == 1
    )
    entries = []
    try:
        if len(requirements) != 1:
            raise ValueError("expected exactly one requirements output")
        entries = parse_checklist(requirements[0]["output"])
        checks["complete_checklist"] = True
    except ValueError as error:
        checks["complete_checklist"] = False
        failures.append(str(error))
    minimum_text = "\n".join(
        entry["requirement"] + " | " + entry["acceptance"]
        for entry in entries
        if entry["priority"] == "minimum"
    )
    for group in case["checklist_coverage"]:
        checks["covers:" + group["name"]] = bool(entries) and all(
            re.search(pattern, minimum_text, re.IGNORECASE) is not None
            for pattern in group["patterns"]
        )
    minimum_acceptance = "\n".join(
        entry["acceptance"] for entry in entries if entry["priority"] == "minimum"
    )
    checks["preserves_required_literals"] = bool(entries) and all(
        literal in minimum_acceptance for literal in case["checklist_literals"]
    )
    checks["excludes_rendering_boilerplate"] = bool(entries) and all(
        "return only the assistant response body" not in entry["source"].lower()
        for entry in entries
    )
    spent = (req_events[0].get("usage") or {}).get("completion_tokens") if req_events else None
    checks["requirements_finish_within_cap"] = (
        isinstance(spent, int) and 0 < spent < requirement_cap
    )
    ids = [entry["id"] for entry in entries]
    checks["audit_covers_checklist_ids"] = (
        bool(ids)
        and bool(audits)
        and all(
            sorted(re.findall(r"(?m)^R(\d+)\s*\|", audit["output"]), key=int) == ids
            and audit["output"].splitlines()[0].strip() in ("PASS", "FAIL")
            for audit in audits
        )
    )
    assessed = []
    try:
        assessed = [parse_audit(audit["output"]) for audit in audits]
        checks["audit_evidence_present"] = bool(assessed)
    except ValueError as error:
        checks["audit_evidence_present"] = False
        failures.append(str(error))
    final_items = {item["id"]: item for item in assessed[-1]} if assessed else {}
    minimum_ids = [entry["id"] for entry in entries if entry["priority"] == "minimum"]
    checks["final_minimum_items_satisfied"] = bool(minimum_ids) and all(
        final_items.get(identifier, {}).get("status") == "satisfied" for identifier in minimum_ids
    )
    final_audit = audit_events[-1].get("detail", {}) if audit_events else {}
    checks["final_audit_passes"] = (
        final_audit.get("pass") is True
        and not final_audit.get("inconclusive")
        and not final_audit.get("refinement_exhausted")
    )
    answer = result.get("answer", "").strip()
    checks["nonempty_final_answer"] = bool(answer)
    checks["checklist_not_published"] = "acceptance_criterion" not in answer
    word_count = sum(bool(re.search(r"\w", word)) for word in answer.split())
    if case["id"] == "headed-comparison":
        checks["final_word_limit"] = word_count < 350
        checks["final_ending"] = answer.endswith("Decision: defer.")
        checks["final_compares_infeasible_options"] = (
            all(
                re.search(pattern, answer, re.IGNORECASE)
                for pattern in (
                    r"\bA\b",
                    r"\bB\b",
                    r"\b60\b",
                    r"\b50\b",
                    r"(neither|infeasib|not.{0,30}feasible|feasible set is empty)",
                    r"assum",
                    r"compar|approach|perspective",
                )
            )
            and len(re.findall(r"\bif\b", answer, re.IGNORECASE)) >= 2
        )
    elif case["id"] == "headless-json":
        try:
            obj = json.loads(answer)
            steps = obj["next_steps"]
            checks["final_json_contract"] = (
                set(obj) == {"feasible", "choice", "violations", "next_steps"}
                and obj["feasible"] is False
                and obj["choice"] is None
                and obj["violations"] == {"A": "latency", "B": "budget"}
                and isinstance(steps, list)
                and len(steps) == 2
                and steps[0] != steps[1]
                and all(
                    isinstance(step, str)
                    and re.search(r"\b(if|when|provided)\b", step, re.IGNORECASE)
                    for step in steps
                )
            )
        except (KeyError, TypeError, ValueError):
            checks["final_json_contract"] = False
    elif case["id"] == "image-requirements":
        images = [event for event in events if event.get("node") == "image_description"]
        descriptions = [item for item in outputs if item["node"] == "image_description"]
        checks["image_root_and_checklist_overlap"] = (
            len(images) == len(req_events) == 1
            and images[0].get("status") == "success"
            and _overlap(images[0], req_events[0])
        )
        checks["image_description_nonempty"] = len(descriptions) == 1 and bool(
            descriptions[0]["output"]
        )
        checks["final_word_limit"] = word_count < 250
        checks["final_ending"] = answer.endswith("Health status: unknown.")
        checks["final_image_constraints"] = all(
            re.search(pattern, answer, re.IGNORECASE)
            for pattern in (
                r"\bred\b",
                r"health",
                r"(text|label)",
                r"shape",
                r"\b(if|when|provided)\b",
            )
        )
    failures.extend(name for name, passed in checks.items() if not passed)
    return {
        "case": case["id"],
        "checks": checks,
        "failures": failures,
        "passed": all(checks.values()),
        "word_count": word_count,
        "requirements_tokens": spent,
        "checklist": entries,
        "stage_outputs": outputs,
        "ttft_s": result.get("ttft_s"),
        "e2e_s": result.get("e2e_s"),
    }


def _request(
    case: dict, directory: Path, *, base_url: str, seed: int, requirement_cap: int
) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    body = {**case["request"], "seed": seed}
    (directory / "request.json").write_text(json.dumps(body, indent=2) + "\n")
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "X-Kairyu-Trace": "1"},
    )
    start = time.monotonic()
    result = {"answer": "", "reasoning": "", "trace": None, "ttft_s": None}
    traces = 0
    done = False
    try:
        with (
            urllib.request.urlopen(request, timeout=1800) as response,
            (directory / "response.sse").open("w") as raw,
        ):
            for line in response:
                text = line.decode()
                raw.write(text)
                raw.flush()
                if not text.startswith("data:"):
                    continue
                if text.strip() == "data: [DONE]":
                    done = True
                    continue
                event = json.loads(text[5:])
                if "kairyu_trace_v2" in event:
                    traces += 1
                    result["trace"] = event["kairyu_trace_v2"]
                for choice in event.get("choices", []):
                    delta = choice.get("delta", {})
                    if delta.get("content"):
                        if result["ttft_s"] is None:
                            result["ttft_s"] = time.monotonic() - start
                        result["answer"] += delta["content"]
                    result["reasoning"] += delta.get("reasoning_content") or ""
        result["e2e_s"] = time.monotonic() - start
        report = validate_result(case, result, requirement_cap=requirement_cap)
        report["checks"]["complete_sse"] = done and traces == 1
        report["passed"] = all(report["checks"].values())
        if not report["checks"]["complete_sse"]:
            report["failures"].append("complete_sse")
    except (OSError, ValueError, urllib.error.HTTPError) as error:
        report = {"case": case["id"], "passed": False, "error": str(error)}
    (directory / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    report["seed"] = seed
    report["directory"] = directory.name
    (directory / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"{directory.name}: {'PASS' if report['passed'] else 'FAIL'} "
        f"{report.get('failures', report.get('error', ''))}",
        flush=True,
    )
    return report


def run_quality(run_dir: Path, *, base_url: str, requirement_cap: int) -> int:
    """Repeat each original counterexample: one serial and two parallel passes."""
    cases = json.loads((HERE / "requirements-quality-cases.json").read_text())["cases"]
    run_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for repetition in range(3):
        concurrency = 1 if repetition == 0 else len(cases)
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [
                pool.submit(
                    _request,
                    case,
                    run_dir / f"r{repetition}-{case['id']}",
                    base_url=base_url,
                    seed=595 + repetition,
                    requirement_cap=requirement_cap,
                )
                for case in cases
            ]
            for future in concurrent.futures.as_completed(futures):
                report = future.result()
                report["repetition"] = repetition
                report["concurrency"] = concurrency
                reports.append(report)
                (run_dir / "summary.json").write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "expected_requests": len(cases) * 3,
                            "passed": len(reports) == len(cases) * 3
                            and all(row["passed"] for row in reports),
                            "reports": reports,
                        },
                        indent=2,
                    )
                    + "\n"
                )
    return 0 if all(report["passed"] for report in reports) else 1


def main() -> None:
    from kairyu.dsl.loader import load_spec

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    args = parser.parse_args()
    spec = load_spec(HERE / "auto-max.yaml")
    requirements = next(role for role in spec.roles if role.name == "requirements")
    raise SystemExit(
        run_quality(
            args.run_dir,
            base_url=args.base_url,
            requirement_cap=requirements.sampling.max_tokens,
        )
    )


if __name__ == "__main__":
    main()
