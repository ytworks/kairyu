#!/usr/bin/env python3
"""Calibrate the verified-tool route's four angle thresholds on recorded DeepSWE turns.

VCO-D18. Each recorded reply of the DeepSWE runs below (mini-swe-agent,
``tools=[bash]``, served by kairyu-verified) is a turn: the conversation the
agent sent, the reply it got, the execution result of that reply's tool call
(the next turn's tool message) and the task's final reward.

  sample  picks about 200 turns: every turn whose call failed when run
          (nonzero returncode) or repeats an earlier command, then the other
          turns round-robin over conversation x position (early/mid/late
          thirds) strata, weighted back to the pool; splits by task into
          calibration and held-out halves; writes
          datasets/deepswe-tool-turns.json and one labelling packet per turn
          (hindsight material, never Jev's reads).
  labels  merges two independent labellers' verdicts (and a third's on
          disagreement) into the dataset.
  judge   reads every sampled reply through the production verified_tool_check
          (verified-always.yaml, the same Kairyu checklist code as serving)
          against both OpenJev replicas; DeepSeek is not involved.
  report  per angle: the smallest threshold whose accepted replies have a
          one-sided 95 % Clopper-Pearson upper bound on the NG rate <= alpha
          on the calibration half; held-out miss rate, the rate of sound
          replies sent to repair, AUROC; the same for all four angles.

Usage (after ./run.sh up): ./verify.sh calibrate-tool   (judge + report)
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import concurrent.futures
import dataclasses
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

import control  # noqa: E402
from calibrate import OPENJEV_URLS, accepted_stats, clopper_pearson_upper  # noqa: E402

from kairyu.dsl.loader import load_spec, role_spec  # noqa: E402
from kairyu.engine.openai_backend import _message_text  # noqa: E402
from kairyu.engine.systemone import HTTPSystemOneBackend  # noqa: E402
from kairyu.entrypoints.server.chat_service import (  # noqa: E402
    validate_orchestration_chat_input,
)
from kairyu.entrypoints.server.protocol import ChatCompletionRequest  # noqa: E402
from kairyu.orchestration.checklist import ChecklistConfig  # noqa: E402
from kairyu.orchestration.checklist import judge as judge_checklist  # noqa: E402

SPEC = control.SPEC
RESULTS = Path.home() / "kairyu-bench" / "results"
# The verified-tool route run (VCO-D17, max effort) and the VCO-D16 replay runs.
RUNS = (
    "deepswe-verified-tool-e-full-max-4w-20261004-r1",
    "deepswe-verified-vco15-4w-20261003-r3",
    "deepswe-verified-vco15-4w-20261003-r4",
    "deepswe-verified-full-max-4w-20261003-r1",
)
DATASET = HERE / "datasets" / "deepswe-tool-turns.json"
ANGLES = ("first_call", "order", "progress", "runs")
# mini-swe-agent's tool (the relay does not record tool definitions).
BASH = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute a bash command",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The bash command to execute"}
            },
            "required": ["command"],
        },
    },
}
TARGET_TURNS = 200
SEED = 20261005


def _messages(root: Path, ids: list[str]) -> list[dict]:
    return [json.loads((root / "messages" / f"{mid}.json").read_text()) for mid in ids]


def _command(message: dict) -> str | None:
    for call in message.get("tool_calls") or ():
        arguments = (call.get("function") or {}).get("arguments") or ""
        try:
            return json.loads(arguments).get("command")
        except (ValueError, AttributeError):
            return arguments
    return None


def _returncode(message: dict) -> int | None:
    try:
        return int(json.loads(message.get("content") or "").get("returncode"))
    except (ValueError, TypeError, AttributeError):
        return None


def _route(response: dict) -> str | None:
    """The route judge's label for this turn, from the trace."""

    for event in (response.get("kairyu_trace_v2") or {}).get("events", []):
        if event.get("node") == "profile_judge":
            return (event.get("detail") or {}).get("verdict")
    return None


def _outcomes(run: str) -> dict[str, float]:
    """Task reward by the first user message of its trial."""

    rewards = {}
    jobs = RESULTS / run / "raw" / "deepswe" / "jobs"
    for result in jobs.glob("*/*/result.json"):
        data = json.loads(result.read_text())
        reward = ((data.get("verifier_result") or {}).get("rewards") or {}).get("reward")
        trajectory = result.parent / "agent" / "mini-swe-agent.trajectory.json"
        if reward is None or not trajectory.is_file():
            continue
        messages = json.loads(trajectory.read_text())["messages"]
        first = next(m["content"] for m in messages if m["role"] == "user")
        rewards[first] = float(reward)
    return rewards


def turns() -> list[dict]:
    """Every recorded reply with a tool call whose execution result was recorded."""

    rows = []
    for run in RUNS:
        root = RESULTS / f"{run}-telemetry"
        rewards = _outcomes(run)
        calls = [json.loads(path.read_text()) for path in sorted((root / "calls").glob("*.json"))]
        calls = [
            c for c in calls if c.get("phase") == "benchmark" and c.get("status") == "completed"
        ]
        by_conversation = collections.defaultdict(list)
        for call in calls:
            by_conversation[call["conversation_id"]].append(call)
        for conversation, group in by_conversation.items():
            group.sort(key=lambda c: len(c["request_message_ids"]))
            replies = []
            for call in group:
                message = ((call.get("response") or {}).get("choices") or [{}])[0].get("message")
                if not message or not message.get("tool_calls"):
                    continue
                ids = call["request_message_ids"]
                later = next(
                    (
                        c
                        for c in group
                        if len(c["request_message_ids"]) > len(ids) + 1
                        and c["request_message_ids"][: len(ids)] == ids
                    ),
                    None,
                )
                if later is None:
                    continue
                result_id = later["request_message_ids"][len(ids) + 1]
                result = _messages(root, [result_id])[0]
                if result.get("role") != "tool":
                    continue
                replies.append((call, message, result, result_id))
            if not replies:
                continue
            first_user = next(
                m["content"]
                for m in _messages(root, replies[0][0]["request_message_ids"][:3])
                if m.get("role") == "user"
            )
            task = first_user.split("\n", 1)[0][:120]
            commands = []
            for index, (call, message, result, result_id) in enumerate(replies):
                command = _command(message)
                rows.append(
                    {
                        "id": call["id"],
                        "run": run,
                        "conversation": conversation,
                        "task": task,
                        "reward": rewards.get(first_user),
                        "turn": index,
                        "turns": len(replies),
                        "position": ("early", "mid", "late")[min(2, 3 * index // len(replies))],
                        "route": _route(call["response"]),
                        "returncode": _returncode(result),
                        "repeated": command is not None and command in commands,
                        "result_id": result_id,
                    }
                )
                commands.append(command)
    for row in rows:
        # A recorded failure of this very call: every such turn is sampled.
        row["signal"] = bool((row["returncode"] not in (0, None)) or row["repeated"])
    return rows


def sample(rows: list[dict]) -> list[dict]:
    """Every turn with a failure signal (weight 1), then the other turns
    round-robin over conversation x position strata up to TARGET_TURNS, each
    weighted by its stratum size over the turns picked from it."""

    rng = random.Random(SEED)
    chosen = [{**row, "weight": 1.0} for row in rows if row["signal"]]
    strata = collections.defaultdict(list)
    for row in rows:
        if not row["signal"]:
            strata[(row["run"], row["conversation"], row["position"])].append(row)
    for members in strata.values():
        rng.shuffle(members)
    picked = collections.defaultdict(list)
    keys = sorted(strata)
    rng.shuffle(keys)
    depth = 0
    while len(chosen) + sum(map(len, picked.values())) < TARGET_TURNS:
        added = False
        for key in keys:
            if (
                depth < len(strata[key])
                and len(chosen) + sum(map(len, picked.values())) < TARGET_TURNS
            ):
                picked[key].append(strata[key][depth])
                added = True
        if not added:
            break
        depth += 1
    for key, rows_ in picked.items():
        chosen += [{**row, "weight": len(strata[key]) / len(rows_)} for row in rows_]
    chosen.sort(key=lambda r: (r["run"], r["conversation"], r["turn"]))
    # By task, balancing the turns of the two halves.
    sizes = collections.Counter(row["task"] for row in chosen)
    tasks = sorted(sizes)
    rng.shuffle(tasks)
    halves = {"calibration": set(), "holdout": set()}
    for task in tasks:
        smaller = min(halves, key=lambda name: sum(sizes[t] for t in halves[name]))
        halves[smaller].add(task)
    for row in chosen:
        row["split"] = "calibration" if row["task"] in halves["calibration"] else "holdout"
    return chosen


def _call(row: dict) -> tuple[list[dict], dict, dict]:
    root = RESULTS / f"{row['run']}-telemetry"
    call = json.loads((root / "calls" / f"{row['id']}.json").read_text())
    messages = _messages(root, call["request_message_ids"])
    reply = call["response"]["choices"][0]["message"]
    result = _messages(root, [row["result_id"]])[0]
    return messages, reply, result


def _clip(text: str | None, limit: int) -> str:
    text = text or ""
    return text if len(text) <= limit else text[: limit // 2] + "\n[...]\n" + text[-limit // 2 :]


def packet(row: dict) -> str:
    """Hindsight material for a labeller (never Jev's reads): the task, every
    call of the run with its returncode, the steps around the reply in
    detail, the reply and its execution result."""

    messages, reply, result = _call(row)
    root = RESULTS / f"{row['run']}-telemetry"
    calls = [json.loads(p.read_text()) for p in (root / "calls").glob("*.json")]
    longest = max(
        (c for c in calls if c.get("conversation_id") == row["conversation"]),
        key=lambda c: len(c["request_message_ids"]),
    )
    trajectory = _messages(root, longest["request_message_ids"])
    target = len(messages)  # the reply's index in the trajectory
    overview, detail = [], []
    for index, message in enumerate(trajectory):
        if message.get("role") != "assistant":
            continue
        following = trajectory[index + 1] if index + 1 < len(trajectory) else {}
        mark = ">>>" if index == target else "   "
        overview.append(
            f"{mark} m{index} rc={_returncode(following)} {_clip(_command(message), 160)!r}"
        )
        if target - 12 <= index <= target + 16 and index != target:
            when = "BEFORE" if index < target else "AFTER"
            detail.append(
                f"--- m{index} ({when} the reply)\nTEXT: {_clip(message.get('content'), 500)}\n"
                f"CALL: {_clip(_command(message), 1200)}\n"
                f"RESULT: {_clip(following.get('content'), 700)}"
            )
    task = next(m["content"] for m in messages if m.get("role") == "user")
    return "\n".join(
        [
            f"# Turn {row['id']} (reply {row['turn'] + 1} of {row['turns']}, message m{target})",
            f"Task final reward: {row['reward']} (1 = solved, 0 = not, None = unknown)",
            "## Task",
            _clip(task, 6000),
            "## Every call of the run (>>> = the reply being labelled)",
            "\n".join(overview),
            "## Steps around the reply",
            "\n".join(detail),
            "## THE REPLY BEING LABELLED",
            f"TEXT: {_clip(reply.get('content'), 4000)}",
            f"CALL: {_clip(_command(reply), 6000)}",
            "## Its actual execution result",
            _clip(result.get("content"), 6000),
        ]
    )


def _checklist() -> ChecklistConfig:
    spec = load_spec(HERE / "verified-always.yaml")
    profile = next(p for p in spec.profiles if p.name == "verified_tool")
    node = next(role for role in profile.roles if role.name == "verified_tool_check")
    config = role_spec(node).checklist
    assert config is not None and [q.id for q in config.questions] == list(ANGLES)
    return config


def _query(messages: list[dict]) -> str:
    chat = ChatCompletionRequest(model=SPEC["public_models"][0], messages=messages, tools=[BASH])
    return validate_orchestration_chat_input(chat).prompt


def read(row: dict, config: ChecklistConfig) -> dict[str, float]:
    """P(angle met) for the recorded reply, via the production checklist."""

    messages, reply, _result = _call(row)
    outputs = {"verified_tool_answer": _message_text(reply)}

    async def run():
        backend = HTTPSystemOneBackend(
            base_urls=OPENJEV_URLS, upstream_model=SPEC["systemone"]["model"], timeout_s=600
        )
        try:
            return await judge_checklist(config, backend, outputs, _query(messages), (BASH,))
        finally:
            await backend.shutdown()

    verdict = asyncio.run(run())
    return {item.id: item.p for item in verdict.items}


def auroc(pairs: list[tuple[float, int]]) -> float | None:
    good = [p for p, label in pairs if label == 1]
    bad = [p for p, label in pairs if label == 0]
    if not good or not bad:
        return None
    wins = sum((g > b) + 0.5 * (g == b) for g in good for b in bad)
    return wins / (len(good) * len(bad))


def _tau(pairs: list[tuple[float, int]], alpha: float, confidence: float) -> dict:
    """The smallest tau meeting alpha; else the tau with the fewest errors."""

    for tau in sorted({p for p, _ in pairs}):
        stats = accepted_stats(pairs, tau, confidence)
        if stats["accepted"] and stats["upper_bound"] <= alpha:
            return {**stats, "meets_alpha": True}
    candidates = sorted({p for p, _ in pairs} | {1.01})
    best = min(
        candidates,
        key=lambda t: sum((p >= t) != bool(label) for p, label in pairs),
    )
    return {**accepted_stats(pairs, best, confidence), "meets_alpha": False}


def _measure(rows: list[dict], taus: dict[str, float], confidence: float) -> dict:
    """Held-out style figures, unweighted and selection-weighted."""

    out = {}
    for name in (*ANGLES, "all"):
        accepted = bad_accepted = good = good_rejected = 0
        w_accepted = w_bad_accepted = w_good = w_good_rejected = 0.0
        for row in rows:
            if name == "all":
                passed = all(row["p"][a] >= taus[a] for a in ANGLES)
                ok = all(row["label"][a] == 1 for a in ANGLES)
            else:
                passed = row["p"][name] >= taus[name]
                ok = row["label"][name] == 1
            weight = row["weight"]
            if passed:
                accepted += 1
                w_accepted += weight
                if not ok:
                    bad_accepted += 1
                    w_bad_accepted += weight
            if ok:
                good += 1
                w_good += weight
                if not passed:
                    good_rejected += 1
                    w_good_rejected += weight
        out[name] = {
            "accepted": accepted,
            "ng_accepted": bad_accepted,
            "miss_rate": bad_accepted / accepted if accepted else None,
            "miss_upper_bound": clopper_pearson_upper(bad_accepted, accepted, confidence),
            "sound_sent_to_repair": good_rejected / good if good else None,
            "weighted_miss_rate": w_bad_accepted / w_accepted if w_accepted else None,
            "weighted_sound_sent_to_repair": w_good_rejected / w_good if w_good else None,
        }
    return out


def report(dataset: list[dict]) -> dict:
    calibration = SPEC["calibration"]
    alpha, confidence = float(calibration["alpha"]), float(calibration["confidence"])
    rows = [row for row in dataset if row.get("label") and row.get("p")]
    halves = {s: [r for r in rows if r["split"] == s] for s in ("calibration", "holdout")}
    out = {"alpha": alpha, "confidence": confidence, "turns": {}, "angles": {}}
    for split, part in halves.items():
        out["turns"][split] = {
            "turns": len(part),
            "tasks": len({r["task"] for r in part}),
            "ng": {a: sum(1 for r in part if r["label"][a] == 0) for a in ANGLES},
        }
    taus = {}
    for angle in ANGLES:
        pairs = {
            s: [(r["p"][angle], r["label"][angle]) for r in part] for s, part in halves.items()
        }
        chosen = _tau(pairs["calibration"], alpha, confidence)
        taus[angle] = chosen["tau"]
        out["angles"][angle] = {
            "calibration": chosen,
            "auroc": {s: auroc(v) for s, v in pairs.items()},
        }
    out["thresholds"] = taus
    out["holdout"] = _measure(halves["holdout"], taus, confidence)
    out["calibration_fit"] = _measure(halves["calibration"], taus, confidence)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("step", choices=("sample", "labels", "judge", "report", "gate"))
    parser.add_argument("--packets", type=Path, help="sample: directory for labelling packets")
    parser.add_argument("--verdicts", type=Path, nargs="*", help="labels: labeller JSON files")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    if args.step == "sample":
        chosen = sample(turns())
        DATASET.write_text(json.dumps(chosen, indent=1, ensure_ascii=False) + "\n")
        if args.packets:
            args.packets.mkdir(parents=True, exist_ok=True)
            for row in chosen:
                (args.packets / f"{row['id']}.md").write_text(packet(row))
        counts = collections.Counter((r["split"], r["signal"]) for r in chosen)
        print(f"{len(chosen)} turns, {len({r['task'] for r in chosen})} tasks: {dict(counts)}")
        return

    dataset = json.loads(DATASET.read_text())
    if args.step == "labels":
        # Each file: [{"id", "labeller", "labels": {angle: 0|1}, "reasons": {angle: str}}].
        verdicts = collections.defaultdict(list)
        for path in args.verdicts or ():
            for item in json.loads(path.read_text()):
                verdicts[item["id"]].append(item)
        agree = collections.Counter()
        for row in dataset:
            votes = verdicts.get(row["id"], [])
            row["labellers"] = votes
            if len(votes) < 2:
                continue
            label = {}
            for angle in ANGLES:
                values = [v["labels"][angle] for v in votes]
                agree[angle] += values[0] == values[1]
                label[angle] = int(sum(values) * 2 > len(values)) if len(values) % 2 else None
                if label[angle] is None and values[0] == values[1]:
                    label[angle] = values[0]
            row["label"] = label if None not in label.values() else None
        DATASET.write_text(json.dumps(dataset, indent=1, ensure_ascii=False) + "\n")
        labelled = sum(1 for r in dataset if r.get("label"))
        print(f"labelled {labelled}/{len(dataset)}; first-two agreement {dict(agree)}")
        return

    if args.step in ("judge", "gate"):
        config = dataclasses.replace(_checklist(), acceptance=None)
        pending = [row for row in dataset if not row.get("p")]
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            for row, p in zip(pending, pool.map(lambda r: read(r, config), pending), strict=True):
                row["p"] = p
        DATASET.write_text(json.dumps(dataset, indent=1, ensure_ascii=False) + "\n")
        print(f"read {len(pending)} replies")
    if args.step in ("report", "gate"):
        out = report(dataset)
        path = control.environment_storage() / "calibration" / "tool-angles.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=2))
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
