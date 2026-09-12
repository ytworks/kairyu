"""Prove cancellation cleanup using native worker gauges and recovery.

Run only in a coordinated idle window. Native defaults exercise DS ranks0/1/2.
Public mode requires --public-request JSON plus exactly three --worker-metrics
entries, e.g. deepseek=http://127.0.0.1:8009/metrics and
qwen-0=docker://CONTAINER. Docker sources execute only a fixed loopback metrics
GET inside an existing container; this script never starts/stops services.

Public prompts target the primary route through content, but truncated SSE
cannot prove its final trace. Public results prove worker cleanup only; route
coverage must be verified separately from server traces. DeepSeek and at least
one Qwen replica must be running at close; all three workers must become idle.
Public mode can cancel before visible output during internal worker activity.

The public-audit suite additionally requires --audit-log qwen-0=SOURCE and
--audit-log qwen-1=SOURCE (docker://container or a timestamped local log file).
It ignores existing records and requires one fresh fixed-high audit dispatch,
then a later SSE keepalive and exactly its Qwen worker running while all other
workers are idle. Optional --expected-audit-sha256 pins a previously known
messages hash; otherwise the fresh observed hash is saved as evidence. This
proves the isolated deferred-audit cancellation window, not audit answer quality.

"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import re
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

METRIC = re.compile(r"^vllm:num_requests_(running|waiting)(?:\{([^}]*)\})?\s+(\S+)(?:\s+\S+)?$")
LABEL = re.compile(r'(\w+)="((?:\\.|[^"\\])*)"')


def stamp():
    return {"utc": datetime.now(UTC).isoformat(), "monotonic_s": time.monotonic()}


def dump(path, body):
    path.write_text(json.dumps(body, indent=2) + "\n")


def parse_metrics(raw, *, rank_label):
    rows = {}
    for line in raw.splitlines():
        match = METRIC.match(line)
        if not match:
            continue
        gauge, labels, number = match.groups()
        labels = dict(LABEL.findall(labels or ""))
        rank = labels.get(rank_label, "_aggregate")
        value = float(number)
        if not math.isfinite(value) or value < 0 or not value.is_integer():
            raise ValueError("Invalid request gauge value")
        rows.setdefault(rank, {}).setdefault(gauge, 0)
        rows[rank][gauge] += int(value)
    if not rows or any(set(row) != {"running", "waiting"} for row in rows.values()):
        raise ValueError("Missing running/waiting gauges")
    return rows


def idle(rows):
    return all(value == 0 for row in rows.values() for value in row.values())


def rank_running(rows, rank):
    if str(rank) not in rows:
        raise ValueError(f"Missing per-rank metrics for rank {rank}")
    return rows[str(rank)]["running"] == 1


def verdict(result):
    visible = result.get("suite") in {"public", "public-audit"} or result.get("first_visible")
    if result.get("suite") == "public-audit" and not result.get("live_audit"):
        return False
    return (
        bool(visible)
        and all(
            result.get(key)
            for key in (
                "observed_active",
                "client_closed",
                "server_idle_after_close",
                "recovery_passed",
            )
        )
        and not result.get("stream_ended_before_close")
    )


def active_workers(values):
    return sorted(
        name
        for name, worker in values.items()
        if sum(row["running"] for row in worker.values()) > 0
    )


def public_active(names):
    return "deepseek" in names and bool({"qwen-0", "qwen-1"}.intersection(names))


AUDIT_HOOK = re.compile(
    r"Kairyu role hook: role=audit max_tokens=(\d+) effort=(\w+) "
    r"messages_sha256=([0-9a-f]{64}) thinking_token_budget=(\d+|None)(?:\s|$)"
)


def audit_hooks(raw, worker, started_utc, expected_hash=None):
    """Parse only timestamped audit dispatches strictly after this request began."""
    started = datetime.fromisoformat(started_utc.replace("Z", "+00:00"))
    found = []
    for line in raw.splitlines():
        if "Kairyu role hook: role=audit " not in line:
            continue
        try:
            logged = datetime.fromisoformat(line.split()[0].replace("Z", "+00:00"))
            if logged.tzinfo is None:
                raise ValueError("missing timezone")
        except ValueError as error:
            raise ValueError(
                "Audit log requires an ISO UTC timestamp (docker --timestamps)"
            ) from error
        if logged <= started:
            continue
        match = AUDIT_HOOK.search(line)
        if match is None:
            raise ValueError("Malformed audit hook fields")
        cap, effort, digest, thinking_budget = match.groups()
        if effort != "high" or int(cap) <= 0:
            raise ValueError("Audit hook must use fixed high effort and positive output cap")
        if expected_hash is not None and digest != expected_hash:
            raise ValueError("Fresh audit hook messages hash differs from expected hash")
        found.append(
            {
                "worker": worker,
                "logged_at": logged.isoformat(),
                "raw": line,
                "messages_sha256": digest,
                "max_tokens": int(cap),
                "effort": effort,
                "thinking_token_budget": None
                if thinking_budget == "None"
                else int(thinking_budget),
                "observed": stamp(),
            }
        )
    return found


def audit_active(hooks, values, keepalive):
    if len(hooks) > 1:
        raise ValueError(
            "Live audit requires one unique fresh audit hook; multiple dispatches observed"
        )
    if not hooks or not keepalive:
        return False
    hook = hooks[0]
    if keepalive["monotonic_s"] <= hook["observed"]["monotonic_s"]:
        return False
    if set(values) != {"deepseek", "qwen-0", "qwen-1"}:
        return False
    for name, rows in values.items():
        if name == hook["worker"]:
            if sum(row["running"] for row in rows.values()) != 1:
                return False
            if any(row["waiting"] for row in rows.values()):
                return False
        elif not idle(rows):
            return False
    return True


def audit_cursors(sources):
    cursors = {}
    for worker, source in sources.items():
        cursor = {"seen": "", "consumed": 0}
        if not source.startswith("docker://"):
            info = Path(source).stat()
            cursor.update(offset=info.st_size, inode=info.st_ino, device=info.st_dev)
        cursors[worker] = cursor
    return cursors


async def fresh_audit_hooks(sources, cursors, started_utc, directory, expected_hash=None):
    found = []
    for worker, source in sources.items():
        cursor = cursors[worker]
        if source.startswith("docker://"):
            container = source[len("docker://") :]
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", container):
                raise ValueError("Invalid audit log container name")
            process = await asyncio.create_subprocess_exec(
                "docker",
                "logs",
                "--timestamps",
                "--since",
                started_utc,
                container,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            try:
                out, _ = await asyncio.wait_for(process.communicate(), 5)
            except BaseException:
                if process.returncode is None:
                    process.kill()
                    await process.wait()
                raise
            if process.returncode:
                raise RuntimeError(out.decode())
            raw = out.decode()
        else:
            path = Path(source)
            info = path.stat()
            if (info.st_dev, info.st_ino) != (cursor["device"], cursor["inode"]):
                raise ValueError("Audit log rotated during probe")
            with path.open("rb") as log:
                log.seek(cursor["offset"])
                raw = log.read(8 * 1024 * 1024 + 1).decode()
        if len(raw.encode()) > 8 * 1024 * 1024:
            raise ValueError("Audit log exceeds bounded 8MiB window")
        if not raw.startswith(cursor["seen"]):
            raise ValueError("Audit log truncated or changed during probe")
        cursor["seen"] = raw
        end = raw.rfind("\n") + 1
        fresh = raw[cursor["consumed"] : end]
        cursor["consumed"] = end
        if fresh:
            with (directory / f"{worker}-fresh.log").open("a") as log:
                log.write(fresh)
            found.extend(audit_hooks(fresh, worker, started_utc, expected_hash))
    return found


async def metric_text(client, source):
    if source.startswith("docker://"):
        container = source[len("docker://") :]
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", container):
            raise ValueError("docker:// must contain one existing container name")
        script = (
            "import urllib.request; print(urllib.request.urlopen("
            '"http://127.0.0.1:8000/metrics",timeout=3).read().decode())'
        )
        process = await asyncio.create_subprocess_exec(
            "docker",
            "exec",
            container,
            "python3",
            "-c",
            script,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            out, error = await asyncio.wait_for(process.communicate(), 5)
        except BaseException:
            if process.returncode is None:
                process.kill()
                await process.wait()
            raise
        if process.returncode:
            raise RuntimeError(error.decode())
        return out.decode()
    response = await client.get(source, timeout=5)
    response.raise_for_status()
    return response.text


async def snapshot(client, sources, directory, phase, rank_label):
    bodies = await asyncio.gather(*(metric_text(client, source) for source in sources.values()))
    now = stamp()
    values = {
        name: parse_metrics(raw, rank_label=rank_label)
        for name, raw in zip(sources, bodies, strict=True)
    }
    inventory = {name: sorted(rows) for name, rows in values.items()}
    inventory_path = directory / "metrics-inventory.json"
    if set(inventory.get("deepseek", [])) != {"0", "1", "2"}:
        raise ValueError("DeepSeek metrics must include exactly ranks 0, 1, 2")
    if inventory_path.exists():
        if inventory != json.loads(inventory_path.read_text()):
            raise ValueError("Worker metric rank inventory changed")
    else:
        dump(inventory_path, inventory)
    with (directory / "metrics.jsonl").open("a") as log:
        log.write(
            json.dumps(
                {
                    "phase": phase,
                    **now,
                    "values": values,
                    "raw": dict(zip(sources, bodies, strict=True)),
                }
            )
            + "\n"
        )
    return values


async def wait_idle(client, sources, directory, phase, args):
    deadline = time.monotonic() + args.cleanup_timeout
    idle_since = None
    while time.monotonic() < deadline:
        values = await snapshot(client, sources, directory, phase, args.rank_label)
        if all(idle(row) for row in values.values()):
            idle_since = time.monotonic() if idle_since is None else idle_since
            if time.monotonic() - idle_since >= args.idle_stability:
                return stamp()
        else:
            idle_since = None
        await asyncio.sleep(args.poll_interval)
    raise TimeoutError(f"Workers did not become stably idle during {phase}")


async def stream(client, endpoint, body, headers, directory, state):
    async with client.stream("POST", endpoint, json=body, headers=headers) as response:
        dump(
            directory / "response-http.json",
            {"status": response.status_code, "headers": dict(response.headers), **stamp()},
        )
        response.raise_for_status()
        with (directory / "response.sse").open("w") as raw:
            async for line in response.aiter_lines():
                raw.write(line + "\n")
                raw.flush()
                if line.startswith(": status "):
                    state["last_keepalive"] = stamp()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    state["terminal_sse"] = stamp()
                    continue
                if not data:
                    continue
                chunk = json.loads(data)
                if chunk.get("error"):
                    raise ValueError(str(chunk["error"]))
                if any(c.get("delta", {}).get("content") for c in chunk.get("choices", [])):
                    if not state.get("first_visible"):
                        state["first_visible"] = stamp()
                if any(c.get("finish_reason") for c in chunk.get("choices", [])):
                    state["finish_event"] = stamp()
    state["natural_stream_end"] = stamp()


async def run_case(client, args, sources, rank):
    directory = args.output / (f"native-rank{rank}" if rank is not None else args.suite)
    directory.mkdir()
    state = {"suite": args.suite, "rank": rank, "started": stamp(), "passed": False}
    task = None
    headers = {"X-data-parallel-rank": str(rank)} if rank is not None else {"X-Kairyu-Trace": "1"}
    body = (
        {
            "model": args.model,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Write a detailed 5000-word tutorial on prime numbers, "
                        "with many worked examples."
                    ),
                }
            ],
            "max_tokens": 8192,
            "min_tokens": 8192,
            "ignore_eos": True,
            "stream": True,
            "temperature": 0,
            "chat_template_kwargs": {"thinking": False, "enable_thinking": False},
        }
        if rank is not None
        else json.loads(args.public_request.read_text())
    )
    body["stream"] = True
    dump(directory / "request.json", body)
    dump(directory / "request-headers.json", headers)
    try:
        state["baseline_idle"] = await wait_idle(client, sources, directory, "before", args)
        audit_mode = args.suite == "public-audit"
        hooks = []
        cursors = audit_cursors(args.audit_logs) if audit_mode else None
        state["request_started"] = stamp()
        if audit_mode:
            dump(directory / "audit-log-cursors.json", cursors)
        task = asyncio.create_task(
            stream(
                client,
                args.base_url.rstrip("/") + "/chat/completions",
                body,
                headers,
                directory,
                state,
            )
        )
        seen_active = set()
        deadline = time.monotonic() + args.activation_timeout
        while time.monotonic() < deadline:
            if task.done():
                await task
                raise ValueError("Request ended naturally before cancellation observation")
            if audit_mode:
                hooks.extend(
                    await fresh_audit_hooks(
                        args.audit_logs,
                        cursors,
                        state["request_started"]["utc"],
                        directory,
                        args.expected_audit_sha256,
                    )
                )
                dump(directory / "audit-hooks.json", hooks)
                audit_keepalive = state.get("last_keepalive")
            values = await snapshot(client, sources, directory, "active", args.rank_label)
            names = active_workers(values)
            seen_active.update(names)
            state["workers_seen_active"] = sorted(seen_active)
            if audit_mode:
                active = audit_active(hooks, values, audit_keepalive)
            else:
                active = (
                    rank_running(values["deepseek"], rank)
                    if rank is not None
                    else public_active(names)
                )
            if active and (rank is None or state.get("first_visible")):
                if audit_mode:
                    state["live_audit"] = {
                        "hook": hooks[0],
                        "keepalive": audit_keepalive,
                        "metrics": values,
                        **stamp(),
                    }
                state["workers_active_at_close"] = names
                state["all_three_active_at_close"] = len(names) == 3
                state["observed_active"] = {"metrics": values, **stamp()}
                break
            await asyncio.sleep(args.poll_interval)
        if not state.get("observed_active"):
            raise TimeoutError("No simultaneous visible output and active-worker observation")
        state["stream_ended_before_close"] = bool(
            state.get("natural_stream_end")
            or state.get("finish_event")
            or state.get("terminal_sse")
            or task.done()
        )
        state["close_started"] = stamp()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        state["client_closed"] = stamp()
        state["server_idle_after_close"] = await wait_idle(
            client, sources, directory, "after", args
        )
        state["cleanup_ms"] = (
            state["server_idle_after_close"]["monotonic_s"] - state["client_closed"]["monotonic_s"]
        ) * 1000
        recovery = {
            "model": args.model,
            "messages": [
                {"role": "user", "content": "What is 17 times 19? Return only the integer."}
            ],
            "max_tokens": 256,
            "temperature": 0,
            "stream": False,
        }
        if rank is not None:
            recovery["chat_template_kwargs"] = {"thinking": False, "enable_thinking": False}
        dump(directory / "recovery-request.json", recovery)
        async with asyncio.timeout(args.recovery_timeout):
            response = await client.post(
                args.base_url.rstrip("/") + "/chat/completions", json=recovery, headers=headers
            )
            (directory / "recovery-response.raw").write_bytes(response.content)
            response.raise_for_status()
            answer = response.json()
        choice = answer.get("choices", [{}])[0]
        state["recovery_passed"] = (
            choice.get("finish_reason") == "stop"
            and choice.get("message", {}).get("content", "").strip() == "323"
        )
        state["recovery_idle"] = await wait_idle(client, sources, directory, "recovery", args)
        state["passed"] = verdict(state)
        if rank is None:
            state["primary_route_verified"] = False
            state["scope"] = "Worker cleanup only; independently verify primary trace coverage"
    except Exception as error:
        state["error"] = f"{type(error).__name__}: {error}"
    finally:
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        state["ended"] = stamp()
        dump(directory / "result.json", state)
    return state


async def main_async(args, sources):
    args.output.mkdir(parents=True, exist_ok=False)
    dump(
        args.output / "provenance.json",
        {
            "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "worker_metrics": sources,
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "idle_window_required": True,
            "started": stamp(),
        },
    )
    results = []
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(args.activation_timeout, connect=5),
        limits=httpx.Limits(max_connections=8),
    ) as client:
        for rank in args.ranks if args.suite == "native" else [None]:
            result = await run_case(client, args, sources, rank)
            results.append(result)
            dump(
                args.output / "results.json",
                {"passed": all(r["passed"] for r in results), "cases": results},
            )
            print(
                json.dumps(
                    {
                        "rank": rank,
                        "passed": result["passed"],
                        "cleanup_ms": result.get("cleanup_ms"),
                        "error": result.get("error"),
                    }
                ),
                flush=True,
            )
            if not result["passed"]:
                break
    return int(not all(r["passed"] for r in results))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["native", "public", "public-audit"], default="native")
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--ranks", nargs="+", type=int, choices=[0, 1, 2], default=[0, 1, 2])
    parser.add_argument("--rank-label", default="engine")
    parser.add_argument("--worker-metrics", action="append", default=[], metavar="NAME=URL")
    parser.add_argument("--public-request", type=Path)
    parser.add_argument(
        "--audit-log",
        action="append",
        default=[],
        metavar="WORKER=SOURCE",
        help="public-audit: qwen-0/qwen-1=docker://container or timestamped log file",
    )
    parser.add_argument(
        "--expected-audit-sha256",
        help="Optional exact audit messages hash; otherwise record fresh observed hash",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--activation-timeout", type=float)
    parser.add_argument("--cleanup-timeout", type=float, default=30)
    parser.add_argument("--recovery-timeout", type=float)
    parser.add_argument("--poll-interval", type=float, default=0.25)
    parser.add_argument("--idle-stability", type=float, default=1)
    args = parser.parse_args()
    default_timeout = 60 if args.suite == "native" else 600
    if args.activation_timeout is None:
        args.activation_timeout = default_timeout
    if args.recovery_timeout is None:
        args.recovery_timeout = default_timeout
    args.base_url = args.base_url or (
        "http://127.0.0.1:8009/v1" if args.suite == "native" else "http://127.0.0.1:8008/v1"
    )
    args.model = args.model or (
        "deepseek-v4.1-flash" if args.suite == "native" else "kairyu-auto-max"
    )
    if (
        len(set(args.ranks)) != len(args.ranks)
        or any(
            not 0 < value <= 600
            for value in (args.activation_timeout, args.cleanup_timeout, args.recovery_timeout)
        )
        or not 0.1 <= args.poll_interval <= 5
        or not args.poll_interval <= args.idle_stability < args.cleanup_timeout
    ):
        parser.error(
            "Require unique ranks, finite timeouts<=600, poll0.1..5 and "
            "poll<=idle-stability<cleanup-timeout"
        )
    try:
        sources = dict(item.split("=", 1) for item in args.worker_metrics)
    except ValueError:
        parser.error("--worker-metrics requires NAME=URL")
    if args.suite == "native":
        sources = sources or {
            "deepseek": args.base_url.removesuffix("/").removesuffix("/v1") + "/metrics"
        }
        if set(sources) != {"deepseek"}:
            parser.error("Native mode requires one deepseek metrics source")
    elif not args.public_request or set(sources) != {"deepseek", "qwen-0", "qwen-1"}:
        parser.error("Public mode requires request JSON and deepseek/qwen-0/qwen-1 metrics sources")
    try:
        args.audit_logs = dict(item.split("=", 1) for item in args.audit_log)
    except ValueError:
        parser.error("--audit-log requires WORKER=SOURCE")
    if args.suite == "public-audit":
        if set(args.audit_logs) != {"qwen-0", "qwen-1"} or len(args.audit_log) != 2:
            parser.error("public-audit requires exactly qwen-0 and qwen-1 audit log sources")
        if args.expected_audit_sha256 and not re.fullmatch(
            r"[0-9a-f]{64}", args.expected_audit_sha256
        ):
            parser.error("--expected-audit-sha256 must be a lowercase SHA256")
    elif args.audit_logs or args.expected_audit_sha256:
        parser.error("Audit log options require --suite public-audit")
    raise SystemExit(asyncio.run(main_async(args, sources)))


if __name__ == "__main__":
    main()
