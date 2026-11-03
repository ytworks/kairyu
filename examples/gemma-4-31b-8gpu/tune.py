"""Find this example's replica layout and L1 settings by measurement.

Run on an idle host after the checkpoints are attested (``run.sh up`` once).
Each candidate replaces the running stack with a rendered layout (compose and
kairyu.yaml in the run directory; the committed files are never edited),
passes the readiness probes, records every chat replica's KV capacity, and
is then measured with the serving rows of ``verify.sh serving`` (8K input, up
to 256 output, natural completion), optionally while three knowledge-base
ingestions keep Docling and the embedding replicas busy. ``finally``
restores the committed configuration with ``run.sh up``.

Steps of the replica search (README, MEASUREMENTS.md):

1. ``layout tp1-facts tp2-facts``: one chat replica each; KV capacity and the
   number of 262,144-token requests it holds. A TP that cannot hold one is out.
2. ``l1 --layout <name> baseline mtp-4 ...``: L1 options one at a time on the
   chosen TP; a change is kept only at 5 % or more.
3. ``embedding --layout <name>``: one embedding replica's batch throughput and
   query latency, and chat rows with and without embedding load when the
   embedding replicas share a chat GPU.
4. ``layout --load <names>``: the full layouts that survive, under ingestion.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import json
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import control
import yaml

import verification

# name -> chat TP, chat replica GPU groups, embedding replica GPUs, memory shares.
LAYOUTS: dict[str, dict] = {
    "tp1-facts": {"tp": 1, "chat": [[0]], "embed": [7, 7], "chat_util": 0.92},
    "tp2-facts": {"tp": 2, "chat": [[0, 1]], "embed": [7, 7], "chat_util": 0.92},
    # Full layouts: every GPU used; embedding replicas on a dedicated GPU or
    # sharing a chat replica's GPU (that replica's memory share shrinks).
    "tp1x7-embed-dedicated": {
        "tp": 1, "chat": [[i] for i in range(7)], "embed": [7, 7], "chat_util": 0.92,
    },
    "tp1x8-embed-shared": {
        "tp": 1, "chat": [[i] for i in range(8)], "embed": [6, 7], "chat_util": 0.86,
    },
    "tp2x4-embed-shared": {
        "tp": 2, "chat": [[0, 1], [2, 3], [4, 5], [6, 7]], "embed": [1, 7], "chat_util": 0.86,
    },
    "tp2x3-embed-dedicated": {
        "tp": 2, "chat": [[0, 1], [2, 3], [4, 5]], "embed": [6, 7], "chat_util": 0.92,
    },
}
EMBED_UTIL = 0.06

# Chat flag -> value (None removes the flag, True adds a bare flag).
DRAFTER = "/models/" + control.SPEC["drafter"]["slug"]
CANDIDATES: dict[str, dict[str, object]] = {
    "baseline": {},
    # Run-to-run variation of a single trial, measured like every candidate.
    "baseline-repeat": {},
    "mtp-4": {"--speculative-config": json.dumps({"model": DRAFTER, "num_speculative_tokens": 4})},
    "mtp-8": {"--speculative-config": json.dumps({"model": DRAFTER, "num_speculative_tokens": 8})},
    "no-async-scheduling": {"--async-scheduling": None},
    "batch-4k": {"--max-num-batched-tokens": "4096"},
    "batch-16k": {"--max-num-batched-tokens": "16384"},
    "seqs-128": {"--max-num-seqs": "128"},
}
BARE_FLAGS = {"--enable-prefix-caching", "--async-scheduling", "--enable-auto-tool-choice"}


def candidate_command(command: list[str], name: str) -> list[str]:
    result = list(command)
    for flag, value in CANDIDATES[name].items():
        if flag in result:
            index = result.index(flag)
            width = 1 if flag in BARE_FLAGS else 2
            replacement = [] if value is None else [flag] if value is True else [flag, str(value)]
            result[index : index + width] = replacement
        elif value is True:
            result.append(flag)
        elif value is not None:
            result.extend([flag, str(value)])
    return result


def _set(command: list[str], flag: str, value: str) -> list[str]:
    result = list(command)
    result[result.index(flag) + 1] = value
    return result


def render(layout_name: str, candidate: str, run_dir: Path, env: dict[str, str]) -> Path:
    """A complete compose file and kairyu.yaml for one layout and chat candidate."""
    layout = LAYOUTS[layout_name]
    rendered = json.loads(
        subprocess.check_output(
            control._compose_command(["config", "--format", "json"]), env=env, cwd=control.ROOT
        )
    )
    services = rendered["services"]
    chat_template = copy.deepcopy(services["gemma-0"])
    embed_template = copy.deepcopy(services["embed-0"])
    for name in [name for name in services if name.startswith(("gemma-", "embed-"))]:
        del services[name]
    gpu_shares: dict[int, float] = {}
    for index in layout["embed"]:
        gpu_shares[index] = gpu_shares.get(index, 0.0) + EMBED_UTIL
    chat_names, embed_names = [], []
    for replica, gpus in enumerate(layout["chat"]):
        service = copy.deepcopy(chat_template)
        command = candidate_command([str(part) for part in service["command"]], candidate)
        command = _set(command, "--tensor-parallel-size", str(layout["tp"]))
        command = _set(command, "--gpu-memory-utilization", str(layout["chat_util"]))
        service["command"] = command
        service["deploy"]["resources"]["reservations"]["devices"][0]["device_ids"] = [
            str(index) for index in gpus
        ]
        service["cpuset"] = env.get(f"GEMMA_{replica}_CPUSET", service.get("cpuset", "0"))
        if any(gpu_shares.get(index, 0.0) + layout["chat_util"] > 0.98 for index in gpus):
            raise SystemExit(f"{layout_name}: a shared GPU exceeds 0.98 of its memory")
        services[f"gemma-{replica}"] = service
        chat_names.append(f"gemma-{replica}")
    for replica, index in enumerate(layout["embed"]):
        service = copy.deepcopy(embed_template)
        service["command"] = _set(
            [str(part) for part in service["command"]], "--gpu-memory-utilization", str(EMBED_UTIL)
        )
        service["deploy"]["resources"]["reservations"]["devices"][0]["device_ids"] = [str(index)]
        services[f"embed-{replica}"] = service
        embed_names.append(f"embed-{replica}")
    services["kairyu"]["depends_on"] = {
        name: {"condition": "service_healthy", "required": True}
        for name in chat_names + embed_names
    }
    kairyu = yaml.safe_load((control.HERE / "kairyu.yaml").read_text())
    pool = kairyu["pools"][control.CHAT_MODEL]
    first = pool["replicas"][0]
    pool["replicas"] = [
        {
            **copy.deepcopy(first),
            "health_url": f"http://{name}:8000/health",
            "options": {**first["options"], "base_url": f"http://{name}:8000/v1"},
        }
        for name in chat_names
    ]
    kairyu["embeddings"][control.EMBED_MODEL]["replicas"] = [
        {"base_url": f"http://{name}:8000/v1", "health_url": f"http://{name}:8000/health"}
        for name in embed_names
    ]
    kairyu_path = run_dir / "kairyu.yaml"
    kairyu_path.write_text(yaml.safe_dump(kairyu, sort_keys=False))
    for volume in services["kairyu"]["volumes"]:
        if volume.get("target") == "/etc/kairyu/kairyu.yaml":
            volume["source"] = str(kairyu_path)
    compose_path = run_dir / "compose.json"
    compose_path.write_text(json.dumps(rendered, indent=2))
    (run_dir / "layout.json").write_text(
        json.dumps({"layout": layout_name, **layout, "candidate": candidate}, indent=2)
    )
    return compose_path


def start(compose_path: Path, env: dict[str, str], log: Path) -> None:
    with log.open("w") as output:
        subprocess.run(
            control._compose_command(["down", "--remove-orphans"]),
            env=env, stdout=output, stderr=subprocess.STDOUT, check=False,
        )
        subprocess.run(
            [
                "docker", "compose", "--project-directory", str(control.HERE),
                "--file", str(compose_path),
                "up", "--detach", "--wait", "--wait-timeout", "7200", "--remove-orphans",
            ],
            env=env, stdout=output, stderr=subprocess.STDOUT, check=True,
        )


def ready(api: str, chat: int, embed: int) -> None:
    metrics = urllib.request.urlopen(f"{api}/metrics", timeout=5).read().decode()
    counts = (
        control.healthy_replicas(metrics, control.CHAT_MODEL),
        control.healthy_replicas(metrics, control.EMBED_MODEL),
    )
    if counts != (chat, embed):
        raise RuntimeError(f"healthy replicas {counts}, expected {(chat, embed)}")
    control._validate_arithmetic(api)
    control.validate_tool_calling(api)
    control.validate_vision(api)
    control.validate_embeddings(api)


def startup(chat: int) -> dict:
    return {
        f"gemma-{replica}": verification.startup_evidence(f"{control.PROJECT}-gemma-{replica}-1")
        for replica in range(chat)
    }


def rows(run_dir: Path, name: str, levels: list[int], requests: int) -> list[dict]:
    run_dir.mkdir(parents=True, exist_ok=True)
    measured = []
    for level in levels:
        dataset = run_dir / f"c{level}.json"
        verification.fixed_dataset(
            dataset, requests, 8192, namespace=f"{run_dir.name}-{name}-{level}"
        )
        peak = verification.GpuPeak()
        with peak:
            code = verification.bench(
                dataset, mode="capped", requests=requests, concurrency=level, max_tokens=256,
                out=run_dir / f"c{level}",
            )
        summary = json.loads((run_dir / f"c{level}" / "row.json").read_text())["summary"]
        measured.append({"concurrency": level, "exit_code": code, "summary": summary,
                         "gpu_peak_mib": peak.peak})
        if code:
            raise RuntimeError(f"c{level} row failed")
    return measured


class IngestionLoad:
    """Knowledge-base ingestions through the Chat UI for as long as the rows run.

    A renderer keeps fresh 20-page scans ready (Open WebUI refuses content it
    already holds) and ``streams`` workers ingest them without end, so Docling
    and the embedding replicas stay loaded through every row. Meanwhile a probe
    asks the knowledge gate's question of a reference knowledge base every
    ``query_interval_s``: retrieval embeds the query through Kairyu, searches
    and reranks. ``result()`` passes only when ingestion covered the measured
    window, nothing failed, and every ingestion and query stayed within the
    knowledge budgets of ``example.json``.
    """

    def __init__(self, run_dir: Path, streams: int = 3, query_interval_s: float = 20.0) -> None:
        knowledge = control.SPEC["verification"]["knowledge"]
        self.run_dir = run_dir
        self.streams = streams
        self.query_interval_s = query_interval_s
        self.ingestion_budget_s = float(knowledge["ingestion_seconds"])
        self.query_budget_s = float(knowledge["query_seconds"])
        self.minimum_coverage = float(knowledge["docling_busy_share"])
        self.errors: list[str] = []
        self.ingestions: list[tuple[float, float]] = []
        self.queries: list[float] = []
        self.window: list[float] = []
        self._active: dict[int, float] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    def _knowledge_base(self, name: str) -> str:
        return verification._webui_request(
            "POST", "/api/v1/knowledge/create", self.token,
            payload={"name": f"{name}-{self.run_dir.name}", "description": "tune load"},
        )["id"]

    def _render(self, pending) -> None:
        import queue

        index = 0
        while not self._stop.is_set():
            scan = self.run_dir / "load" / f"load-{index:04d}.pdf"
            verification.scan_pdf(scan, 20, {13}, namespace=f"{self.run_dir.name}-{index}")
            index += 1
            while not self._stop.is_set():
                try:
                    pending.put(scan, timeout=1)
                    break
                except queue.Full:
                    continue

    def _ingest(self, pending) -> None:
        import queue

        worker = threading.get_ident()
        while not self._stop.is_set():
            try:
                scan = pending.get(timeout=1)
            except queue.Empty:
                continue
            started = time.monotonic()
            with self._lock:
                self._active[worker] = started
            try:
                verification._ingest(
                    self.token, self.load_id, scan.name, scan.read_bytes(), "application/pdf"
                )
            except Exception as error:  # noqa: BLE001 - recorded, fails the layout
                self.errors.append(f"ingestion: {error!r}")
                self._stop.set()
                return
            finally:
                with self._lock:
                    self._active.pop(worker, None)
            self.ingestions.append((started, time.monotonic()))

    def _query(self) -> None:
        while not self._stop.wait(self.query_interval_s):
            started = time.monotonic()
            try:
                verification._webui_request(
                    "POST", "/api/v1/retrieval/query/collection", self.token,
                    payload={
                        "collection_names": [self.reference_id],
                        "query": verification.KNOWLEDGE_QUERY,
                    },
                )
            except Exception as error:  # noqa: BLE001 - recorded, fails the layout
                self.errors.append(f"query: {error!r}")
                continue
            self.queries.append(time.monotonic() - started)

    def __enter__(self) -> IngestionLoad:
        import queue

        self.token = verification._chat_ui_token()
        self.reference_id = self._knowledge_base("kairyu-tune-reference")
        reference = self.run_dir / verification.KNOWLEDGE_SCAN
        verification.scan_pdf(reference, 20, {13}, namespace=f"{self.run_dir.name}-reference")
        verification._ingest(
            self.token, self.reference_id, reference.name, reference.read_bytes(),
            "application/pdf",
        )
        self.load_id = self._knowledge_base("kairyu-tune-load")
        pending: queue.Queue[Path] = queue.Queue(maxsize=2 * self.streams)
        self._threads = [threading.Thread(target=self._render, args=(pending,), daemon=True)]
        self._threads += [
            threading.Thread(target=self._ingest, args=(pending,), daemon=True)
            for _ in range(self.streams)
        ]
        self._threads.append(threading.Thread(target=self._query, daemon=True))
        for thread in self._threads:
            thread.start()
        time.sleep(30)  # let Docling and the embedding replicas reach steady load
        self.window.append(time.monotonic())
        return self

    def __exit__(self, *_: object) -> None:
        self.window.append(time.monotonic())
        with self._lock:
            in_flight = list(self._active.values())
        self._open = [(started, self.window[1]) for started in in_flight]
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=2 * self.ingestion_budget_s)
        self.stuck = sum(thread.is_alive() for thread in self._threads)
        for knowledge_id in (self.load_id, self.reference_id):
            verification._delete_knowledge(self.token, knowledge_id)

    def result(self) -> dict:
        """Coverage of the measured window by ingestions, failures and the budgets."""
        start, end = self.window
        intervals = sorted(
            (max(first, start), min(last, end))
            for first, last in [*self.ingestions, *self._open]
            if last > start and first < end
        )
        covered, reach = 0.0, start
        for first, last in intervals:
            if last > reach:
                covered += last - max(first, reach)
                reach = last
        coverage = covered / (end - start) if end > start else 0.0
        longest_ingestion = max(
            [last - first for first, last in self.ingestions]
            + [end - first for first, _ in self._open],
            default=0.0,
        )
        slowest_query = max(self.queries, default=None)
        passed = (
            not self.errors
            and not self.stuck
            and coverage >= self.minimum_coverage
            and longest_ingestion <= self.ingestion_budget_s
            and slowest_query is not None
            and slowest_query <= self.query_budget_s
        )
        return {
            "window_s": round(end - start, 1),
            "coverage": round(coverage, 3),
            "ingestions_completed": len(self.ingestions),
            "longest_ingestion_s": round(longest_ingestion, 1),
            "queries": len(self.queries),
            "slowest_query_s": round(slowest_query, 2) if slowest_query is not None else None,
            "errors": self.errors,
            "threads_stuck": self.stuck,
            "passed": passed,
        }


def embedding_bench(api: str, batch: int = 32, concurrency: int = 4, rounds: int = 32) -> dict:
    """Open WebUI's ingestion shape: batches of 300-character chunks, a few in flight."""
    chunk = "試料の保管条件と点検の記録を確認した。" * 16
    texts = [control.SPEC["embedding"]["document_prefix"] + f"{chunk[:290]} {index}"
             for index in range(batch)]
    latencies: list[float] = []

    def one(_: int) -> None:
        started = time.monotonic()
        status, _body = verification.post_embeddings(texts)
        if status != 200:
            raise RuntimeError(f"embedding batch returned HTTP {status}")
        latencies.append(time.monotonic() - started)

    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        list(pool.map(one, range(rounds)))
    elapsed = time.monotonic() - started
    query = control.SPEC["embedding"]["query_prefix"] + "KR-7319の保存温度は？"
    single = []
    for _ in range(16):
        begun = time.monotonic()
        verification.post_embeddings([query])
        single.append(time.monotonic() - begun)
    return {
        "texts_per_s": batch * rounds / elapsed,
        "batch_p50_s": statistics.median(latencies),
        "batch_max_s": max(latencies),
        "query_p50_s": statistics.median(single),
        "query_max_s": max(single),
    }


class EmbeddingLoad:
    """Continuous embedding batches (Open WebUI's shape) while chat rows run."""

    def __init__(self, api: str) -> None:
        self.api = api
        self._stop = threading.Event()
        self.batches = 0
        self.errors = 0

    def __enter__(self) -> EmbeddingLoad:
        def loop() -> None:
            texts = [control.SPEC["embedding"]["document_prefix"] + "点検記録の確認。" * 30] * 32
            while not self._stop.is_set():
                status, _ = verification.post_embeddings(texts)
                self.batches += status == 200
                self.errors += status != 200

        self._threads = [threading.Thread(target=loop, daemon=True) for _ in range(4)]
        for thread in self._threads:
            thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=60)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    layout = sub.add_parser("layout", help="measure whole layouts")
    layout.add_argument("layouts", nargs="+", choices=list(LAYOUTS))
    layout.add_argument("--load", action="store_true", help="three ingestions during the rows")
    layout.add_argument("--candidate", default="baseline", choices=list(CANDIDATES))
    l1 = sub.add_parser("l1", help="chat L1 options one at a time on one layout")
    l1.add_argument("candidates", nargs="+", choices=list(CANDIDATES))
    l1.add_argument("--layout", required=True, choices=list(LAYOUTS))
    embedding = sub.add_parser("embedding", help="embedding throughput and interference")
    embedding.add_argument("--layout", required=True, choices=list(LAYOUTS))
    for command in (layout, l1, embedding):
        command.add_argument("--requests", type=int, default=64)
        command.add_argument("--concurrency", type=int, nargs="+", default=[1, 8, 16, 32, 64])
    args = parser.parse_args()
    env = control._compose_env()
    control._preflight(env)
    run_root = verification.RESULTS_ROOT / f"{verification._run_id()}-tune-{args.action}"
    run_root.mkdir(parents=True)
    api = f"http://127.0.0.1:{env['API_PORT']}"
    ui = control.chat_ui_local_url(env)
    if args.action == "layout":
        jobs = [(name, args.candidate) for name in args.layouts]
    elif args.action == "l1":
        jobs = [(args.layout, candidate) for candidate in args.candidates]
    else:
        jobs = [(args.layout, "baseline")]
    reports: list[dict] = []
    try:
        for layout_name, candidate in jobs:
            row_dir = run_root / f"{layout_name}--{candidate}"
            row_dir.mkdir()
            report: dict = {"layout": layout_name, "candidate": candidate}
            reports.append(report)
            spec = LAYOUTS[layout_name]
            try:
                compose_path = render(layout_name, candidate, row_dir, env)
                start(compose_path, env, row_dir / "startup.log")
                ready(api, len(spec["chat"]), len(spec["embed"]))
                control.provision_chat_ui(ui)
                report["startup"] = startup(len(spec["chat"]))
                report["max_concurrency_262144"] = {
                    replica: [value for tokens, value in evidence["max_concurrency"]
                              if int(tokens.replace(",", "")) == 262144]
                    for replica, evidence in report["startup"].items()
                }
                if args.action == "embedding":
                    report["embedding_alone"] = embedding_bench(api)
                    report["chat_rows_idle"] = rows(row_dir / "idle", "idle", [1, 8], args.requests)
                    with EmbeddingLoad(api) as load:
                        report["chat_rows_with_embedding_load"] = rows(
                            row_dir / "loaded", "loaded", [1, 8], args.requests
                        )
                    report["embedding_load"] = {"batches": load.batches, "errors": load.errors}
                elif args.action == "layout" and args.load:
                    # Rows, then the embedding batches, all while the ingestions
                    # run; the layout passes only within the knowledge budgets.
                    with IngestionLoad(row_dir) as load:
                        report["rows"] = rows(row_dir, layout_name, args.concurrency, args.requests)
                        report["embedding_under_load"] = embedding_bench(api, rounds=8)
                    report["ingestion"] = load.result()
                    if not report["ingestion"]["passed"]:
                        raise RuntimeError(f"knowledge load failed: {report['ingestion']}")
                else:
                    report["rows"] = rows(row_dir, candidate, args.concurrency, args.requests)
                report["passed"] = True
            except (Exception, SystemExit) as error:  # noqa: BLE001 - recorded per candidate
                report.update(passed=False, error=str(error))
            finally:
                for service in [f"gemma-{r}" for r in range(len(spec["chat"]))] + [
                    f"embed-{r}" for r in range(len(spec["embed"]))
                ]:
                    with (row_dir / f"{service}.log").open("w") as log:
                        subprocess.run(
                            ["docker", "logs", "--tail", "4000", f"{control.PROJECT}-{service}-1"],
                            stdout=log, stderr=subprocess.STDOUT, check=False,
                        )
                (run_root / "selection.json").write_text(json.dumps(reports, indent=2) + "\n")
    finally:
        # The committed configuration, exactly as run.sh starts it.
        subprocess.run(control._compose_command(["down", "--remove-orphans"]), env=env, check=False)
        control.up()
    print(f"candidate evidence: {run_root}")
    raise SystemExit(int(any(not report["passed"] for report in reports)))


if __name__ == "__main__":
    main()
