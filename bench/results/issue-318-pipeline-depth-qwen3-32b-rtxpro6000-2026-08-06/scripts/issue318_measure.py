import argparse
import asyncio
import hashlib
import json
import math
import subprocess
import time
import urllib.request
from pathlib import Path

from bench import g2_a6_vllm_bench as a6
from bench import issue_333_proc_http_bench as diagnostic


def fetch_json(url: str):
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.load(response)


def fetch_text(url: str) -> str:
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.read().decode()


def command_json(command: list[str]):
    return json.loads(subprocess.check_output(command, text=True))


def gpu_snapshot() -> list[dict[str, str]]:
    fields = (
        "index,uuid,name,temperature.gpu,power.draw,clocks.sm,clocks.mem,"
        "memory.used"
    )
    output = subprocess.check_output(
        [
            "nvidia-smi",
            f"--query-gpu={fields}",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    keys = fields.split(",")
    return [
        dict(zip(keys, (value.strip() for value in line.split(",")), strict=True))
        for line in output.splitlines()
        if line.strip()
    ]


def quantile(values, fraction: float) -> int:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--trace-bundle", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--container-name", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    trace = json.loads(args.trace_bundle.read_text())
    traces = trace.get("traces", {})
    if not (
        isinstance(traces, dict)
        and a6._trace_valid(
            traces.get("sharegpt"),
            workload="sharegpt",
            dataset_sha256=a6.SHAREGPT_DATASET_SHA256,
        )
        and a6._graph_warmup_trace_valid(trace.get("graph_warmup"))
        and a6._graph_warmup_disjoint(trace.get("graph_warmup"), traces)
    ):
        raise ValueError("retained ShareGPT request trace is invalid")
    config_bytes = args.config.read_bytes()
    trace_bytes = args.trace_bundle.read_bytes()
    container = command_json(["docker", "inspect", args.container_name])[0]
    gpu_before = gpu_snapshot()
    started = time.time_ns()
    rows, evidence = asyncio.run(
        diagnostic.measure_cell_requests(
            endpoint=args.endpoint,
            trace_bundle=trace,
            request_timeout_s=600.0,
            api_key=None,
            probe_backend=lambda: fetch_json(f"{args.endpoint}/backends"),
            probe_process_tree=lambda: {"diagnostic": "not-collected"},
        )
    )
    ended = time.time_ns()
    measured = [row for row in rows if row.get("phase") == "measurement"]
    successful = [row for row in measured if row.get("status") == "success"]
    timings = [row["timing_ns"] for row in successful]
    output_tokens = sum(row["usage"]["completion_tokens"] for row in successful)
    window_ns = max(item["end"] for item in timings) - min(
        item["start"] for item in timings
    )
    tpot_ns = [
        (row["timing_ns"]["end"] - row["timing_ns"]["first_token"])
        // max(1, row["usage"]["completion_tokens"] - 1)
        for row in successful
    ]
    source_diff = subprocess.check_output(
        ["git", "diff", "--binary", "HEAD"], cwd=args.source
    )
    payload = {
        "schema_version": 2,
        "label": args.label,
        "source": {
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=args.source, text=True
            ).strip(),
            "diff_sha256": hashlib.sha256(source_diff).hexdigest(),
            "dirty": bool(source_diff),
        },
        "inputs": {
            "config_path": str(args.config.resolve()),
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "config_text": config_bytes.decode(),
            "trace_path": str(args.trace_bundle.resolve()),
            "trace_sha256": hashlib.sha256(trace_bytes).hexdigest(),
            "measurement_script_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
        },
        "container": {
            "id": container["Id"],
            "image": container["Config"]["Image"],
            "image_id": container["Image"],
            "created": container["Created"],
            "path": container["Path"],
            "args": container["Args"],
            "mounts": container["Mounts"],
        },
        "gpu_before": gpu_before,
        "gpu_after": gpu_snapshot(),
        "wall_started_ns": started,
        "wall_ended_ns": ended,
        "measurement": {
            "requests": len(measured),
            "successes": len(successful),
            "output_tokens": output_tokens,
            "window_ns": window_ns,
            "output_tokens_per_s": output_tokens * 1_000_000_000 / window_ns,
            "ttft_p50_ns": quantile([item["ttft"] for item in timings], 0.50),
            "ttft_p99_ns": quantile([item["ttft"] for item in timings], 0.99),
            "tpot_p50_ns": quantile(tpot_ns, 0.50),
            "tpot_p99_ns": quantile(tpot_ns, 0.99),
            "output_sha256_by_sequence": {
                str(row["sequence"]): row["output_text_sha256"] for row in successful
            },
        },
        "backend_evidence": evidence,
        "pipeline_metrics": [
            line
            for line in fetch_text(f"{args.endpoint}/metrics").splitlines()
            if "pipeline_depth" in line
        ],
        "requests": rows,
    }
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    summary = {
        key: value
        for key, value in payload["measurement"].items()
        if key != "output_sha256_by_sequence"
    }
    print(json.dumps({"label": args.label, **summary}, indent=2))


if __name__ == "__main__":
    main()
