"""Compare named L1 candidates one parameter at a time and restore the baseline.

Run after ``run.sh up``. Each candidate restarts only this example's vLLM
service with a Compose override, must pass the readiness probes (exact
finite answers, a tool call, an image) and the prefix-cache consistency probe
(exact answers behind a shared cached prefix, cold and warm), and is then
measured with fixed 8K-in / 256-out rows. Raw evidence, the exact command,
and failures are kept; the committed configuration is restored in
``finally``. Final numbers come from ``verify.sh`` on the committed
configuration, not from these exploratory rows.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import control
import yaml

import verification

# Flag -> value (None removes the flag, True adds a bare flag). Every
# candidate is a change against the committed (selected) command and names
# each flag it changes, so a candidate never silently equals the baseline.
# The exploration that selected this command ran against the DP6 command
# under the earlier names (tp2-dp3-seqs-32-mtp-3-nccl and the rest); each
# run's selection.json and MEASUREMENTS.md keep their exact commands.
_MTP_5 = '{"method":"mtp","num_speculative_tokens":5}'
# The pinned KV pool fits the selected shape only; other shapes size their
# pool from --gpu-memory-utilization.
_UNPINNED = {"--kv-cache-memory": None}
_DP6 = {
    "--tensor-parallel-size": "1",
    "--data-parallel-size": "6",
    "--max-num-seqs": "16",
    "--disable-custom-all-reduce": None,
    **_UNPINNED,
}
CANDIDATES: dict[str, dict[str, object]] = {
    "baseline": {},
    # The committed command after a fresh restart: run-to-run variation of a
    # single trial, measured the same way as every other candidate.
    "baseline-repeat": {},
    # MTP: off, and the recipe's depth (5) instead of 3. Adopted only with
    # exact answers behind a cached prefix (vllm#53912 corrupted
    # prefix-cached MTP output on another hybrid linear-attention model).
    "no-mtp": {"--speculative-config": None},
    "mtp-5": {"--speculative-config": _MTP_5, **_UNPINNED},
    # vLLM's custom all-reduce instead of NCCL: fails while the MTP drafter's
    # CUDA graphs are captured (custom_all_reduce.cuh:164); without MTP it
    # measured no different from NCCL.
    "custom-all-reduce": {"--disable-custom-all-reduce": None},
    "custom-all-reduce-no-mtp": {"--disable-custom-all-reduce": None, "--speculative-config": None},
    # DP6 (TP1 per GPU, 6 x 16 = the same 96 sequences), without and with MTP;
    # with MTP no 1M-token KV pool fits.
    "dp6": {**_DP6, "--speculative-config": None},
    "dp6-mtp-3": _DP6,
    # Scheduler chunk (official tuning order: one parameter at a time).
    "batch-4k": {"--max-num-batched-tokens": "4096", **_UNPINNED},
    "batch-16k": {"--max-num-batched-tokens": "16384", **_UNPINNED},
}
CANDIDATE_ENVIRONMENT: dict[str, dict[str, str]] = {}
BARE_FLAGS = {"--enable-expert-parallel", "--enable-prefix-caching", "--disable-custom-all-reduce"}


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


def _compose_base() -> list[str]:
    return [
        "docker",
        "compose",
        "--project-directory",
        str(control.HERE),
        "--file",
        str(control.HERE / "compose.yaml"),
    ]


def restart(env: dict[str, str], override: Path | None, log: Path, timeout_s: int) -> None:
    command = _compose_base() + (["--file", str(override)] if override else [])
    command += ["up", "--detach", "--no-deps", "--force-recreate", "glm"]
    with log.open("w") as output:
        subprocess.run(command, env=env, stdout=output, stderr=subprocess.STDOUT, check=True)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        item = json.loads(subprocess.check_output(["docker", "inspect", control.L1_CONTAINER]))[0]
        if item["RestartCount"] or not item["State"]["Running"]:
            raise RuntimeError("L1 exited during startup; see worker.log")
        if item["State"].get("Health", {}).get("Status") == "healthy":
            return
        time.sleep(5)
    raise TimeoutError(f"L1 not healthy after {timeout_s} s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidates", nargs="+", choices=list(CANDIDATES))
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 8, 16])
    parser.add_argument("--startup-timeout", type=int, default=5400)
    parser.add_argument("--image-ttft", action="store_true", help="also time 8-image prompts")
    args = parser.parse_args()
    if args.requests < max(args.concurrency):
        parser.error("requests must cover every concurrency")
    env = control._compose_env()
    control._preflight(env)
    run_dir = verification.RESULTS_ROOT / f"{verification._run_id()}-tuning"
    run_dir.mkdir(parents=True)
    original = yaml.safe_load((control.HERE / "compose.yaml").read_text())["services"]["glm"][
        "command"
    ]
    api = control.api_check_url(env)
    reports: list[dict] = []
    try:
        for index, name in enumerate(args.candidates):
            row_dir = run_dir / name
            row_dir.mkdir()
            command = candidate_command(original, name)
            environment = CANDIDATE_ENVIRONMENT.get(name, {})
            service: dict = {"command": command}
            if environment:
                service["environment"] = environment
            override = row_dir / "override.json"
            override.write_text(json.dumps({"services": {"glm": service}}))
            report: dict = {
                "candidate": name,
                "command": command,
                "environment_override": environment,
                "served_config_sha256": verification.served_config_sha256(),
                "rows": [],
            }
            reports.append(report)
            try:
                if index == 0 and name == "baseline":
                    verification.runtime_evidence()
                    report["reused_running_baseline"] = True
                else:
                    restart(env, override, row_dir / "startup.log", args.startup_timeout)
                control.validate_ready(api)
                control._validate_arithmetic(api)
                control.validate_tool_calling(api)
                control.validate_vision(api)
                report["startup"] = verification._startup_evidence()
                report["kv_pool_error"] = verification.kv_pool_error(report["startup"])
                error, detail = verification.prefix_cache_consistency()
                report["prefix_cache_consistency"] = detail
                if error:
                    raise RuntimeError(f"prefix-cache consistency: {error}")
                if args.image_ttft:
                    report["image_ttft"] = verification.image_ttft(row_dir)
                for level in args.concurrency:
                    dataset = row_dir / f"c{level}.json"
                    verification.fixed_dataset(
                        dataset, args.requests, 8192, namespace=f"{run_dir.name}-{name}-{level}"
                    )
                    peak = verification.GpuPeak()
                    with peak:
                        code = verification.bench(
                            dataset,
                            mode="fixed",
                            requests=args.requests,
                            concurrency=level,
                            max_tokens=256,
                            out=row_dir / f"c{level}",
                        )
                    summary = json.loads((row_dir / f"c{level}" / "row.json").read_text())[
                        "summary"
                    ]
                    report["rows"].append(
                        {
                            "concurrency": level,
                            "exit_code": code,
                            "summary": summary,
                            "gpu_peak_mib": peak.peak,
                        }
                    )
                    if code:
                        raise RuntimeError(f"c{level} row failed")
                report["passed"] = True
            except (Exception, SystemExit) as error:  # noqa: BLE001 - recorded per candidate
                report.update(passed=False, error=str(error))
            finally:
                with (row_dir / "worker.log").open("w") as log:
                    subprocess.run(
                        ["docker", "logs", "--tail", "4000", control.L1_CONTAINER],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                (run_dir / "selection.json").write_text(json.dumps(reports, indent=2) + "\n")
    finally:
        if any(name != "baseline" for name in args.candidates):
            restart(env, None, run_dir / "restore.log", args.startup_timeout)
    print(f"candidate evidence: {run_dir}")
    raise SystemExit(int(any(not report["passed"] for report in reports)))


if __name__ == "__main__":
    main()
