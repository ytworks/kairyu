# Qwen3-32B Benchmark Shell Progress Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show periodic startup and request-completion progress when the Qwen3-32B example launches and runs its serving benchmark.

**Architecture:** Keep orchestration in the two existing POSIX shell entrypoints. `run-benchmark.sh` reports readiness polling; `benchmark.sh` runs an auxiliary background monitor that reads the local Prometheus counter and falls back to elapsed time while the existing Docker benchmark remains the source of truth for exit status.

**Tech Stack:** POSIX `sh`, Docker Compose, Docker CLI, `curl`, `awk`, pytest subprocess tests.

## Global Constraints

- Do not modify `kairyu/`, `bench/serving_bench.py`, the Compose topology, or the model configuration.
- Production changes are limited to shell scripts and README content in `examples/qwen3-32b-multi-gpu/`.
- Progress refreshes every 5 seconds by default.
- Metrics failure must never fail or change the result of the benchmark.
- The benchmark container's exit status remains authoritative.
- Scripts remain compatible with `/bin/sh`.

---

### Task 1: Readiness progress

**Files:**
- Create: `tests/examples/test_qwen3_32b_multi_gpu_shell.py`
- Modify: `examples/qwen3-32b-multi-gpu/run-benchmark.sh`

**Interfaces:**
- Consumes: `run.sh --detach`, `GET http://127.0.0.1:8001/readyz`, and `benchmark.sh`.
- Produces: `[startup]` stage lines and periodic `waiting for readiness (elapsed Ns)` output.

- [ ] **Step 1: Write the failing readiness-progress test**

Create reusable helpers that copy the script under test into a temporary
example directory and install executable fake commands:

```python
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "qwen3-32b-multi-gpu"


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def _run(script: Path, fake_bin: Path, state_dir: Path, **extra: str):
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:/usr/bin:/bin",
        "STATE_DIR": str(state_dir),
        "PROGRESS_INTERVAL_S": "5",
        **extra,
    }
    return subprocess.run(
        ["/bin/sh", str(script)],
        cwd=script.parent,
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )


def _readiness_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    example = tmp_path / "example"
    fake_bin = tmp_path / "bin"
    state = tmp_path / "state"
    example.mkdir()
    fake_bin.mkdir()
    state.mkdir()
    shutil.copy(EXAMPLE / "run-benchmark.sh", example / "run-benchmark.sh")
    _write_executable(example / "run.sh", "#!/bin/sh\necho run-called\n")
    _write_executable(example / "benchmark.sh", "#!/bin/sh\necho benchmark-called\n")
    _write_executable(fake_bin / "sleep", "#!/bin/sh\nexit 0\n")
    return example, fake_bin, state


def test_run_benchmark_reports_readiness_wait_progress(tmp_path: Path) -> None:
    example, fake_bin, state = _readiness_fixture(tmp_path)
    _write_executable(
        fake_bin / "curl",
        """#!/bin/sh
count_file="$STATE_DIR/curl-count"
count=0
[ ! -f "$count_file" ] || count="$(cat "$count_file")"
count=$((count + 1))
printf '%s\n' "$count" >"$count_file"
[ "$count" -ge 3 ]
""",
    )

    result = _run(example / "run-benchmark.sh", fake_bin, state)

    assert result.returncode == 0, result.stderr
    assert "[startup] starting Qwen3-32B service" in result.stdout
    assert "[startup] waiting for readiness" in result.stdout
    assert "[startup] waiting for readiness (elapsed 5s)" in result.stdout
    assert "[startup] waiting for readiness (elapsed 10s)" in result.stdout
    assert "[startup] ready after 10s" in result.stdout
    assert "[benchmark] starting" in result.stdout
    assert "benchmark-called" in result.stdout
```

- [ ] **Step 2: Run the readiness-progress test and verify RED**

Run:

```bash
uv run pytest tests/examples/test_qwen3_32b_multi_gpu_shell.py::test_run_benchmark_reports_readiness_wait_progress -q
```

Expected: FAIL because the current script prints none of the `[startup]` or
`[benchmark]` progress lines.

- [ ] **Step 3: Add the timeout regression test**

Add:

```python
def test_run_benchmark_timeout_keeps_compose_diagnostics(tmp_path: Path) -> None:
    example, fake_bin, state = _readiness_fixture(tmp_path)
    _write_executable(fake_bin / "curl", "#!/bin/sh\nexit 1\n")
    _write_executable(
        fake_bin / "docker",
        """#!/bin/sh
[ "${1:-}" = compose ] && [ "${2:-}" = logs ] && echo compose-diagnostics >&2
""",
    )

    result = _run(example / "run-benchmark.sh", fake_bin, state)

    assert result.returncode == 1
    assert "[startup] waiting for readiness (elapsed 900s)" in result.stdout
    assert "Kairyu did not become ready" in result.stderr
    assert "compose-diagnostics" in result.stderr
    assert "benchmark-called" not in result.stdout
```

- [ ] **Step 4: Implement readiness stage and elapsed output**

In `run-benchmark.sh`, add a configurable test seam whose production default is
five seconds, track elapsed time, and retain the existing 180-attempt timeout:

```sh
progress_interval_s="${PROGRESS_INTERVAL_S:-5}"

printf '[startup] starting Qwen3-32B service\n'
./run.sh --detach

printf '[startup] waiting for readiness at http://127.0.0.1:8001/readyz\n'
attempt=0
elapsed_s=0
until curl --fail --silent http://127.0.0.1:8001/readyz >/dev/null 2>&1; do
  attempt=$((attempt + 1))
  elapsed_s=$((elapsed_s + progress_interval_s))
  printf '[startup] waiting for readiness (elapsed %ss)\n' "$elapsed_s"
  if [ "$attempt" -ge 180 ]; then
    echo "Kairyu did not become ready on http://127.0.0.1:8001" >&2
    docker compose logs kairyu >&2
    exit 1
  fi
  sleep "$progress_interval_s"
done

printf '[startup] ready after %ss\n' "$elapsed_s"
printf '[benchmark] starting\n'
exec ./benchmark.sh
```

- [ ] **Step 5: Run Task 1 tests and verify GREEN**

Run:

```bash
uv run pytest tests/examples/test_qwen3_32b_multi_gpu_shell.py -q
/bin/sh -n examples/qwen3-32b-multi-gpu/run-benchmark.sh
```

Expected: `2 passed`; shell syntax exits 0.

- [ ] **Step 6: Commit Task 1**

```bash
git add tests/examples/test_qwen3_32b_multi_gpu_shell.py examples/qwen3-32b-multi-gpu/run-benchmark.sh
git commit -m "feat: show qwen startup progress"
```

### Task 2: Benchmark completion progress

**Files:**
- Modify: `tests/examples/test_qwen3_32b_multi_gpu_shell.py`
- Modify: `examples/qwen3-32b-multi-gpu/benchmark.sh`

**Interfaces:**
- Consumes: `GET /metrics` lines named `kairyu_requests_total` with `model="qwen3-32b"`.
- Produces: `metric_request_count MODEL -> integer`, a background `monitor_progress` process, and `[benchmark] completed N/total` or elapsed-only output.

- [ ] **Step 1: Add fake benchmark commands and the failing metric-progress test**

Add these helpers:

```python
def _benchmark_fixture(
    tmp_path: Path, *, metrics: bool = True
) -> tuple[Path, Path, Path]:
    example = tmp_path / "example"
    fake_bin = tmp_path / "bin"
    state = tmp_path / "state"
    example.mkdir()
    fake_bin.mkdir()
    state.mkdir()
    (example / "results").mkdir()
    shutil.copy(EXAMPLE / "benchmark.sh", example / "benchmark.sh")
    (example / "benchmark_report.py").write_text("", encoding="utf-8")
    metrics_body = (
        """count_file="$STATE_DIR/metrics-count"
count=0
[ ! -f "$count_file" ] || count="$(cat "$count_file")"
count=$((count + 1))
printf '%s\n' "$count" >"$count_file"
if [ "$count" -eq 1 ]; then value=100; else value=150; fi
printf 'kairyu_requests_total{code="200",model="qwen3-32b"} %s.0\n' "$value"
exit 0
"""
        if metrics
        else "exit 1\n"
    )
    _write_executable(
        fake_bin / "curl",
        f"""#!/bin/sh
case "$*" in
  *readyz*) exit 0 ;;
  *metrics*) {metrics_body}  ;;
esac
exit 1
""",
    )
    _write_executable(
        fake_bin / "sleep",
        """#!/bin/sh
count_file="$STATE_DIR/sleep-count"
count=0
[ ! -f "$count_file" ] || count="$(cat "$count_file")"
count=$((count + 1))
printf '%s\n' "$count" >"$count_file"
[ "$count" -lt 2 ] || touch "$STATE_DIR/release"
/bin/sleep 0.02
""",
    )
    _write_executable(
        fake_bin / "docker",
        """#!/bin/sh
if [ "${1:-}" = compose ] && [ "${2:-}" = exec ]; then
  echo 4
  exit 0
fi
if [ "${1:-}" = compose ] && [ "${2:-}" = images ]; then
  echo sha256:test
  exit 0
fi
case "$*" in
  *benchmark_report.py*)
    touch "$STATE_DIR/report-called"
    echo report-output
    exit 0
    ;;
esac
while [ ! -f "$STATE_DIR/release" ]; do /bin/sleep 0.01; done
echo benchmark-output
exit "${BENCHMARK_EXIT:-0}"
""",
    )
    return example, fake_bin, state


def test_benchmark_reports_metric_progress_and_configuration(tmp_path: Path) -> None:
    example, fake_bin, state = _benchmark_fixture(tmp_path)

    result = _run(
        example / "benchmark.sh",
        fake_bin,
        state,
        NUM_REQUESTS="3",
        CONCURRENCY="2",
        MAX_TOKENS="16",
    )

    assert result.returncode == 0, result.stderr
    assert (
        "[benchmark] requests=3 concurrency=2 max_tokens=16 GPUs/TP=4"
        in result.stdout
    )
    assert "[benchmark] completed 3/3 (elapsed 5s)" in result.stdout
    assert "benchmark-output" in result.stdout
    assert "[report] generating Markdown report" in result.stdout
    assert "report-output" in result.stdout
```

- [ ] **Step 2: Run metric-progress test and verify RED**

Run:

```bash
uv run pytest tests/examples/test_qwen3_32b_multi_gpu_shell.py::test_benchmark_reports_metric_progress_and_configuration -q
```

Expected: FAIL because `benchmark.sh` has no configuration banner, progress
monitor, or report-stage banner.

- [ ] **Step 3: Add failing fallback and exit-status tests**

Add:

```python
def test_benchmark_falls_back_when_metrics_are_unavailable(tmp_path: Path) -> None:
    example, fake_bin, state = _benchmark_fixture(tmp_path, metrics=False)

    result = _run(example / "benchmark.sh", fake_bin, state, NUM_REQUESTS="3")

    assert result.returncode == 0, result.stderr
    assert "[benchmark] running (elapsed 5s)" in result.stdout


def test_benchmark_failure_stays_failure_and_skips_report(tmp_path: Path) -> None:
    example, fake_bin, state = _benchmark_fixture(tmp_path)

    result = _run(
        example / "benchmark.sh",
        fake_bin,
        state,
        NUM_REQUESTS="3",
        BENCHMARK_EXIT="7",
    )

    assert result.returncode == 7
    assert not (state / "report-called").exists()
    assert "[report] generating Markdown report" not in result.stdout
```

- [ ] **Step 4: Implement metric parsing and the non-authoritative monitor**

In `benchmark.sh`, add:

```sh
progress_interval_s="${PROGRESS_INTERVAL_S:-5}"
metrics_url="http://127.0.0.1:8001/metrics"

metric_request_count() {
  model="$1"
  curl --fail --silent "$metrics_url" 2>/dev/null |
    awk -v model="$model" '
      $1 ~ /^kairyu_requests_total\{/ &&
      index($1, "model=\"" model "\"") {
        total += $2
        found = 1
      }
      END {
        if (!found) exit 1
        printf "%.0f\n", total
      }
    '
}

monitor_progress() {
  baseline="$1"
  elapsed_s=0
  while :; do
    sleep "$progress_interval_s"
    elapsed_s=$((elapsed_s + progress_interval_s))
    current=""
    if [ -n "$baseline" ]; then
      current="$(metric_request_count qwen3-32b)" || current=""
    fi
    if [ -n "$current" ]; then
      completed=$((current - baseline))
      [ "$completed" -ge 0 ] || completed=0
      [ "$completed" -le "$num_requests" ] || completed="$num_requests"
      printf '[benchmark] completed %s/%s (elapsed %ss)\n' \
        "$completed" "$num_requests" "$elapsed_s"
    else
      printf '[benchmark] running (elapsed %ss)\n' "$elapsed_s"
    fi
  done
}

stop_progress_monitor() {
  if [ -n "${progress_pid:-}" ]; then
    kill "$progress_pid" 2>/dev/null || true
    wait "$progress_pid" 2>/dev/null || true
    progress_pid=""
  fi
}
```

Before the benchmark `docker run`, print the resolved configuration, capture a
best-effort baseline, and launch the monitor:

```sh
printf '[benchmark] requests=%s concurrency=%s max_tokens=%s GPUs/TP=%s\n' \
  "$num_requests" "$concurrency" "$max_tokens" "$gpu_count"

baseline="$(metric_request_count qwen3-32b)" || baseline=""
progress_pid=""
trap stop_progress_monitor EXIT HUP INT TERM
monitor_progress "$baseline" &
progress_pid=$!

benchmark_status=0
docker run --rm --network host \
  --entrypoint python \
  --volume "$repo_root/bench:/bench:ro" \
  --volume "$(pwd)/results:/results" \
  "$image_id" \
  /bench/serving_bench.py \
  --base-url http://127.0.0.1:8001 \
  --model qwen3-32b \
  --num-requests "$num_requests" \
  --concurrency "$concurrency" \
  --max-tokens "$max_tokens" \
  --ttft-slo-s "$ttft_slo_s" \
  --timeout "$timeout_s" \
  --tensor-parallel "$gpu_count" \
  --results-dir /results || benchmark_status=$?

stop_progress_monitor
trap - EXIT HUP INT TERM
if [ "$benchmark_status" -ne 0 ]; then
  exit "$benchmark_status"
fi

printf '[report] generating Markdown report\n'
docker run --rm \
  --entrypoint python \
  --volume "$(pwd)/benchmark_report.py:/opt/kairyu/benchmark_report.py:ro" \
  --volume "$(pwd)/results:/results" \
  "$image_id" \
  /opt/kairyu/benchmark_report.py \
  /results \
  --output /results/report.md
```

These are the complete existing Docker argument lists; only their surrounding
status handling and stage lines change.

- [ ] **Step 5: Run Task 2 tests and verify GREEN**

Run:

```bash
uv run pytest tests/examples/test_qwen3_32b_multi_gpu_shell.py -q
/bin/sh -n examples/qwen3-32b-multi-gpu/benchmark.sh
```

Expected: `5 passed`; shell syntax exits 0.

- [ ] **Step 6: Commit Task 2**

```bash
git add tests/examples/test_qwen3_32b_multi_gpu_shell.py examples/qwen3-32b-multi-gpu/benchmark.sh
git commit -m "feat: show qwen benchmark progress"
```

### Task 3: Example documentation and final verification

**Files:**
- Modify: `examples/qwen3-32b-multi-gpu/README.md`

**Interfaces:**
- Consumes: the final `[startup]`, `[benchmark]`, and `[report]` output contract.
- Produces: user-facing documentation of automatic progress and its fallback.

- [ ] **Step 1: Document the progress output**

After the one-command example in README, add:

```markdown
While the command runs, the shell reports model readiness and benchmark
progress every five seconds. Completed request counts come from Kairyu's local
metrics endpoint; if that endpoint cannot be read, the shell continues to show
elapsed time. Progress reporting does not change the benchmark result.
```

- [ ] **Step 2: Run focused verification**

Run:

```bash
uv run pytest tests/examples/test_qwen3_32b_multi_gpu_shell.py -q
/bin/sh -n examples/qwen3-32b-multi-gpu/run.sh
/bin/sh -n examples/qwen3-32b-multi-gpu/run-benchmark.sh
/bin/sh -n examples/qwen3-32b-multi-gpu/benchmark.sh
git diff --check
```

Expected: `5 passed`; all syntax checks and `git diff --check` exit 0.

- [ ] **Step 3: Confirm the framework boundary**

Run:

```bash
git diff --name-only 690eee6..HEAD
```

Expected: only the specification/plan, Qwen example shell/README files, and the
dedicated shell test are listed; no path under `kairyu/` or shared file under
`bench/` appears.

- [ ] **Step 4: Commit Task 3**

```bash
git add examples/qwen3-32b-multi-gpu/README.md
git commit -m "docs: explain qwen benchmark progress"
```
