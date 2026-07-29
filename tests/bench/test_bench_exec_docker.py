"""Real, non-skipping Docker conformance gate for issue #210.

The default suite deselects this marker. CI builds the digest-pinned execution
image and runs this file explicitly; a missing daemon/image is a failure, never
reported as a skipped/pass result.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from uuid import uuid4

import pytest

from kairyu.bench.adapters.livecodebench import grade_code
from kairyu.bench.execution import build_execution_runner
from kairyu.bench.types import ExecutionConfig

pytestmark = pytest.mark.docker_exec


@dataclass
class _RunnerPair:
    local: object
    container: object
    container_names: list[str]

    def __iter__(self):
        return iter((self.local, self.container))


@pytest.fixture(scope="module")
def runners():
    image = os.environ.get("KAIRYU_BENCH_EXEC_IMAGE")
    assert image, "KAIRYU_BENCH_EXEC_IMAGE must name the pinned CI image"
    local = build_execution_runner(ExecutionConfig())
    container = build_execution_runner(
        ExecutionConfig(
            runner="docker",
            image=image,
            cpus=0.5,
            pids_limit=16,
            disk_mb=16,
        )
    )
    for runner in (local, container):
        available, detail = runner.availability()
        assert available, detail
    container_names: list[str] = []
    nonce = uuid4().hex

    def next_name() -> str:
        name = f"kairyu-bench-test210-{nonce}-{len(container_names)}"
        container_names.append(name)
        return name

    container._name_factory = next_name
    pair = _RunnerPair(local, container, container_names)
    yield pair
    leaked = {
        name: ids
        for name in container_names
        if (ids := _bench_containers(name))
    }
    assert not leaked, f"execution containers leaked after module: {leaked}"


def _semantic(result) -> tuple[bool, bool]:
    return result.ok, result.timed_out


def _bench_containers(name: str) -> set[str]:
    result = subprocess.run(
        ["docker", "ps", "-aq", "--filter", f"name=^/{name}$"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return set(result.stdout.split())


def test_local_and_container_have_identical_scoring_semantics(runners):
    local, container = runners
    code = (
        "import sys\n"
        "payload = open('fixture.txt', encoding='utf-8').read().strip()\n"
        "print(payload + ':' + sys.stdin.read().strip().upper())\n"
    )
    expected = None
    for runner in (local, container):
        result = runner.run_python(
            code,
            stdin="input",
            files={"fixture.txt": b"artifact"},
            timeout_s=3,
            memory_mb=128,
        )
        assert result.ok, result.stderr
        assert result.stdout.strip() == "artifact:INPUT"
        expected = _semantic(result) if expected is None else expected
        assert _semantic(result) == expected

    for runner in (local, container):
        failed = runner.run_python(
            "raise RuntimeError('representative failure')",
            timeout_s=3,
            memory_mb=128,
        )
        assert _semantic(failed) == (False, False)
        assert "representative failure" in failed.stderr

    for runner in (local, container):
        timed_out = runner.run_python(
            "import time; time.sleep(30)",
            timeout_s=3.0,
            memory_mb=128,
        )
        assert _semantic(timed_out) == (False, True)
    for name in runners.container_names:
        assert _bench_containers(name) == set()


def test_livecodebench_pass_and_wrong_answer_match_across_runners(runners):
    tests = [{"input": "7\n", "output": "14\n"}]
    verdicts = []
    for runner in runners:
        passing = grade_code(
            "value = int(input()); print(value * 2)",
            tests,
            fn_name=None,
            runner=runner,
        )
        wrong = grade_code(
            "value = int(input()); print(value * 3)",
            tests,
            fn_name=None,
            runner=runner,
        )
        verdicts.append((passing[0], wrong[0]))
    assert verdicts == [(True, False), (True, False)]


def test_container_enforces_network_rootfs_and_environment_boundaries(
    runners, monkeypatch
):
    _local, container = runners
    monkeypatch.setenv("KAIRYU_HOST_SECRET", "must-not-cross")
    code = """
import os
import socket

status = {}
with open("/proc/self/status", encoding="utf-8") as handle:
    for line in handle:
        if ":" in line:
            key, value = line.split(":", 1)
            status[key] = value.strip()

print("secret=" + str(os.environ.get("KAIRYU_HOST_SECRET")))
print(f"identity={os.getuid()}:{os.getgid()}")
print("cap_eff=" + status.get("CapEff", "missing"))
print("no_new_privs=" + status.get("NoNewPrivs", "missing"))
print("root_readonly=" + str(bool(os.statvfs("/").f_flag & os.ST_RDONLY)))
print("input_readonly=" + str(bool(os.statvfs("/input").f_flag & os.ST_RDONLY)))
try:
    with open("/tmp/kairyu-root-probe", "w", encoding="utf-8") as handle:
        handle.write("unexpected")
except OSError:
    print("rootfs=blocked")
else:
    print("rootfs=writable")

interfaces = sorted(name for _index, name in socket.if_nameindex())
print("interfaces=" + ",".join(interfaces))
"""
    result = container.run_python(code, timeout_s=3, memory_mb=128)
    assert result.ok, result.stderr
    observations = set(result.stdout.splitlines())
    assert "secret=None" in observations
    assert "identity=65534:65534" in observations
    assert "cap_eff=0000000000000000" in observations
    assert "no_new_privs=1" in observations
    assert "root_readonly=True" in observations
    assert "input_readonly=True" in observations
    assert "rootfs=blocked" in observations
    assert "interfaces=lo" in observations
    assert "rootfs=writable" not in observations


def test_read_only_large_artifact_does_not_consume_work_tmpfs(runners):
    _local, container = runners
    # Larger than the configured 16 MiB /work limit. This succeeds only if the
    # runner keeps the staged artifact on /input and exposes a read-only link.
    payload = b"x" * (20 * 1024 * 1024)
    result = container.run_python(
        "import os; print(os.path.getsize('large.bin'))",
        files={"large.bin": payload},
        timeout_s=4,
        memory_mb=128,
    )
    assert result.ok, result.stderr
    assert result.stdout.strip() == str(len(payload))


def test_container_enforces_memory_disk_pid_and_output_limits(runners):
    _local, container = runners

    limits = container.run_python(
        """
from pathlib import Path
import os

def first(*paths):
    for path in paths:
        candidate = Path(path)
        if candidate.exists():
            return candidate.read_text(encoding="utf-8").strip()
    raise RuntimeError(f"no cgroup control found in {paths}")

memory_max = first(
    "/sys/fs/cgroup/memory.max",
    "/sys/fs/cgroup/memory/memory.limit_in_bytes",
)
pids_max = first(
    "/sys/fs/cgroup/pids.max",
    "/sys/fs/cgroup/pids/pids.max",
)
if Path("/sys/fs/cgroup/cpu.max").exists():
    quota_text, period_text = first("/sys/fs/cgroup/cpu.max").split()
else:
    quota_text = first("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
    period_text = first("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
quota = int(quota_text)
period = int(period_text)
work = os.statvfs("/work")
work_bytes = work.f_blocks * work.f_frsize
with open("/work/small-write.bin", "wb") as handle:
    handle.write(b"x" * 1024 * 1024)
os.unlink("/work/small-write.bin")

print(f"memory_max={memory_max}")
print(f"pids_max={pids_max}")
print(f"cpu_ratio={quota / period}")
print(f"work_bytes={work_bytes}")
print("small_write=ok")
""",
        timeout_s=4,
        memory_mb=128,
    )
    assert limits.ok, limits.stderr
    observed_limits = dict(
        line.split("=", 1) for line in limits.stdout.splitlines()
    )
    assert observed_limits["memory_max"] == str(128 * 1024 * 1024)
    assert observed_limits["pids_max"] == "16"
    assert float(observed_limits["cpu_ratio"]) == 0.5
    assert 0 < int(observed_limits["work_bytes"]) <= 16 * 1024 * 1024
    assert observed_limits["small_write"] == "ok"

    memory = container.run_python(
        "x = bytearray(512 * 1024 * 1024); print('allocated')",
        timeout_s=4,
        memory_mb=64,
    )
    assert not memory.ok
    assert not memory.timed_out
    assert "allocated" not in memory.stdout

    disk = container.run_python(
        """
try:
    with open("growth.bin", "wb") as handle:
        for _ in range(64):
            handle.write(b"x" * 1024 * 1024)
    print("wrote-unbounded")
except OSError:
    print("disk-bounded")
""",
        timeout_s=4,
        memory_mb=128,
    )
    assert "wrote-unbounded" not in disk.stdout
    assert disk.ok, disk.stderr
    assert "disk-bounded" in disk.stdout.splitlines()

    pids = container.run_python(
        """
import subprocess
import sys

children = []
try:
    for _ in range(48):
        children.append(
            subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"])
        )
except OSError:
    pass
finally:
    for child in children:
        child.terminate()
    for child in children:
        child.wait()
print(len(children))
""",
        timeout_s=4,
        memory_mb=128,
    )
    assert pids.ok, pids.stderr
    assert int(pids.stdout.strip()) < 16

    output = container.run_python(
        "print('x' * 1_000_000)",
        timeout_s=4,
        memory_mb=128,
    )
    assert output.ok, output.stderr
    assert len(output.stdout.encode()) <= 64_000


def test_supplied_image_has_scicode_runtime(runners):
    _local, container = runners
    assert container.has_module("json")
    assert container.has_module("numpy")
    assert container.has_module("h5py")
    assert not container.has_module("definitely_not_a_module_xyz")
