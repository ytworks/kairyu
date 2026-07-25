"""Topology parsing is pure, so it is CPU-tested against real captures.

The fixtures are verbatim `nvidia-smi topo -m` output: the 8x RTX PRO 6000
PCIe host this runs on, and an NVLink box for the branch that host cannot
exercise.
"""

from kairyu.engine.core.gpu_topology import (
    interconnect_from_topology,
    parse_topology_matrix,
    slowest_link,
)

# verbatim from the 8x RTX PRO 6000 Blackwell host, ANSI escapes included:
# pairs (0,1) (2,3) (4,5) (6,7) share a NUMA node, everything else crosses
_PCIE_HEADER = (
    "\x1b[4mGPU0\tGPU1\tGPU2\tGPU3\tGPU4\tGPU5\tGPU6\tGPU7\tNIC0\tNIC1"
    "\tCPU Affinity\tNUMA Affinity\tGPU NUMA ID\x1b[0m"
)
PCIE_8GPU = _PCIE_HEADER + """
GPU0\t X \tNODE\tSYS\tSYS\tSYS\tSYS\tSYS\tSYS\tSYS\tSYS\t0-15,64-79\t0\t\tN/A
GPU1\tNODE\t X \tSYS\tSYS\tSYS\tSYS\tSYS\tSYS\tSYS\tSYS\t0-15,64-79\t0\t\tN/A
GPU2\tSYS\tSYS\t X \tNODE\tSYS\tSYS\tSYS\tSYS\tSYS\tSYS\t16-31,80-95\t1\t\tN/A
GPU3\tSYS\tSYS\tNODE\t X \tSYS\tSYS\tSYS\tSYS\tSYS\tSYS\t16-31,80-95\t1\t\tN/A
GPU4\tSYS\tSYS\tSYS\tSYS\t X \tNODE\tSYS\tSYS\tNODE\tNODE\t32-47,96-111\t2\t\tN/A
GPU5\tSYS\tSYS\tSYS\tSYS\tNODE\t X \tSYS\tSYS\tNODE\tNODE\t32-47,96-111\t2\t\tN/A
GPU6\tSYS\tSYS\tSYS\tSYS\tSYS\tSYS\t X \tNODE\tSYS\tSYS\t48-63,112-127\t3\t\tN/A
GPU7\tSYS\tSYS\tSYS\tSYS\tSYS\tSYS\tNODE\t X \tSYS\tSYS\t48-63,112-127\t3\t\tN/A
NIC0\tSYS\tSYS\tSYS\tSYS\tNODE\tNODE\tSYS\tSYS\t X \tPIX\t\t\t
NIC1\tSYS\tSYS\tSYS\tSYS\tNODE\tNODE\tSYS\tSYS\tPIX\t X \t\t\t

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes
"""

NVLINK_4GPU = """\
\tGPU0\tGPU1\tGPU2\tGPU3\tCPU Affinity\tNUMA Affinity
GPU0\t X \tNV12\tNV12\tNV12\t0-31\t0
GPU1\tNV12\t X \tNV12\tNV12\t0-31\t0
GPU2\tNV12\tNV12\t X \tNV12\t32-63\t1
GPU3\tNV12\tNV12\tNV12\t X \t32-63\t1
"""


def test_parses_every_gpu_pair_and_drops_the_nic_rows():
    matrix = parse_topology_matrix(PCIE_8GPU)
    assert len({row for row, _ in matrix}) == 8  # NIC rows excluded
    assert len(matrix) == 64
    assert matrix[(0, 0)] == "X"
    assert matrix[(0, 1)] == "NODE"  # same NUMA node
    assert matrix[(0, 7)] == "SYS"  # across the socket


def test_numa_local_pairs_are_symmetric():
    matrix = parse_topology_matrix(PCIE_8GPU)
    for first, second in ((0, 1), (2, 3), (4, 5), (6, 7)):
        assert matrix[(first, second)] == "NODE"
        assert matrix[(second, first)] == "NODE"


def test_pcie_host_is_not_reported_as_nvlink():
    # NODE starts with "NO", not "NV" — the distinction decides whether the
    # roadmap's NVLink-HBM or PCIe-fleet strategy applies
    assert interconnect_from_topology(parse_topology_matrix(PCIE_8GPU)) == "pcie"


def test_nvlink_host_is_detected():
    assert interconnect_from_topology(parse_topology_matrix(NVLINK_4GPU)) == "nvlink"


def test_unparseable_input_yields_no_topology():
    assert parse_topology_matrix("no matrix here") == {}
    assert interconnect_from_topology({}) == "none"


def test_slowest_link_picks_the_all_reduce_floor():
    matrix = parse_topology_matrix(PCIE_8GPU)
    # a TP pair inside one NUMA node never leaves it...
    assert slowest_link(matrix, (0, 1)) == "NODE"
    # ...while any group spanning nodes is bounded by the SYS hop
    assert slowest_link(matrix, (0, 2)) == "SYS"
    assert slowest_link(matrix, (0, 1, 2, 3)) == "SYS"
    assert slowest_link(matrix, tuple(range(8))) == "SYS"


def test_slowest_link_of_a_single_rank_is_undefined():
    assert slowest_link(parse_topology_matrix(PCIE_8GPU), (3,)) is None


def test_record_env_marks_a_skipped_measurement_instead_of_implying_zero(monkeypatch):
    """--no-measure must say so: a null next to real numbers reads as measured-0."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "gpu_gates"))
    import record_env

    from kairyu.engine.core.hw_profile import HardwareProfile

    outputs = {
        "driver_version": "595.84",
        "topo": PCIE_8GPU,
    }

    def fake_run(command):
        joined = " ".join(command)
        if "driver_version" in joined:
            return outputs["driver_version"]
        if "topo" in joined:
            return outputs["topo"]
        return "0, RTX PRO 6000, 0000:16:00.0, 97887 MiB, N/A, 98.04.30.00.01"

    record = record_env.build_env_record(
        profile=HardwareProfile(arch="cuda", device_count=8, interconnect="pcie"),
        record_date="2026-07-25",
        run_command=fake_run,
        measure=False,
    )
    assert record.profile.p2p_matrix is None
    assert record.profile.measured_bandwidth_gbs is None
    assert "--no-measure" in record.notes
    assert "skipped" in record.notes
    # the topology is still authoritative for interconnect even without timing
    assert record.profile.interconnect == "pcie"


def test_environment_records_are_not_ignored_by_git():
    """Review [P2] on #128: `!bench/results/env-*.json` under `bench/results/`
    never matched, because git does not descend into an excluded directory. The
    files already committed with `-f` stayed, but the NEXT env record would have
    vanished from `git status` and the evidence gap would have recurred."""
    import subprocess
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]

    def ignored(relative: str) -> bool:
        result = subprocess.run(
            ["git", "check-ignore", "-q", relative],
            cwd=repo, capture_output=True, timeout=30,
        )
        return result.returncode == 0

    # a future environment record must be tracked without `git add -f`
    assert not ignored("bench/results/env-2099-01-01.json")
    # while routine per-run output stays ignored
    assert ignored("bench/results/parity-tp-2099-01-01.json")
    assert ignored("bench/data/whatever.bin")
