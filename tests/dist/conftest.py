"""Spawn harness (m16 D5/A6): file:// rendezvous, polled join, JSON results."""

import itertools
import json
import sys
import time
from pathlib import Path

import pytest
import torch.multiprocessing as mp

SPAWN_TIMEOUT_S = 120


@pytest.fixture()
def spawn2(tmp_path, monkeypatch):
    """Run a dist_targets function on 2 ranks; returns per-rank result dicts."""

    # Every target in this fixture runs on the same host.  A file rendezvous
    # does not constrain Gloo's transport interface, so runner-specific virtual
    # NIC discovery can advertise an address that the sibling process cannot
    # reach.  Bind the real loopback interface and keep the collective/result
    # assertions independent of host networking.
    if sys.platform.startswith("linux"):
        monkeypatch.setenv("GLOO_SOCKET_IFNAME", "lo")
    elif sys.platform == "darwin":
        monkeypatch.setenv("GLOO_SOCKET_IFNAME", "lo0")

    rendezvous_ids = itertools.count()

    def run(target, *args):
        out_dir = tmp_path / "results"
        out_dir.mkdir(exist_ok=True)
        # FileStore requires a fresh path.  Some tests deliberately invoke the
        # same target twice to prove replay determinism; reusing its old path
        # leaves prior rendezvous state and cleanup races in the second group.
        init_file = tmp_path / f"rdv-{target.__name__}-{next(rendezvous_ids)}"
        context = mp.start_processes(
            target,
            args=(2, str(init_file), str(out_dir), *args),
            nprocs=2,
            join=False,
            start_method="spawn",
        )
        deadline = time.monotonic() + SPAWN_TIMEOUT_S
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                for process in context.processes:
                    process.terminate()
                pytest.fail(f"{target.__name__} timed out after {SPAWN_TIMEOUT_S}s")
        results = []
        for rank in range(2):
            path = Path(out_dir, f"rank{rank}.json")
            assert path.is_file(), f"rank {rank} produced no result"
            results.append(json.loads(path.read_text()))
        return results

    return run
