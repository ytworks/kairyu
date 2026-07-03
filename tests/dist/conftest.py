"""Spawn harness (m16 D5/A6): file:// rendezvous, polled join, JSON results."""

import json
import time
from pathlib import Path

import pytest
import torch.multiprocessing as mp

SPAWN_TIMEOUT_S = 120


@pytest.fixture()
def spawn2(tmp_path):
    """Run a dist_targets function on 2 ranks; returns per-rank result dicts."""

    def run(target, *args):
        out_dir = tmp_path / "results"
        out_dir.mkdir(exist_ok=True)
        init_file = tmp_path / f"rdv-{target.__name__}"
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
