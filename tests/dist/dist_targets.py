"""Module-level spawn targets (m16 A6: must be importable, side-effect free).

Each target: init gloo via file:// rendezvous, run its scenario, write its
result as JSON to ``out_dir/rank{r}.json`` (crash-safe result transport).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch


def _setup(rank: int, world_size: int, init_file: str):
    from kairyu.engine.core.dist_comm import TorchDistCommunicator, init_distributed

    torch.set_num_threads(1)
    init_distributed(rank, world_size, f"file://{init_file}")
    return TorchDistCommunicator()


def _finish(out_dir: str, rank: int, payload: dict) -> None:
    Path(out_dir, f"rank{rank}.json").write_text(json.dumps(payload))
    import torch.distributed as dist

    dist.barrier()
    dist.destroy_process_group()


def comm_contract(rank: int, world_size: int, init_file: str, out_dir: str) -> None:
    comm = _setup(rank, world_size, init_file)
    broadcasted = comm.broadcast({"step": 7} if rank == 0 else None, src=0)
    reduced = comm.all_reduce((float(rank + 1), 10.0))
    gathered = comm.all_gather(f"r{rank}")
    if rank == 0:
        comm.send(1, {"hello": rank})
        received = None
    else:
        received = comm.recv(0)
    # uneven all_to_all: rank0 sends [1,3] rows, rank1 sends [2,2]
    sizes = [[1, 3], [2, 2]][rank]
    payload = torch.arange(sum(sizes), dtype=torch.float32) + 100 * rank
    recv_sizes = [[1, 2], [3, 2]][rank]
    out = torch.empty(sum(recv_sizes))
    comm.tensor_all_to_all_single(out, payload, recv_sizes, sizes)
    _finish(out_dir, rank, {
        "broadcast": broadcasted,
        "reduced": list(reduced),
        "gathered": list(gathered),
        "received": received,
        "a2a": out.tolist(),
    })


def tp_engine_parity(rank: int, world_size: int, init_file: str, out_dir: str,
                     model_dir: str, prompt: list[int], max_new: int) -> None:
    """Rank 0 drives EngineCore via DistTPModelRunner; rank 1 runs the worker loop."""
    from kairyu.engine.core.worker import (
        DistTPModelRunner,
        build_tp_runner,
        make_handshake,
        validate_handshake,
        worker_step_loop,
    )

    comm = _setup(rank, world_size, init_file)
    num_pages, page_size = 64, 4
    runner, _ = build_tp_runner(model_dir, world_size, rank, comm, num_pages, page_size)
    handshake = comm.broadcast(
        make_handshake(model_dir, num_pages, page_size) if rank == 0 else None, src=0
    )
    validate_handshake(handshake, model_dir, num_pages, page_size)

    if rank == 0:
        from kairyu.engine.core.engine_core import EngineCore
        from kairyu.engine.core.radix_kv import RadixKVCache
        from kairyu.engine.core.sampling_types import EngineSampling
        from kairyu.engine.core.scheduler import EngineRequest, Scheduler

        cache = RadixKVCache(num_pages=num_pages, page_size=page_size)
        scheduler = Scheduler(cache, max_num_batched_tokens=6, page_size=page_size)
        dist_runner = DistTPModelRunner(comm, runner)
        engine = EngineCore(scheduler, dist_runner)
        engine.add_request(
            EngineRequest("a", tuple(prompt), max_new_tokens=max_new,
                          sampling=EngineSampling())
        )
        outputs = engine.run_to_completion()["a"]
        dist_runner.shutdown()
        _finish(out_dir, rank, {"outputs": list(outputs)})
    else:
        steps = worker_step_loop(comm, runner)
        _finish(out_dir, rank, {"steps": steps})


def ep_block_parity(rank: int, world_size: int, init_file: str, out_dir: str,
                    model_dir: str) -> None:
    """EP=2 EpMoeBlock forward vs the saved single-process reference output."""
    from kairyu.models.loader import load_model
    from kairyu.models.moe_parallel import EpMoeBlock

    comm = _setup(rank, world_size, init_file)
    model, config, _ = load_model(model_dir)
    block = model.model.layers[1].mlp  # sparse layer of the moe fixture
    torch.manual_seed(61)
    hidden = torch.randn(9, config.hidden_size)
    reference = block(hidden)
    ep_block = EpMoeBlock(block, comm, ep_rank=rank, ep_size=world_size)
    out = ep_block(hidden)
    diff = (out - reference).abs().max().item()
    _finish(out_dir, rank, {"maxdiff": diff, "local_experts": len(ep_block.local_experts)})


def pp_greedy_parity(rank: int, world_size: int, init_file: str, out_dir: str,
                     model_dir: str, prompt: list[int], max_new: int) -> None:
    from kairyu.engine.core.kv_pool import PagedKVPool
    from kairyu.engine.core.pp_worker import PpStageModel, pp_greedy_generate
    from kairyu.models.loader import load_model

    comm = _setup(rank, world_size, init_file)
    full, config, _ = load_model(model_dir)
    stage = PpStageModel(full, num_stages=world_size, stage=rank)
    pool = PagedKVPool(
        num_layers=stage.num_local_layers, num_pages=64, page_size=4,
        num_kv_heads=config.kv_cache_num_heads, head_dim=config.kv_cache_head_dim,
    )
    page_table = list(range(32))
    outputs = pp_greedy_generate(stage, comm, pool, page_table, prompt, max_new)
    _finish(out_dir, rank, {"outputs": outputs})

