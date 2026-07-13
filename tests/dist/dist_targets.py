"""Module-level spawn targets (m16 A6: must be importable, side-effect free).

Each target: init gloo via file:// rendezvous, run its scenario, write its
result as JSON to ``out_dir/rank{r}.json`` (crash-safe result transport).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import torch


class _TPTokenizer:
    def __init__(self, vocab: list[str]) -> None:
        self._vocab = list(vocab)
        self.eos_token_id = self._vocab.index("<eos>")

    def encode(self, text: str) -> tuple[int, ...]:
        return (self._vocab.index("a"),)

    def decode(self, token_ids: Sequence[int]) -> str:
        return "".join(
            self._vocab[token_id]
            for token_id in token_ids
            if token_id != self.eos_token_id
        )

    def vocab(self) -> list[str]:
        return list(self._vocab)


class _ReleaseRecordingRunner:
    def __init__(self, runner) -> None:
        self.runner = runner
        self.released: list[str] = []

    def execute(self, scheduled, states):
        return self.runner.execute(scheduled, states)

    def release(self, request_id: str) -> None:
        self.released.append(request_id)
        self.runner.release(request_id)


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
                     model_dir: str, prompt: list[int], max_new: int,
                     vocab: list[str]) -> None:
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
    runner, _ = build_tp_runner(
        model_dir, world_size, rank, comm, num_pages, page_size, vocab
    )
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


def tp_structured_release(
    rank: int,
    world_size: int,
    init_file: str,
    out_dir: str,
    model_dir: str,
    vocab: list[str],
) -> None:
    """Structured TP sampling and explicit release on every owning rank."""
    from kairyu.engine.core.radix_kv import RadixKVCache
    from kairyu.engine.core.scheduler import Scheduler
    from kairyu.engine.core.worker import (
        DistTPModelRunner,
        build_tp_runner,
        make_handshake,
        validate_handshake,
        worker_step_loop,
    )
    from kairyu.engine.engine_loop import EngineLoop
    from kairyu.sampling_params import SamplingParams

    comm = _setup(rank, world_size, init_file)
    num_pages, page_size = 64, 4
    runner, _ = build_tp_runner(
        model_dir, world_size, rank, comm, num_pages, page_size, vocab
    )
    recording = _ReleaseRecordingRunner(runner)
    handshake = comm.broadcast(
        make_handshake(model_dir, num_pages, page_size) if rank == 0 else None, src=0
    )
    validate_handshake(handshake, model_dir, num_pages, page_size)
    payload: dict = {"structured_completed": False}

    if rank == 0:
        dist_runner = DistTPModelRunner(comm, recording)
        loop = EngineLoop(
            _TPTokenizer(vocab),
            Scheduler(
                RadixKVCache(num_pages=num_pages, page_size=page_size),
                max_num_batched_tokens=6,
                page_size=page_size,
            ),
            dist_runner,
        )

        def drive_to_terminal(request_id: str):
            for _ in range(64):
                for update_id, update in loop.step():
                    if update_id == request_id and update.finished:
                        return update
            raise AssertionError(f"request {request_id!r} did not finish")

        try:
            loop.submit(
                "structured",
                "prompt",
                SamplingParams(
                    max_tokens=16,
                    temperature=0.0,
                    extra_args={
                        "response_format": {
                            "type": "json_schema",
                            "json_schema": {
                                "schema": {
                                    "type": "object",
                                    "properties": {"a": {"type": "integer"}},
                                    "required": ["a"],
                                    "additionalProperties": False,
                                }
                            },
                        }
                    },
                ),
            )
            structured = drive_to_terminal("structured")
            payload["structured_completed"] = (
                structured.finish_reason == "stop"
                and json.loads(structured.text) == {"a": 1}
            )

            for index in range(32):
                request_id = f"short-{index}"
                loop.submit(
                    request_id,
                    "prompt",
                    SamplingParams(max_tokens=1, temperature=0.0),
                )
                drive_to_terminal(request_id)

            loop.submit(
                "cancelled",
                "prompt",
                SamplingParams(max_tokens=16, temperature=0.0, ignore_eos=True),
            )
            loop.step()
            if "cancelled" not in runner._sampler._states:
                raise AssertionError("cancellation request was not sampled before abort")
            loop.abort("cancelled")
            cancelled = drive_to_terminal("cancelled")
            if cancelled.finish_reason != "abort":
                raise AssertionError("cancellation did not finish with abort")
        except Exception as error:
            payload["error"] = f"{type(error).__name__}: {error}"
        finally:
            dist_runner.shutdown()
    else:
        try:
            payload["steps"] = worker_step_loop(comm, recording)
        except Exception as error:
            payload["error"] = f"{type(error).__name__}: {error}"
            # Rank 0's finally block sends shutdown after its matching local
            # execute fails. Receive it so both ranks reach _finish().
            try:
                comm.broadcast(None, src=0)
            except Exception:
                pass

    payload["sampler_states"] = len(runner._sampler._states)
    payload["released_requests"] = len(recording.released)
    _finish(out_dir, rank, payload)


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


def pd_prefill_process(rank: int, world_size: int, init_file: str, out_dir: str,
                       model_dir: str, prompt: list[int], max_new: int) -> None:
    """m18 D5 prefill half: prefill token 0, extract page BYTES between
    execute() and update() (the copy-before-commit point), send over TCP."""
    import asyncio
    import hashlib
    import time

    from kairyu.engine.core.kv_pool import PagedKVPool
    from kairyu.engine.core.kv_serde import extract_pages, pool_fingerprint
    from kairyu.engine.core.kv_transport import SequenceMeta, TcpLoopbackTransport
    from kairyu.engine.core.model_runner import PagedModelRunner
    from kairyu.engine.core.radix_kv import RadixKVCache
    from kairyu.engine.core.sampler import Sampler
    from kairyu.engine.core.sampling_types import EngineSampling
    from kairyu.engine.core.scheduler import EngineRequest, Scheduler
    from kairyu.models.loader import load_model

    torch.set_num_threads(1)
    model, config, _ = load_model(model_dir)
    cache = RadixKVCache(num_pages=64, page_size=4)
    scheduler = Scheduler(cache, max_num_batched_tokens=6, page_size=4)
    pool = PagedKVPool.for_cache(cache, config)
    runner = PagedModelRunner(model, pool, sampler=Sampler())

    # rendezvous: decode wrote "addr|fingerprint"
    deadline = time.monotonic() + 60
    address = None
    while time.monotonic() < deadline:
        try:
            content = Path(init_file).read_text()
            if "|" in content:
                address, fingerprint = content.split("|")
                break
        except FileNotFoundError:
            pass
        time.sleep(0.05)
    assert address, "decode process never published its address"
    assert fingerprint == pool_fingerprint(pool), "pool mismatch (m18 D1 handshake)"

    scheduler.add_request(
        EngineRequest("req", tuple(prompt), max_new_tokens=1, sampling=EngineSampling())
    )
    transport = TcpLoopbackTransport("prefill")
    transport.register(pool.num_pages)
    hashes: list[str] = []

    async def _run() -> None:
        while scheduler.has_unfinished():
            plan = scheduler.schedule()
            sampled = runner.execute(plan.scheduled, scheduler.states)
            if "req" in sampled:
                # copy-before-commit: extract while every page is locked
                pages = tuple(scheduler.states["req"].allocation.pages)
                frames = extract_pages(pool, pages)
                for frame in frames:
                    hashes.append(hashlib.sha256(b"".join(frame.fragments)).hexdigest())
                token0 = sampled["req"][0].token_id
                await transport.send(
                    address, frames, SequenceMeta(tuple(prompt), token0)
                )
            scheduler.update({rid: [t.token_id for t in ts] for rid, ts in sampled.items()})
        await transport.close()

    asyncio.run(_run())
    Path(out_dir, f"rank{rank}.json").write_text(json.dumps({"hashes": hashes}))


def pd_decode_process(rank: int, world_size: int, init_file: str, out_dir: str,
                      model_dir: str, prompt: list[int], max_new: int) -> None:
    """m18 D5 decode half: recv bytes, adopt via resume_with_kv, decode."""
    import asyncio
    import hashlib

    from kairyu.engine.core.engine_core import EngineCore
    from kairyu.engine.core.kv_pool import PagedKVPool
    from kairyu.engine.core.kv_serde import pool_fingerprint
    from kairyu.engine.core.kv_transport import TcpLoopbackTransport
    from kairyu.engine.core.model_runner import PagedModelRunner
    from kairyu.engine.core.pd_remote import RemoteKVReceiver
    from kairyu.engine.core.radix_kv import RadixKVCache
    from kairyu.engine.core.sampler import Sampler
    from kairyu.engine.core.sampling_types import EngineSampling
    from kairyu.engine.core.scheduler import EngineRequest, Scheduler
    from kairyu.models.loader import load_model

    torch.set_num_threads(1)
    model, config, _ = load_model(model_dir)
    cache = RadixKVCache(num_pages=64, page_size=4)
    scheduler = Scheduler(cache, max_num_batched_tokens=6, page_size=4)
    pool = PagedKVPool.for_cache(cache, config)
    runner = PagedModelRunner(model, pool, sampler=Sampler())
    receiver = RemoteKVReceiver(cache, pool)
    transport = TcpLoopbackTransport("decode")
    transport.register(pool.num_pages)

    async def _run() -> dict:
        address = await transport.start_server()
        Path(init_file).write_text(f"{address}|{pool_fingerprint(pool)}")
        frames, meta = await transport.recv("prefill")
        received_hashes = [
            hashlib.sha256(b"".join(frame.fragments)).hexdigest() for frame in frames
        ]
        allocation = receiver.adopt(frames, meta)
        request = EngineRequest(
            "req", tuple(meta.token_ids), max_new_tokens=max_new,
            sampling=EngineSampling(),
        )
        engine = EngineCore(scheduler, runner)
        finished = scheduler.resume_with_kv(request, allocation, meta.first_token)
        while not finished and scheduler.has_unfinished():
            engine.step()
            finished = not scheduler.has_unfinished()
        outputs = list(scheduler.output_tokens("req"))
        await transport.close()
        return {
            "outputs": outputs,
            "hashes": received_hashes,
            "injected": receiver.injected_pages,
        }

    result = asyncio.run(_run())
    Path(out_dir, f"rank{rank}.json").write_text(json.dumps(result))


def pd_two_process(rank: int, world_size: int, init_file: str, out_dir: str,
                   model_dir: str, prompt: list[int], max_new: int) -> None:
    """Rank 0 = prefill, rank 1 = decode (no torch.distributed needed)."""
    if rank == 0:
        pd_prefill_process(rank, world_size, init_file, out_dir, model_dir, prompt, max_new)
    else:
        pd_decode_process(rank, world_size, init_file, out_dir, model_dir, prompt, max_new)
