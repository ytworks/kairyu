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

    def execute_passive(self, scheduled, states):
        return self.runner.execute_passive(scheduled, states)

    def make_sampling_token_packet(self, scheduled, states, sampled=None):
        return self.runner.make_sampling_token_packet(
            scheduled, states, sampled=sampled
        )

    def adopt_sampling_token_packet(self, scheduled, states, packet) -> None:
        self.runner.adopt_sampling_token_packet(scheduled, states, packet)

    def release(self, request_id: str) -> None:
        self.released.append(request_id)
        self.runner.release(request_id)


class _NcclScaleExpert(torch.nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.register_buffer("scale", torch.tensor(scale, dtype=torch.float32))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden * self.scale


class _NcclTinyMoeBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.experts = torch.nn.ModuleList(
            _NcclScaleExpert(scale) for scale in (0.5, -2.0, 3.0, 1.25)
        )
        self.register_buffer("route", torch.tensor([0, 2, 1, 3, 2, 0]))

    def _route(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if hidden.shape[0] != self.route.shape[0]:
            raise ValueError(f"expected {self.route.shape[0]} tokens, got {hidden.shape[0]}")
        weights = torch.ones(
            hidden.shape[0], 1, dtype=hidden.dtype, device=hidden.device
        )
        return self.route[:, None], weights

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        expert_ids, _ = self._route(hidden)
        out = torch.zeros_like(hidden)
        for expert_id, expert in enumerate(self.experts):
            mask = expert_ids[:, 0] == expert_id
            out[mask] = expert(hidden[mask])
        return out


class _EpInvariantExpert(torch.nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return torch.full_like(hidden, self.value)


class _EpInvariantMoeBlock(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.experts = torch.nn.ModuleList(
            _EpInvariantExpert(value)
            for value in (0.66015625, 0.267578125, 0.061767578125, 0.62109375)
        )

    def _route(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        indices = torch.tensor([[0, 2, 1, 3]], device=hidden.device)
        weights = torch.tensor(
            [[0.2010020762681961, 0.2674923539161682,
              0.06888598203659058, 0.4626196026802063]],
            dtype=torch.float32,
            device=hidden.device,
        )
        return indices.expand(hidden.shape[0], -1), weights.expand(hidden.shape[0], -1)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        indices, weights = self._route(hidden)
        rows = torch.stack(
            [self.experts[int(index)](hidden) for index in indices[0]],
            dim=1,
        )
        return (rows.float() * weights[:, :, None]).sum(dim=1).to(hidden.dtype)


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
    token_packet = (
        torch.tensor([17, -1, 29], dtype=torch.int64)
        if rank == 0
        else torch.empty(3, dtype=torch.int64)
    )
    comm.tensor_broadcast(token_packet, src=0)
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
        "token_packet": token_packet.tolist(),
        "received": received,
        "a2a": out.tolist(),
    })


def tp_engine_parity(rank: int, world_size: int, init_file: str, out_dir: str,
                     model_dir: str, prompt: list[int], max_new: int,
                     vocab: list[str], sampling_kwargs: dict | None = None) -> None:
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
                          sampling=EngineSampling(**(sampling_kwargs or {})))
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

    payload["sampling_owner"] = runner.sampling_owner
    payload["sampler_states"] = (
        0 if runner._sampler is None else len(runner._sampler._states)
    )
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


def ep_block_bf16_invariance(
    rank: int, world_size: int, init_file: str, out_dir: str
) -> None:
    """EP2 generic combine matches one fixed FP32 slot order in BF16."""
    from kairyu.models.moe_parallel import EpMoeBlock

    comm = _setup(rank, world_size, init_file)
    block = _EpInvariantMoeBlock()
    hidden = torch.zeros(1, 1, dtype=torch.bfloat16)
    reference = block(hidden)
    ep_block = EpMoeBlock(block, comm, ep_rank=rank, ep_size=world_size)
    out = ep_block(hidden)
    _finish(
        out_dir,
        rank,
        {
            "exact": torch.equal(out, reference),
            "actual": float(out.item()),
            "reference": float(reference.item()),
            "dtype": str(out.dtype),
        },
    )


def ep_block_nccl_parity(
    rank: int, world_size: int, init_file: str, out_dir: str
) -> None:
    """EP=2 dispatch/return parity with CUDA tensors and an NCCL process group."""
    import torch.distributed as dist

    from kairyu.engine.core.dist_comm import TorchDistCommunicator, init_distributed
    from kairyu.models.moe_parallel import EpMoeBlock

    if world_size != 2:
        raise ValueError(f"NCCL EP parity requires world_size=2, got {world_size}")
    torch.cuda.set_device(rank)
    init_distributed(rank, world_size, f"file://{init_file}", backend="nccl")
    try:
        comm = TorchDistCommunicator()
        device = torch.device("cuda", rank)
        block = _NcclTinyMoeBlock().to(device)
        hidden = torch.tensor(
            [
                [1.0, 2.0, -1.0],
                [0.5, -3.0, 4.0],
                [-2.0, 1.5, 3.0],
                [4.0, -0.5, 2.0],
                [3.0, 2.5, -4.0],
                [-1.0, 0.25, 5.0],
            ],
            dtype=torch.float32,
            device=device,
        )
        reference = block(hidden)
        ep_block = EpMoeBlock(block, comm, ep_rank=rank, ep_size=world_size)
        out = ep_block(hidden)
        result = {
            "max_error": (out - reference).abs().max().item(),
            "device": str(out.device),
            "local_experts": len(ep_block.local_experts),
        }
        Path(out_dir, f"rank{rank}.json").write_text(json.dumps(result))
    finally:
        if dist.is_initialized():
            try:
                dist.barrier()
            finally:
                dist.destroy_process_group()


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


def reduce_scatter_equivalence(
    rank: int, world_size: int, init_file: str, out_dir: str
) -> None:
    """reduce_scatter must equal all_reduce sliced to this rank's shard.

    On gloo the implementation IS that, so this pins the contract the NCCL path
    has to honour; the GPU mirror runs the real collective.
    """
    comm = _setup(rank, world_size, init_file)
    rows = 4 * world_size
    payload = torch.arange(rows * 3, dtype=torch.float32).reshape(rows, 3) + 100 * rank

    scattered = comm.tensor_reduce_scatter(payload)

    summed = payload.clone()
    comm.tensor_all_reduce(summed)
    span = rows // world_size
    expected = summed[rank * span : (rank + 1) * span]

    _finish(out_dir, rank, {
        "shape": list(scattered.shape),
        "max_error": float((scattered - expected).abs().max()),
        # re-gathering the shards must reconstruct the full all_reduce
        "regathered_matches": bool(
            torch.allclose(comm.tensor_all_gather(scattered.contiguous()), summed)
        ),
    })


def reduce_scatter_rejects_indivisible(
    rank: int, world_size: int, init_file: str, out_dir: str
) -> None:
    """A ragged first dimension has no equal shard; fail loudly, not silently."""
    comm = _setup(rank, world_size, init_file)
    payload = torch.ones(world_size * 2 + 1, 2)
    try:
        comm.tensor_reduce_scatter(payload)
        _finish(out_dir, rank, {"raised": False})
    except ValueError as error:
        _finish(out_dir, rank, {"raised": True, "message": str(error)})


def reduce_scatter_nccl_equivalence(
    rank: int, world_size: int, init_file: str, out_dir: str
) -> None:
    """The real NCCL collective, against the same all_reduce-and-slice contract."""
    import torch.distributed as dist

    from kairyu.engine.core.dist_comm import TorchDistCommunicator, init_distributed

    torch.cuda.set_device(rank)
    init_distributed(rank, world_size, f"file://{init_file}", backend="nccl")
    try:
        comm = TorchDistCommunicator(device=f"cuda:{rank}")
        device = torch.device("cuda", rank)
        rows = 4 * world_size
        payload = (
            torch.arange(rows * 3, dtype=torch.float32, device=device).reshape(rows, 3)
            + 100 * rank
        )
        scattered = comm.tensor_reduce_scatter(payload)
        summed = payload.clone()
        comm.tensor_all_reduce(summed)
        span = rows // world_size
        expected = summed[rank * span : (rank + 1) * span]
        Path(out_dir, f"rank{rank}.json").write_text(
            json.dumps(
                {
                    "shape": list(scattered.shape),
                    "max_error": float((scattered - expected).abs().max()),
                    "device": str(scattered.device),
                }
            )
        )
    finally:
        dist.destroy_process_group()


def tp_control_model_group_stress(
    rank: int, world_size: int, init_file: str, out_dir: str, rounds: int
) -> None:
    """Alternate control objects and model tensors on their real backends."""
    import torch.distributed as dist

    from kairyu.engine.core.dist_comm import TorchDistCommunicator, init_distributed
    from kairyu.engine.core.worker import serving_groups

    torch.cuda.set_device(rank)
    init_distributed(rank, world_size, f"file://{init_file}", backend="nccl")
    try:
        groups = serving_groups("nccl")
        control = TorchDistCommunicator(group=groups.control)
        model = TorchDistCommunicator(group=groups.model, device=f"cuda:{rank}")
        device = torch.device("cuda", rank)
        expected = world_size * (world_size + 1) / 2
        last_step = -1
        for step in range(rounds):
            payload = control.broadcast(
                {"step": step, "request_id": f"request-{step % 8}"}
                if rank == 0
                else None,
                src=0,
            )
            if payload["step"] != step:
                raise AssertionError(f"control sequence diverged at {step}: {payload}")
            reduced = model.tensor_all_reduce(
                torch.tensor([float(rank + 1)], device=device)
            )
            if reduced.item() != expected:
                raise AssertionError(
                    f"model reduction diverged at {step}: {reduced.item()} != {expected}"
                )
            last_step = step
        Path(out_dir, f"rank{rank}.json").write_text(
            json.dumps(
                {
                    "control_backend": dist.get_backend(groups.control),
                    "model_backend": dist.get_backend(groups.model),
                    "last_step": last_step,
                }
            )
        )
    finally:
        dist.destroy_process_group()


def control_idle_wait(
    rank: int, world_size: int, init_file: str, out_dir: str
) -> None:
    """A worker control receive may wait longer than the model timeout."""
    import time

    import torch.distributed as dist

    from kairyu.engine.core.dist_comm import TorchDistCommunicator, init_distributed
    from kairyu.engine.core.worker import serving_groups

    model_timeout_s = 0.1
    control_timeout_s = 2.0
    idle_s = 0.3
    init_distributed(rank, world_size, f"file://{init_file}", backend="gloo")
    groups = serving_groups(
        "gloo",
        control_timeout_s=control_timeout_s,
        model_timeout_s=model_timeout_s,
    )
    control = TorchDistCommunicator(group=groups.control)
    if rank == 0:
        time.sleep(idle_s)
    payload = control.broadcast({"after_idle": True} if rank == 0 else None, src=0)
    dist.destroy_process_group(groups.model)
    dist.destroy_process_group(groups.startup_control)
    dist.destroy_process_group(groups.control)
    _finish(
        out_dir,
        rank,
        {
            "payload": payload,
            "idle_s": idle_s,
            "model_timeout_s": model_timeout_s,
        },
    )


def sequence_parallel_parity(
    rank: int, world_size: int, init_file: str, out_dir: str, model_dir: str
) -> None:
    """SP must be the same arithmetic as plain TP, only sharded differently."""
    from kairyu.models.parallel import build_tp_model

    comm = _setup(rank, world_size, init_file)
    torch.manual_seed(11)
    tokens = torch.randint(0, 256, (12,))
    positions = torch.arange(12)

    from kairyu.engine.core.kv_pool import PagedKVPool

    def run(sequence_parallel: bool):
        model, local, _full = build_tp_model(
            model_dir, world_size, rank, comm, sequence_parallel=sequence_parallel
        )
        pool = PagedKVPool(
            num_layers=local.num_hidden_layers, num_pages=8, page_size=16,
            num_kv_heads=local.kv_cache_num_heads, head_dim=local.kv_cache_head_dim,
        )
        hidden = model.forward_tokens(
            tokens, positions, pool, [0, 1], seq_len=12, write_from=0
        )
        return model.logits(hidden[-1])

    plain = run(False)
    sharded = run(True)
    _finish(out_dir, rank, {
        "shape": list(sharded.shape),
        "max_error": float((plain - sharded).abs().max()),
        "argmax_equal": int(plain.argmax()) == int(sharded.argmax()),
    })


def sequence_parallel_ragged(
    rank: int, world_size: int, init_file: str, out_dir: str, model_dir: str
) -> None:
    """A token count that does not divide by tp must still be exact."""
    from kairyu.engine.core.kv_pool import PagedKVPool
    from kairyu.models.parallel import build_tp_model

    comm = _setup(rank, world_size, init_file)
    torch.manual_seed(13)
    length = 11  # odd on purpose: 11 % 2 != 0
    tokens = torch.randint(0, 256, (length,))
    positions = torch.arange(length)

    def run(sequence_parallel: bool):
        model, local, _full = build_tp_model(
            model_dir, world_size, rank, comm, sequence_parallel=sequence_parallel
        )
        pool = PagedKVPool(
            num_layers=local.num_hidden_layers, num_pages=8, page_size=16,
            num_kv_heads=local.kv_cache_num_heads, head_dim=local.kv_cache_head_dim,
        )
        return model.forward_tokens(
            tokens, positions, pool, [0, 1], seq_len=length, write_from=0
        )

    plain = run(False)
    sharded = run(True)
    _finish(out_dir, rank, {
        "rows": int(sharded.shape[0]),
        "expected_rows": length,
        "max_error": float((plain - sharded).abs().max()),
    })


def sequence_parallel_nccl_parity(
    rank: int, world_size: int, init_file: str, out_dir: str, model_dir: str
) -> None:
    """SP == plain TP on real devices, bf16, over NCCL."""
    import torch.distributed as dist

    from kairyu.engine.core.dist_comm import TorchDistCommunicator, init_distributed
    from kairyu.engine.core.kv_pool import PagedKVPool
    from kairyu.models.parallel import build_tp_model

    torch.cuda.set_device(rank)
    init_distributed(rank, world_size, f"file://{init_file}", backend="nccl")
    try:
        comm = TorchDistCommunicator(device=f"cuda:{rank}")
        device = torch.device("cuda", rank)
        torch.manual_seed(11)
        tokens = torch.randint(0, 256, (12,), device=device)
        positions = torch.arange(12, device=device)

        def run(sequence_parallel: bool):
            model, local, _full = build_tp_model(
                model_dir, world_size, rank, comm,
                dtype=torch.bfloat16, device=f"cuda:{rank}",
                sequence_parallel=sequence_parallel,
            )
            pool = PagedKVPool(
                num_layers=local.num_hidden_layers, num_pages=8, page_size=16,
                num_kv_heads=local.kv_cache_num_heads, head_dim=local.kv_cache_head_dim,
                dtype=torch.bfloat16, device=f"cuda:{rank}",
            )
            return model.forward_tokens(
                tokens, positions, pool, [0, 1], seq_len=12, write_from=0
            )

        plain, sharded = run(False), run(True)
        Path(out_dir, f"rank{rank}.json").write_text(
            json.dumps(
                {
                    "rows": int(sharded.shape[0]),
                    "max_error": float((plain.float() - sharded.float()).abs().max()),
                    "device": str(sharded.device),
                }
            )
        )
    finally:
        dist.destroy_process_group()


def direct_nccl_graph_workspace_growth(
    rank: int,
    world_size: int,
    init_file: str,
    out_dir: str,
) -> None:
    """Small capture remains correct after a larger capture grows workspaces."""

    import torch.distributed as dist

    from kairyu.engine.core.direct_nccl import DirectNcclCommunicator
    from kairyu.engine.core.dist_comm import TorchDistCommunicator, init_distributed

    torch.cuda.set_device(rank)
    init_distributed(rank, world_size, f"file://{init_file}", backend="gloo")
    direct = None
    small_graph = None
    large_graph = None
    try:
        control = TorchDistCommunicator()
        direct, metadata = DirectNcclCommunicator.try_create(
            control,
            torch.device("cuda", rank),
        )
        if direct is None:
            raise RuntimeError(f"direct NCCL unavailable: {metadata!r}")
        pool = torch.cuda.graph_pool_handle()

        def capture(rows: int, value: float):
            static_input = torch.full(
                (rows, 8),
                value,
                dtype=torch.bfloat16,
                device=f"cuda:{rank}",
            )
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                gathered = direct.all_gather_many((static_input,))[0]
                reduced = direct.reduce_scatter(gathered)
            torch.cuda.current_stream().wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool):
                gathered = direct.all_gather_many((static_input,))[0]
                reduced = direct.reduce_scatter(gathered)
            graph.replay()
            torch.cuda.synchronize()
            return graph, static_input, gathered, reduced

        small_graph, small_input, small_gather, small_reduce = capture(
            1,
            float(rank + 1),
        )
        small_gather_ptr = small_gather.data_ptr()
        small_reduce_ptr = small_reduce.data_ptr()
        control.barrier()
        large_graph, _large_input, _large_gather, _large_reduce = capture(
            4,
            float(rank + 101),
        )
        control.barrier()

        small_input.fill_(float(rank + 11))
        small_graph.replay()
        torch.cuda.synchronize()
        expected = torch.full_like(
            small_reduce,
            float(world_size * (rank + 11)),
        )
        max_error = float((small_reduce - expected).abs().max().item())
        retained_gather_ptrs = {
            tensor.data_ptr()
            for tensor in direct._captured_gather_workspaces.values()
        }
        retained_reduce_ptrs = {
            tensor.data_ptr()
            for tensor in direct._captured_reduce_scatter_workspaces.values()
        }
        Path(out_dir, f"rank{rank}.json").write_text(
            json.dumps(
                {
                    "max_error": max_error,
                    "small_gather_retained": (
                        small_gather_ptr in retained_gather_ptrs
                    ),
                    "small_reduce_retained": (
                        small_reduce_ptr in retained_reduce_ptrs
                    ),
                    "direct_nccl_active": metadata["direct_nccl_active"],
                }
            )
        )
        control.barrier()
    finally:
        if small_graph is not None:
            small_graph.reset()
        if large_graph is not None:
            large_graph.reset()
        if direct is not None:
            direct.close()
        dist.destroy_process_group()


def ep4_attention_dp_cuda_graph_startup_replay(
    rank: int,
    world_size: int,
    init_file: str,
    out_dir: str,
) -> None:
    """Capture EP4 attention-DP before readiness, then replay its first decode.

    The topology, layout lifecycle, CUDA graph, and direct-NCCL transport are
    production objects.  Tiny CUDA tensor shims replace only the checkpoint-
    owned NVFP4 kernels so this gate needs neither Qwen3-235B weights nor
    FlashInfer.
    """

    from types import SimpleNamespace

    import torch.distributed as dist

    from kairyu.engine.core import worker as worker_module
    from kairyu.engine.core.cuda_graph_gpu import CudaGraphBackend
    from kairyu.engine.core.dist_comm import TorchDistCommunicator, init_distributed
    from kairyu.engine.core.step_executor import GraphStepExecutor, build_decode_batch
    from kairyu.kernels import nvfp4_moe_gpu
    from kairyu.models.moe_parallel import (
        EpMoeBlock,
        assert_attention_dp_layouts_idle,
    )

    if world_size != 4:
        raise ValueError(f"attention-DP CUDA graph startup requires EP4, got {world_size}")
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    groups = None
    deferred = None
    executor = None
    result: dict[str, object] | None = None
    clean_shutdown = False
    try:
        init_distributed(
            rank,
            world_size,
            f"file://{init_file}",
            backend="nccl",
            timeout_s=60,
        )
        startup_comm = TorchDistCommunicator(device=device)
        deferred = worker_module._DeferredComm(startup_comm)

        base = _NcclTinyMoeBlock().to(device)
        base.route = torch.tensor([rank], dtype=torch.int64, device=device)
        block = EpMoeBlock(
            base,
            deferred,
            ep_rank=rank,
            ep_size=world_size,
            attention_dp=True,
        )
        object.__setattr__(
            block,
            "_prepared_nvfp4_moe",
            SimpleNamespace(
                fc1_act_scale=torch.tensor(
                    1.0,
                    dtype=torch.float32,
                    device=device,
                ),
                local_expert_start=rank,
            ),
        )
        model = torch.nn.ModuleList([block])

        def fake_quantize(
            hidden: torch.Tensor,
            _scale: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return (
                hidden[:, : hidden.shape[1] // 2].to(torch.uint8).contiguous(),
                torch.zeros(
                    hidden.shape[0],
                    hidden.shape[1] // 16,
                    dtype=torch.uint8,
                    device=hidden.device,
                ),
            )

        def fake_interleave(scale_rows: torch.Tensor) -> torch.Tensor:
            return scale_rows.reshape(-1).contiguous()

        def fake_fused(
            packed_hidden: torch.Tensor,
            _input_sf: torch.Tensor,
            selected_experts: torch.Tensor,
            routing_weights: torch.Tensor,
            prepared: SimpleNamespace,
            *,
            output: torch.Tensor,
            ep_size: int,
            ep_rank: int,
        ) -> torch.Tensor:
            if ep_size != 4 or ep_rank != prepared.local_expert_start:
                raise AssertionError("synthetic EP ownership drift")
            selected = (selected_experts[:, :1] == ep_rank).to(output.dtype)
            values = (
                packed_hidden[:, :1].to(output.dtype)
                * float(ep_rank + 1)
                * routing_weights[:, :1].to(output.dtype)
            )
            output.copy_((selected * values).expand_as(output))
            return output

        # EpMoeBlock imports these names inside every Python forward.  Child
        # processes own the replacements, and graph replay never re-enters them.
        nvfp4_moe_gpu.quantize_moe_input = fake_quantize
        nvfp4_moe_gpu.interleave_moe_input_scales = fake_interleave
        nvfp4_moe_gpu.fused_moe_forward_quantized = fake_fused

        groups = worker_module.serving_groups(
            "nccl",
            control_timeout_s=60,
            model_timeout_s=60,
        )
        if groups.control is groups.startup_control:
            raise AssertionError("startup capture reused the long-idle control group")
        startup_control = TorchDistCommunicator(group=groups.startup_control)
        model_comm = TorchDistCommunicator(group=groups.model, device=device)
        worker_module._bind_ep_model_communicator(
            deferred,
            model_comm,
            startup_control,
            attention_dp=True,
        )
        direct_metadata = model_comm.attention_dp_collective_metadata()
        if direct_metadata.get("direct_nccl_active") is not True:
            raise RuntimeError(f"direct NCCL unavailable: {direct_metadata!r}")

        forward_calls = [0]
        backend = CudaGraphBackend(warmup_iters=1)

        def decode(batch) -> torch.Tensor:
            forward_calls[0] += 1
            hidden = batch.token_ids.to(torch.bfloat16).unsqueeze(1).expand(-1, 16).contiguous()
            return block(hidden)

        executor = GraphStepExecutor(
            decode,
            backend,
            max_batch=1,
            scratch_page=7,
            max_pages=1,
            device=device,
        )

        class _TinyGraphRunner:
            def __init__(self) -> None:
                self._model = model

            def decode_graph_capture_plan(self):
                return executor.capture_plan()

            def capture_decode_graphs(self):
                return executor.capture_all()

            def synchronize_decode_graph_capture(self) -> None:
                torch.cuda.synchronize(device)

            def invalidate_graphs(self) -> None:
                executor.invalidate()

        captured = worker_module._capture_decode_graphs_transaction(
            startup_control,
            _TinyGraphRunner(),
            attention_dp=True,
        )
        ready_stats = executor.execution_stats()
        ready_layout = block.attention_dp_metadata()
        direct = model_comm._attention_dp_direct_nccl
        if direct is None:
            raise AssertionError("direct NCCL metadata was active without a communicator")
        ready = {
            "captured": captured,
            "backend_captures": backend.captures,
            "backend_replays": backend.replays,
            "graph_executions": ready_stats["graph_executions"],
            "fallbacks": ready_stats["eager_fallbacks"],
            "forward_calls": forward_calls[0],
            "completed_layout_steps": ready_layout["completed_layout_steps"],
            "layout_queue_depth": ready_layout["layout_queue_depth"],
            "direct_gather_capture_buffers": len(direct._captured_gather_workspaces),
            "direct_scatter_capture_buffers": len(direct._captured_reduce_scatter_workspaces),
        }

        token = rank + 2
        batch = build_decode_batch(
            token_ids=[token],
            positions=[0],
            page_lists=[[0]],
            seq_lens=[1],
            max_pages=1,
            scratch_page=7,
            device=device,
        )
        decision = executor.decode_decision(batch_size=1, max_pages=1)
        if decision.kind != "replay":
            raise AssertionError(f"first live decode was not a replay: {decision!r}")
        executor.coordinate_next_decode(decision)
        output = executor.execute_decode(batch)
        executor.assert_coordinated_decode_consumed()
        torch.cuda.synchronize(device)
        assert_attention_dp_layouts_idle(model)
        after_stats = executor.execution_stats()
        expected = torch.full_like(output, float(token * (rank + 1)))
        result = {
            "rank": rank,
            "device": str(device),
            "direct_active": True,
            "startup_control_distinct": True,
            "ready": ready,
            "first_decision": decision.kind,
            "after_backend_captures": backend.captures,
            "after_backend_replays": backend.replays,
            "after_graph_executions": after_stats["graph_executions"],
            "after_fallbacks": after_stats["eager_fallbacks"],
            "after_forward_calls": forward_calls[0],
            "after_completed_layout_steps": block.attention_dp_metadata()["completed_layout_steps"],
            "max_error": float((output - expected).abs().max().item()),
        }
        clean_shutdown = True
    finally:
        invalidation_error: Exception | None = None
        if executor is not None:
            try:
                executor.invalidate()
            except Exception as error:
                invalidation_error = error
        if clean_shutdown and deferred is not None and groups is not None:
            torch.cuda.synchronize(device)
            deferred.barrier()
            deferred.close_direct_nccl()
            if deferred.attention_dp_direct_nccl_active:
                raise AssertionError("normal teardown left direct NCCL active")
            dist.destroy_process_group(deferred.group)
            dist.destroy_process_group(groups.startup_control)
            dist.destroy_process_group(groups.control)
            dist.destroy_process_group()
            if invalidation_error is not None:
                raise invalidation_error
        elif deferred is not None:
            worker_module._abort_ep_worker_model_communicator(deferred, str(device))

    if result is None:
        raise AssertionError("EP4 startup replay completed without a result")
    result["direct_closed"] = not deferred.attention_dp_direct_nccl_active
    result["distributed_destroyed"] = not dist.is_initialized()
    result["normal_teardown"] = True
    Path(out_dir, f"rank{rank}.json").write_text(
        json.dumps(result),
        encoding="utf-8",
    )
