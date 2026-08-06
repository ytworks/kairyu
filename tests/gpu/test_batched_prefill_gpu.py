"""Real SM120/FlashInfer proof for issue #224."""

from __future__ import annotations

import gc

import pytest
import torch

from kairyu.bench.profiling import profile_scope
from kairyu.engine.core.prefill import PrefillSequence, build_prefill_batch

pytestmark = pytest.mark.gpu

PAGE = 16
TINY = {
    "architectures": ["Qwen3ForCausalLM"],
    "hidden_size": 256,
    "num_hidden_layers": 2,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
    "head_dim": 64,
    "intermediate_size": 512,
    "vocab_size": 1024,
    "rms_norm_eps": 1e-6,
}
TP_WORLD_SIZE = 2
TP_NUM_PAGES = 64
TP_PAGE_SIZE = 8
TP_VOCAB_SIZE = 1024
TP_PROMPTS = (
    (11, 12, 13, 14, 15, 16, 17, 18, 19),
    (31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43),
)


class _CountedWrapper:
    def __init__(self, inner) -> None:
        self.inner = inner
        self.plans = 0
        self.runs = 0

    def plan(self, *args, **kwargs):
        self.plans += 1
        return self.inner.plan(*args, **kwargs)

    def run(self, *args, **kwargs):
        self.runs += 1
        return self.inner.run(*args, **kwargs)

    def reset(self) -> None:
        self.plans = 0
        self.runs = 0

    def __getattr__(self, name):
        return getattr(self.inner, name)


@pytest.fixture(scope="module")
def cuda():
    if not torch.cuda.is_available():  # pragma: no cover - GPU suite only
        pytest.skip("CUDA required")
    return torch.device("cuda")


def _require_two_gpus() -> None:
    if not torch.cuda.is_available():  # pragma: no cover - GPU suite only
        pytest.skip("TP2 batched prefill requires CUDA")
    if torch.cuda.device_count() < TP_WORLD_SIZE:  # pragma: no cover - small host
        pytest.skip(
            f"TP2 batched prefill requires {TP_WORLD_SIZE} GPUs; "
            f"found {torch.cuda.device_count()}"
        )


@pytest.fixture(scope="module")
def tp2_model_dir(tmp_path_factory) -> str:
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(224)
    model = transformers.LlamaForCausalLM(
        transformers.LlamaConfig(
            vocab_size=TP_VOCAB_SIZE,
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=256,
        )
    )
    path = tmp_path_factory.mktemp("tp2-batched-prefill")
    model.to(torch.float32).eval().save_pretrained(
        path, safe_serialization=True
    )
    return str(path)


def _cuda_event_count(profile) -> int:
    return sum(
        event.count
        for event in profile.key_averages()
        if event.device_type == torch.autograd.DeviceType.CUDA
    )


def test_single_prefill_partial_write_avoids_dynamic_mask_host_sync(cuda):
    """Host chunk metadata must avoid scalar reads and dynamic CUDA masks."""
    from kairyu.engine.core.attention.flashinfer_gpu import FlashInferBackend
    from kairyu.engine.core.kv_pool import PagedKVPool
    from kairyu.models.config import parse_model_config
    from kairyu.models.llama import DenseDecoder

    torch.manual_seed(319)
    backend = FlashInferBackend(device=cuda)
    model = DenseDecoder(
        parse_model_config(TINY), attention_backend=backend
    ).to(device=cuda, dtype=torch.bfloat16).eval()
    pool = PagedKVPool(
        model.config.num_hidden_layers,
        4,
        PAGE,
        model.config.num_key_value_heads,
        model.config.head_dim,
        dtype=torch.bfloat16,
        device=cuda,
    )
    token_ids = torch.tensor([41, 42, 43, 44], dtype=torch.long, device=cuda)
    positions = torch.arange(4, 8, device=cuda)

    # Warm the exact FlashInfer plan and kernels before observing the candidate.
    model.forward_tokens(
        token_ids,
        positions,
        pool,
        [0],
        seq_len=8,
        write_from=6,
        chunk_start=4,
        has_writable=True,
    )
    torch.cuda.synchronize()

    torch.manual_seed(320)
    pool.k.copy_(torch.randn_like(pool.k))
    pool.v.copy_(torch.randn_like(pool.v))
    reference_pool = PagedKVPool(
        model.config.num_hidden_layers,
        4,
        PAGE,
        model.config.num_key_value_heads,
        model.config.head_dim,
        dtype=torch.bfloat16,
        device=cuda,
    )
    reference_pool.k.copy_(pool.k)
    reference_pool.v.copy_(pool.v)
    cached_k = pool.k[:, 0, 4:6].clone()
    cached_v = pool.v[:, 0, 4:6].clone()
    reference = model.forward_tokens(
        token_ids,
        positions,
        reference_pool,
        [0],
        seq_len=8,
        write_from=6,
    )
    torch.cuda.synchronize()

    with profile_scope(
        enabled=True,
        activities=("cpu", "cuda"),
        acc_events=True,
        record_shapes=False,
        profile_memory=False,
        with_stack=True,
    ) as candidate:
        actual = model.forward_tokens(
            token_ids,
            positions,
            pool,
            [0],
            seq_len=8,
            write_from=6,
            chunk_start=4,
            has_writable=True,
        )

    scalar_or_mask_ops = {
        "aten::nonzero",
        "aten::_local_scalar_dense",
        "aten::item",
        "aten::is_nonzero",
    }
    forbidden_events = [
        (
            event.name,
            None if event.cpu_parent is None else event.cpu_parent.name,
            event.stack,
        )
        for event in candidate.events()
        if event.name in scalar_or_mask_ops
        or (
            event.name == "cudaStreamSynchronize"
            and event.cpu_parent is not None
            and event.cpu_parent.name in scalar_or_mask_ops
        )
    ]
    assert not forbidden_events, forbidden_events
    assert torch.equal(pool.k[:, 0, 4:6], cached_k)
    assert torch.equal(pool.v[:, 0, 4:6], cached_v)
    assert torch.equal(pool.k, reference_pool.k)
    assert torch.equal(pool.v, reference_pool.v)
    assert torch.allclose(actual, reference, atol=1e-3)


def test_real_flashinfer_ragged_prefill_matches_sequential_and_reduces_launches(
    cuda,
):
    from kairyu.engine.core.attention.flashinfer_gpu import FlashInferBackend
    from kairyu.engine.core.kv_pool import PagedKVPool
    from kairyu.models.config import parse_model_config
    from kairyu.models.llama import DenseDecoder

    torch.manual_seed(224)
    backend = FlashInferBackend(device=cuda)
    counted = _CountedWrapper(backend._prefill)
    backend._prefill = counted
    model = DenseDecoder(
        parse_model_config(TINY), attention_backend=backend
    ).to(device=cuda, dtype=torch.bfloat16).eval()

    def new_pool():
        pool = PagedKVPool(
            model.config.num_hidden_layers,
            16,
            PAGE,
            model.config.num_key_value_heads,
            model.config.head_dim,
            dtype=torch.bfloat16,
            device=cuda,
        )
        prefix = torch.arange(1, PAGE + 1, dtype=torch.long, device=cuda)
        model.forward_tokens(
            prefix,
            torch.arange(PAGE, device=cuda),
            pool,
            [0],
            seq_len=PAGE,
            write_from=0,
        )
        torch.manual_seed(99)
        pool.k[:, 10].copy_(torch.randn_like(pool.k[:, 10]))
        pool.v[:, 10].copy_(torch.randn_like(pool.v[:, 10]))
        return pool

    rows = [
        PrefillSequence(
            "a",
            (101, 102, 103),
            (0, 1),
            PAGE,
            PAGE + 3,
            PAGE,
        ),
        PrefillSequence(
            "b",
            (201, 202, 203, 204, 205),
            (0, 2),
            PAGE,
            PAGE + 5,
            PAGE,
        ),
        PrefillSequence(
            "cold",
            (301, 302, 303, 304, 305, 306, 307),
            (3,),
            0,
            7,
            0,
        ),
    ]
    batch = build_prefill_batch(rows, page_size=PAGE, device=cuda)
    sequential_pool = new_pool()
    batched_pool = new_pool()
    cached_k = batched_pool.k[:, 0].clone()
    cached_v = batched_pool.v[:, 0].clone()
    poison_k = batched_pool.k[:, 10].clone()
    poison_v = batched_pool.v[:, 10].clone()

    counted.reset()
    backend.prefill_execution_stats(reset=True)
    backend._plan_key = None
    sequential = [
        model.forward_tokens(
            torch.tensor(row.token_ids, dtype=torch.long, device=cuda),
            torch.arange(row.chunk_start, row.seq_len, device=cuda),
            sequential_pool,
            list(row.page_table),
            seq_len=row.seq_len,
            write_from=row.write_from,
        )
        for row in rows
    ]
    sequential_plans = counted.plans
    sequential_runs = counted.runs
    sequential_stats = backend.prefill_execution_stats(reset=True)

    counted.reset()
    backend._plan_key = None
    batched = batch.split(model.forward_prefill_batch(batch, batched_pool))
    torch.cuda.synchronize()

    assert sequential_plans == len(rows)
    assert sequential_runs == len(rows) * model.config.num_hidden_layers
    assert sequential_stats["plans"] == sequential_plans
    assert sequential_stats["runs"] == sequential_runs
    assert counted.plans == 1
    assert counted.runs == model.config.num_hidden_layers
    assert backend.prefill_execution_stats() == {
        "type": "FlashInferBackend",
        "plans": 1,
        "runs": model.config.num_hidden_layers,
    }
    for actual, reference in zip(batched, sequential, strict=True):
        assert torch.allclose(actual, reference, atol=5e-2, rtol=5e-2)
    sequential_tokens = torch.stack(
        [model.logits(hidden[-1]).argmax() for hidden in sequential]
    )
    batched_tokens = torch.stack(
        [model.logits(hidden[-1]).argmax() for hidden in batched]
    )
    assert torch.equal(batched_tokens, sequential_tokens)
    assert torch.allclose(batched_pool.k, sequential_pool.k, atol=2e-2)
    assert torch.allclose(batched_pool.v, sequential_pool.v, atol=2e-2)
    assert torch.equal(batched_pool.k[:, 0], cached_k)
    assert torch.equal(batched_pool.v[:, 0], cached_v)
    assert torch.equal(batched_pool.k[:, 10], poison_k)
    assert torch.equal(batched_pool.v[:, 10], poison_v)

    # Warm both shapes before profiling so JIT compilation is outside the gate.
    for row in rows:
        model.forward_tokens(
            torch.tensor(row.token_ids, dtype=torch.long, device=cuda),
            torch.arange(row.chunk_start, row.seq_len, device=cuda),
            sequential_pool,
            list(row.page_table),
            seq_len=row.seq_len,
            write_from=row.write_from,
        )
    model.forward_prefill_batch(batch, batched_pool)
    torch.cuda.synchronize()

    with profile_scope(
        enabled=True,
        activities=("cpu", "cuda"),
        acc_events=True,
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as sequential_profile:
        for row in rows:
            model.forward_tokens(
                torch.tensor(row.token_ids, dtype=torch.long, device=cuda),
                torch.arange(row.chunk_start, row.seq_len, device=cuda),
                sequential_pool,
                list(row.page_table),
                seq_len=row.seq_len,
                write_from=row.write_from,
            )
        torch.cuda.synchronize()
    with profile_scope(
        enabled=True,
        activities=("cpu", "cuda"),
        acc_events=True,
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as batched_profile:
        model.forward_prefill_batch(batch, batched_pool)
        torch.cuda.synchronize()

    sequential_events = _cuda_event_count(sequential_profile)
    batched_events = _cuda_event_count(batched_profile)
    assert batched_events < sequential_events, (
        f"batched CUDA events did not fall: {sequential_events} -> "
        f"{batched_events}"
    )


def _tp2_prefill_cell(launcher, *, enabled: bool, prefix: str):
    from kairyu.engine.core.engine_core import token_ids
    from kairyu.engine.core.radix_kv import RadixKVCache
    from kairyu.engine.core.sampling_types import EngineSampling
    from kairyu.engine.core.scheduler import EngineRequest, Scheduler

    runner = launcher.runner
    runner.set_batched_prefill_enabled(enabled)
    runner.prefill_execution_stats(reset=True)
    cache = RadixKVCache(
        num_pages=TP_NUM_PAGES, page_size=TP_PAGE_SIZE
    )
    scheduler = Scheduler(
        cache,
        max_num_batched_tokens=sum(map(len, TP_PROMPTS)),
        max_num_seqs=len(TP_PROMPTS),
        page_size=TP_PAGE_SIZE,
    )
    request_ids = []
    for index, prompt in enumerate(TP_PROMPTS):
        request_id = f"{prefix}-{index}"
        request_ids.append(request_id)
        scheduler.add_request(
            EngineRequest(
                request_id=request_id,
                prompt_token_ids=prompt,
                max_new_tokens=1,
                ignore_eos=True,
                sampling=EngineSampling(),
            )
        )

    scheduled = scheduler.schedule().scheduled
    assert tuple(chunk.request_id for chunk in scheduled) == tuple(
        request_ids
    )
    assert tuple(chunk.num_tokens for chunk in scheduled) == tuple(
        map(len, TP_PROMPTS)
    )
    assert all(chunk.is_prefill for chunk in scheduled)
    sampled = runner.execute(scheduled, scheduler.states)
    scheduler.update(token_ids(sampled))
    outputs = tuple(
        scheduler.output_tokens(request_id)[0]
        for request_id in request_ids
    )
    for request_id in request_ids:
        runner.release(request_id)
    return outputs, runner.prefill_execution_stats()


def _assert_tp2_prefill_stats(
    rows, *, enabled: bool, layers: int
) -> None:
    assert len(rows) == TP_WORLD_SIZE
    for rank, row in enumerate(rows):
        assert row["rank"] == rank
        assert row["world_size"] == TP_WORLD_SIZE
        assert row["device"] == f"cuda:{rank}"
        stats = row["stats"]
        assert stats["enabled"] is enabled
        assert stats["capability_gap"] is None
        assert stats["rows"] == len(TP_PROMPTS)
        assert stats["model_calls"] == (
            1 if enabled else len(TP_PROMPTS)
        )
        assert stats["batched_groups"] == (1 if enabled else 0)
        assert stats["sequential_rows"] == (
            0 if enabled else len(TP_PROMPTS)
        )
        assert len(stats["backend"]) == 1
        backend = stats["backend"][0]
        assert backend["type"] == "FlashInferBackend"
        assert backend["plans"] == (
            1 if enabled else len(TP_PROMPTS)
        )
        assert backend["runs"] == (
            layers if enabled else len(TP_PROMPTS) * layers
        )


def test_tp2_real_nccl_prefill_mode_toggle_reduces_every_rank_launch_chain(
    tp2_model_dir, monkeypatch
):
    from kairyu.engine.core.worker import DistTPLauncher

    _require_two_gpus()
    monkeypatch.setenv("KAIRYU_ATTENTION_BACKEND", "flashinfer")
    launcher = DistTPLauncher(
        tp2_model_dir,
        tp=TP_WORLD_SIZE,
        num_pages=TP_NUM_PAGES,
        page_size=TP_PAGE_SIZE,
        vocab=[f"token-{index}" for index in range(TP_VOCAB_SIZE)],
    )
    try:
        topology = launcher.runner.sampling_ownership_metadata()
        assert [row["model_backend"] for row in topology] == [
            "nccl",
            "nccl",
        ]
        baseline, baseline_stats = _tp2_prefill_cell(
            launcher, enabled=False, prefix="sequential"
        )
        candidate, candidate_stats = _tp2_prefill_cell(
            launcher, enabled=True, prefix="batched"
        )

        assert candidate == baseline
        _assert_tp2_prefill_stats(
            baseline_stats, enabled=False, layers=2
        )
        _assert_tp2_prefill_stats(
            candidate_stats, enabled=True, layers=2
        )
    finally:
        launcher.shutdown()
        gc.collect()
        torch.cuda.empty_cache()
