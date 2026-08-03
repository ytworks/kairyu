"""FlashInfer honours the tensor decode contract too (m13 seam, m17 D1).

#137 added `attend_decode` for the torch backend so decode could be captured.
FlashInfer is the kernel a GPU deployment actually selects, so the contract has
to hold there or the capturable path is a fallback nobody runs.

FlashInfer splits the work in two: `plan()` builds the split-KV schedule on the
CPU and "cannot be used in Cuda Graph" (its own docstring), while `run()` reads
the plan out of device buffers. So the gates below are the two halves — eager
parity for `plan_decode` + `attend_decode` together, and a REAL captured graph
whose replay picks up an in-place page-table change once the step is re-planned.
"""

import pytest
import torch

pytestmark = pytest.mark.gpu

PAGE_SIZE = 16
HEAD_DIM = 64
KV_HEADS = 2
QO_HEADS = 4


def _require_cuda() -> None:
    if not torch.cuda.is_available():  # pragma: no cover - CPU box
        pytest.skip("FlashInfer gates need CUDA")


def _pool(*, seed: int = 3):
    from kairyu.engine.core.kv_pool import PagedKVPool

    torch.manual_seed(seed)
    pool = PagedKVPool(
        num_layers=1, num_pages=8, page_size=PAGE_SIZE, num_kv_heads=KV_HEADS,
        head_dim=HEAD_DIM, dtype=torch.bfloat16, device="cuda:0",
    )
    with torch.no_grad():
        pool.k.copy_(torch.randn(pool.k.shape, device="cuda:0").to(torch.bfloat16) * 0.1)
        pool.v.copy_(torch.randn(pool.v.shape, device="cuda:0").to(torch.bfloat16) * 0.1)
    return pool


@pytest.mark.parametrize(
    ("tables", "seq_lens"),
    [
        ([[0, 1], [2, 3]], [20, 30]),
        ([[0, 1], [2, 3]], [16, 32]),  # exact page multiples
        ([[4, 5]], [17]),  # single row, one token into the second page
    ],
    ids=["ragged", "aligned", "single"],
)
def test_flashinfer_matches_torch_on_the_tensor_contract(tables, seq_lens):
    from kairyu.engine.core.attention.flashinfer_gpu import FlashInferBackend
    from kairyu.engine.core.attention.torch_backend import TorchAttentionBackend

    _require_cuda()
    pool = _pool()
    rows = len(tables)
    query = torch.randn(rows, QO_HEADS, HEAD_DIM, device="cuda:0", dtype=torch.bfloat16)
    page_tables = torch.tensor(tables, dtype=torch.int32, device="cuda:0")
    lengths = torch.tensor(seq_lens, dtype=torch.int32, device="cuda:0")

    reference = TorchAttentionBackend().attend_decode(query, pool, 0, page_tables, lengths)
    actual = FlashInferBackend().attend_decode(query, pool, 0, page_tables, lengths)

    assert actual.shape == reference.shape
    assert torch.allclose(actual.float(), reference.float(), atol=5e-2)


def test_the_paged_arrays_come_from_the_tensors_not_a_python_list():
    """A page table mutated IN PLACE must change the answer — the property the
    list-based `attend_batched` cannot have."""
    from kairyu.engine.core.attention.flashinfer_gpu import FlashInferBackend

    _require_cuda()
    pool = _pool()
    backend = FlashInferBackend()
    query = torch.randn(1, QO_HEADS, HEAD_DIM, device="cuda:0", dtype=torch.bfloat16)
    tables = torch.tensor([[0, 1]], dtype=torch.int32, device="cuda:0")
    lengths = torch.tensor([20], dtype=torch.int32, device="cuda:0")

    first = backend.attend_decode(query, pool, 0, tables, lengths)
    tables.copy_(torch.tensor([[4, 5]], dtype=torch.int32, device="cuda:0"))
    second = backend.attend_decode(query, pool, 0, tables, lengths)

    assert not torch.allclose(first.float(), second.float())


def test_a_captured_graph_replays_against_the_current_pages():
    """The gate the eager tests structurally cannot give: a REAL CUDA graph.

    Capture fails outright on any `.tolist()`/`.cpu()` inside the region
    ("Cannot copy between CPU and CUDA tensors during CUDA graph capture"). It
    succeeding is half the property; the other half is that after the static
    page-table/seq-len buffers are rewritten in place and the step is
    re-planned on the host, `replay()` attends over the NEW pages — not the
    ones that happened to be there at capture time.
    """
    from kairyu.engine.core.attention.flashinfer_gpu import FlashInferBackend
    from kairyu.engine.core.attention.torch_backend import TorchAttentionBackend
    from kairyu.engine.core.cuda_graph_gpu import CudaGraphBackend

    _require_cuda()
    pool = _pool()
    backend = FlashInferBackend()
    query = torch.randn(2, QO_HEADS, HEAD_DIM, device="cuda:0", dtype=torch.bfloat16)
    # STATIC buffers: the graph is captured against these exact tensors
    tables = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32, device="cuda:0")
    lengths = torch.tensor([20, 30], dtype=torch.int32, device="cuda:0")

    def decode(_batch=None):
        return backend.attend_decode(query, pool, 0, tables, lengths)

    backend.plan_decode(
        pool, tables, lengths, num_qo_heads=QO_HEADS, q_dtype=query.dtype
    )
    replayable = CudaGraphBackend(warmup_iters=2).capture(decode, None)

    captured = replayable.replay().clone()
    torch.cuda.synchronize()
    assert torch.allclose(
        captured.float(),
        TorchAttentionBackend().attend_decode(query, pool, 0, tables, lengths).float(),
        atol=5e-2,
    )

    tables.copy_(torch.tensor([[4, 5], [6, 7]], dtype=torch.int32, device="cuda:0"))
    lengths.copy_(torch.tensor([16, 17], dtype=torch.int32, device="cuda:0"))
    backend.plan_decode(  # replay-only fast host half, OUTSIDE the graph
        pool,
        tables,
        lengths,
        num_qo_heads=QO_HEADS,
        q_dtype=query.dtype,
        replay=True,
        host_seq_lens=(16, 17),
    )
    replayed = replayable.replay()
    torch.cuda.synchronize()

    expected = TorchAttentionBackend().attend_decode(query, pool, 0, tables, lengths)
    assert torch.allclose(replayed.float(), expected.float(), atol=5e-2)
    assert not torch.allclose(replayed.float(), captured.float())


def test_fast_replay_plan_has_no_stream_sync_or_nonzero_and_keeps_parity():
    """Steady replay planning is device-only apart from CPU scheduler inputs."""

    from torch.profiler import ProfilerActivity, profile

    from kairyu.engine.core.attention.flashinfer_gpu import FlashInferBackend
    from kairyu.engine.core.attention.torch_backend import TorchAttentionBackend

    _require_cuda()
    pool = _pool()
    backend = FlashInferBackend()
    query = torch.randn(3, QO_HEADS, HEAD_DIM, device="cuda:0", dtype=torch.bfloat16)
    tables = torch.tensor(
        [[0, 1, 7], [2, 3, 7], [4, 5, 6]],
        dtype=torch.int32,
        device="cuda:0",
    )
    lengths = torch.tensor([20, 30, 33], dtype=torch.int32, device="cuda:0")

    # Stock initialization is required once for the wrapper/module.  Warm the
    # Triton and fast-plan modules before measuring the steady replay path.
    backend.plan_decode(
        pool, tables, lengths, num_qo_heads=QO_HEADS, q_dtype=query.dtype
    )
    backend.plan_decode(
        pool,
        tables,
        lengths,
        num_qo_heads=QO_HEADS,
        q_dtype=query.dtype,
        replay=True,
        host_seq_lens=(20, 30, 33),
    )
    torch.cuda.synchronize()

    tables.copy_(
        torch.tensor(
            [[6, 5, 7], [4, 3, 7], [2, 1, 0]],
            dtype=torch.int32,
            device="cuda:0",
        )
    )
    lengths.copy_(torch.tensor([16, 17, 47], dtype=torch.int32, device="cuda:0"))
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        acc_events=True,
    ) as prof:
        backend.plan_decode(
            pool,
            tables,
            lengths,
            num_qo_heads=QO_HEADS,
            q_dtype=query.dtype,
            replay=True,
            host_seq_lens=(16, 17, 47),
        )
        actual = backend.attend_decode(query, pool, 0, tables, lengths)
        torch.cuda.synchronize()

    expected = TorchAttentionBackend().attend_decode(
        query, pool, 0, tables, lengths
    )
    keys = {event.key for event in prof.key_averages()}
    assert not any("cudaStreamSynchronize" in key for key in keys), sorted(keys)
    assert "aten::nonzero" not in keys, sorted(keys)
    assert torch.allclose(actual.float(), expected.float(), atol=5e-2)
    stats = backend.decode_execution_stats()
    assert stats["fast_replay_plans"] == 2
    assert stats["stock_replay_fallbacks"] == 0


def test_planning_inside_a_capture_fails_instead_of_corrupting_the_graph():
    """FlashInfer's plan cannot be captured; the adapter says so itself rather
    than letting the capture die inside the kernel library."""
    from kairyu.engine.core.attention.flashinfer_gpu import FlashInferBackend
    from kairyu.engine.core.cuda_graph_gpu import CudaGraphBackend

    _require_cuda()
    pool = _pool()
    backend = FlashInferBackend()
    query = torch.randn(1, QO_HEADS, HEAD_DIM, device="cuda:0", dtype=torch.bfloat16)
    tables = torch.tensor([[0, 1]], dtype=torch.int32, device="cuda:0")
    lengths = torch.tensor([20], dtype=torch.int32, device="cuda:0")

    def decode(_batch=None):
        return backend.attend_decode(query, pool, 0, tables, lengths)

    decode()  # warm the kernels up eagerly (this plans)
    backend._decode_tensor_key = None  # ...then let the step's plan go missing

    with pytest.raises(RuntimeError, match="plan_decode"):
        CudaGraphBackend(warmup_iters=0).capture(decode, None)


# --- production wiring: FlashInfer under GraphStepExecutor (#141 review [P1]) --
#
# Everything above drives `plan_decode` by hand. That is exactly what the review
# called out: no production caller existed, so these gates passed while the real
# path had none. These run the plan through the shipping wiring instead —
# PagedModelRunner -> GraphStepExecutor -> DenseDecoder.plan_decode_tensors ->
# FlashInferBackend.plan_decode — so a missing hook fails HERE, with the "no
# live plan" RuntimeError, during the first capture.

TINY = dict(
    vocab_size=256, hidden_size=256, intermediate_size=512, num_hidden_layers=2,
    num_attention_heads=QO_HEADS, num_key_value_heads=KV_HEADS,
    max_position_embeddings=512,
)


@pytest.fixture(scope="module")
def llama_dir(tmp_path_factory):
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(71)
    path = tmp_path_factory.mktemp("flashinfer-graph")
    transformers.LlamaForCausalLM(transformers.LlamaConfig(**TINY)).to(
        torch.float32
    ).eval().save_pretrained(path, safe_serialization=True)
    return str(path)


def _flashinfer_graph_runner(
    llama_dir, *, max_batch=4, max_pages=4, num_pages=64, warmup_iters=3
):
    """A real runner over the real kernel and a real CUDA graph backend."""
    from kairyu.engine.core.attention.flashinfer_gpu import FlashInferBackend
    from kairyu.engine.core.cuda_graph_gpu import CudaGraphBackend
    from kairyu.engine.core.kv_pool import PagedKVPool
    from kairyu.engine.core.model_runner import PagedModelRunner
    from kairyu.engine.core.radix_kv import RadixKVCache
    from kairyu.models.loader import load_model

    # ONE backend instance for every layer (m13) — the plan is shared, which is
    # why the model-level hook plans per instance and not per layer.
    backend = FlashInferBackend(device="cuda:0")
    model, config, _ = load_model(
        llama_dir, dtype=torch.bfloat16, attention_backend=backend,
        target_device="cuda:0",
    )
    cache = RadixKVCache(num_pages=num_pages, page_size=PAGE_SIZE)
    pool = PagedKVPool.for_cache(cache, config, dtype=torch.bfloat16, device="cuda:0")
    runner = PagedModelRunner(
        model.to("cuda:0"), pool, cache=cache,
        graph_backend=CudaGraphBackend(warmup_iters=warmup_iters),
        graph_max_batch=max_batch, graph_max_pages=max_pages,
    )
    return runner, backend, cache, pool


def test_the_runner_accepts_flashinfer_for_graph_decode(llama_dir):
    """The other side of #138's gate: FlashInfer now DECLARES capture support
    honestly, so the runner must build the graph path for it."""
    _require_cuda()
    runner, _backend, _cache, _pool = _flashinfer_graph_runner(llama_dir)
    assert runner._graph is not None


def test_flashinfer_decode_captures_and_replays_through_the_runner(llama_dir):
    """The production integration gate: real capture, real replays, and the
    pages CHANGE underneath between steps.

    Each step is compared against the same model's eager tensor decode over the
    same KV — same kernels, same inputs, the only difference being that one went
    through a recorded graph. Without the step-boundary hook the first capture
    raises "no live plan"; with the plan taken before `_copy_in` instead of
    after, every replay would attend over the previous step's pages and the
    comparison below fails.
    """
    from kairyu.engine.core.step_executor import build_decode_batch

    _require_cuda()
    runner, backend, cache, _pool = _flashinfer_graph_runner(llama_dir)
    graph = runner._graph
    scratch = runner._graph_scratch_page

    # three sequences, each owning two private pages, decoding several steps
    rows = [[cache.allocate_private_page() for _ in range(2)] for _ in range(3)]
    tokens = [11, 22, 33]

    for step in range(4):
        # lengths GROW every step, and after step 1 the sequences also swap to
        # freshly allocated pages: a stale plan cannot survive either change
        seq_lens = [PAGE_SIZE + 1 + step, PAGE_SIZE + 4 + step, 2 * PAGE_SIZE - step]
        positions = [length - 1 for length in seq_lens]
        if step == 2:
            rows = [[cache.allocate_private_page() for _ in range(2)] for _ in range(3)]

        graphed = runner._graph_logits(tokens, positions, rows, seq_lens).clone()
        torch.cuda.synchronize()
        # batch 3 -> bucket 4, page tables of width 2 <= max_pages: the graph
        # path, never the eager fallback
        assert list(graph._captured) == [4], f"step {step} did not use the graph"

        # the eager answer over the SAME padded buffers: re-plan, then run the
        # captured region directly instead of replaying it. Re-writing the same
        # K/V to the same slots is idempotent, so the pool is unchanged.
        batch = build_decode_batch(
            token_ids=tokens, positions=positions, page_lists=rows,
            seq_lens=seq_lens, max_pages=2, scratch_page=scratch, device="cuda:0",
        )
        runner._plan_graph_decode(batch)
        expected = runner._graph_decode(batch)
        torch.cuda.synchronize()

        assert torch.allclose(
            graphed.float(), expected.float(), atol=5e-2
        ), f"step {step}: the replay disagrees with the eager decode"

    # captured ONCE and replayed for every later step (not recaptured per step)
    assert list(graph._captured) == [4]
    assert backend._decode_tensor_wrapper is not None
    stats = backend.decode_execution_stats()
    # The eager oracle above switches to a different wrapper shape between
    # graph steps.  Returning to the already-stock-initialized graph wrapper
    # must still use fast replay rather than treating the global "last wrapper"
    # pointer as its initialization state.
    assert stats["fast_replay_plans"] == 4
    assert stats["stock_replay_fallbacks"] == 0


def test_the_capture_itself_is_planned_by_the_hook_not_by_warmup(llama_dir):
    """With no warmup iterations, NOTHING plans lazily before the capture — the
    step-boundary hook is the only thing standing between the first capture and
    FlashInfer's "no live plan" RuntimeError.

    Warmup normally hides this (it runs eagerly, so `attend_decode` plans for
    itself), which is precisely why the hook must not depend on it.
    """
    _require_cuda()
    runner, _backend, cache, _pool = _flashinfer_graph_runner(
        llama_dir, warmup_iters=0
    )
    pages = [[cache.allocate_private_page()] for _ in range(2)]

    runner._graph_logits([5, 6], [3, 3], pages, [4, 4])  # must not raise
    torch.cuda.synchronize()

    assert list(runner._graph._captured) == [2]


def test_the_replay_follows_the_pages_not_the_capture(llama_dir):
    """A blunt statement of the same property: run one step, then move every
    sequence onto different pages and run again. If the plan were taken once at
    capture, or before `_copy_in`, the second answer would repeat the first."""
    _require_cuda()
    runner, backend, cache, pool = _flashinfer_graph_runner(llama_dir)

    first_pages = [[cache.allocate_private_page()] for _ in range(2)]
    second_pages = [[cache.allocate_private_page()] for _ in range(2)]
    # give the two page sets clearly different KV so the answers must differ
    with torch.no_grad():
        for (page,), value in zip(second_pages, (0.5, -0.5), strict=True):
            pool.k[:, page].fill_(value)
            pool.v[:, page].fill_(value)

    tokens, positions, seq_lens = [7, 9], [5, 5], [6, 6]
    first = runner._graph_logits(tokens, positions, first_pages, seq_lens).clone()
    second = runner._graph_logits(tokens, positions, second_pages, seq_lens)
    torch.cuda.synchronize()

    assert not torch.allclose(first.float(), second.float(), atol=1e-3)
    stats = backend.decode_execution_stats()
    assert stats["fast_replay_plans"] == 2
    assert stats["stock_replay_fallbacks"] == 0
