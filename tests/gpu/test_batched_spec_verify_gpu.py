"""Real CUDA/FlashInfer proof for grouped speculative target verification.

The eager gate compares one flattened multi-request target batch against the
rollback path's per-position model calls, including every sampled target and
the logical KV prefix.  The graph gate repeats that shape with new requests so
one real captured bucket must replay over new inputs without recapture.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.gpu

PAGE = 16
LAYERS = 2
POSITIONS = 5
TINY = {
    "architectures": ["Qwen3ForCausalLM"],
    "hidden_size": 256,
    "num_hidden_layers": LAYERS,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
    "head_dim": 64,
    "intermediate_size": 512,
    "vocab_size": 1024,
    "rms_norm_eps": 1e-6,
}
FIRST_CASE = (
    ("a", tuple(range(11, 28)), (401, 402, 403), 3),
    ("b", tuple(range(31, 40)), (501, 502), 2),
)
SECOND_CASE = (
    ("c", tuple(range(61, 73)), (601, 602, 603), 3),
    ("d", tuple(range(81, 98)), (701, 702), 2),
)


@pytest.fixture(scope="module")
def cuda() -> torch.device:
    if not torch.cuda.is_available():  # pragma: no cover - GPU suite only
        pytest.skip("CUDA/FlashInfer verification gate requires CUDA")
    return torch.device("cuda")


def _runner(cuda: torch.device, *, graph: bool):
    from kairyu.engine.core.attention.flashinfer_gpu import FlashInferBackend
    from kairyu.engine.core.kv_pool import PagedKVPool
    from kairyu.engine.core.model_runner import PagedModelRunner
    from kairyu.engine.core.radix_kv import RadixKVCache
    from kairyu.models.config import parse_model_config
    from kairyu.models.llama import DenseDecoder

    torch.manual_seed(215)
    backend = FlashInferBackend(device=cuda)
    model = (
        DenseDecoder(parse_model_config(TINY), attention_backend=backend)
        .to(device=cuda, dtype=torch.bfloat16)
        .eval()
    )
    cache = RadixKVCache(num_pages=32, page_size=PAGE)
    pool = PagedKVPool.for_cache(
        cache,
        model.config,
        dtype=torch.bfloat16,
        device=cuda,
    )
    graph_options = {}
    if graph:
        from kairyu.engine.core.cuda_graph_gpu import CudaGraphBackend

        graph_options = {
            "graph_backend": CudaGraphBackend(warmup_iters=1),
            "graph_max_batch": 8,
            "graph_max_pages": 2,
        }
    runner = PagedModelRunner(
        model,
        pool,
        cache=cache,
        **graph_options,
    )
    return runner, model, backend, cache, pool


def _state(
    request_id: str,
    prompt: tuple[int, ...],
    pages: tuple[int, ...],
    outputs: tuple[int, ...],
):
    return SimpleNamespace(
        request=SimpleNamespace(
            request_id=request_id,
            sampling_identity=request_id,
            prompt_token_ids=prompt,
            sampling=None,
            eos_token_id=None,
        ),
        allocation=SimpleNamespace(
            pages=pages,
            num_cached_tokens=len(prompt),
        ),
        decode_pages=[],
        computed_prompt=len(prompt),
        prefill_done=True,
        outputs=list(outputs),
        outputs_override=True,
        output_epoch=0,
    )


def _prepare_case(model, pool, cache, case, cuda: torch.device):
    from kairyu.engine.core.scheduler import ScheduledChunk

    chunks = []
    states = {}
    pages_by_request = {}
    for request_id, prompt, outputs, target_count in case:
        logical_length = len(prompt) + target_count
        page_count = -(-logical_length // PAGE)
        pages = tuple(
            cache.allocate_private_page() for _ in range(page_count)
        )
        model.forward_tokens(
            torch.tensor(prompt, dtype=torch.long, device=cuda),
            torch.arange(len(prompt), device=cuda),
            pool,
            list(pages),
            seq_len=len(prompt),
            write_from=0,
        )
        chunks.append(
            ScheduledChunk(
                request_id,
                num_tokens=target_count,
                is_prefill=False,
                position=1,
            )
        )
        states[request_id] = _state(
            request_id,
            prompt,
            pages,
            outputs,
        )
        pages_by_request[request_id] = pages
    return tuple(chunks), states, pages_by_request


def _execute_tokens(runner, chunks, states) -> dict[str, tuple[int, ...]]:
    sampled = runner.execute(chunks, states)
    return {
        chunk.request_id: tuple(
            record.token_id for record in sampled[chunk.request_id]
        )
        for chunk in chunks
    }


def _logical_kv(pool, pages: tuple[int, ...], logical_length: int):
    keys = torch.stack(
        [
            pool.k[:, pages[position // PAGE], position % PAGE]
            for position in range(logical_length)
        ],
        dim=1,
    )
    values = torch.stack(
        [
            pool.v[:, pages[position // PAGE], position % PAGE]
            for position in range(logical_length)
        ],
        dim=1,
    )
    return keys, values


def _assert_case_parity(
    candidate_pool,
    candidate_pages,
    reference_pool,
    reference_pages,
    case,
) -> None:
    for request_id, prompt, _outputs, target_count in case:
        logical_length = len(prompt) + target_count
        actual = _logical_kv(
            candidate_pool,
            candidate_pages[request_id],
            logical_length,
        )
        expected = _logical_kv(
            reference_pool,
            reference_pages[request_id],
            logical_length,
        )
        assert torch.allclose(actual[0], expected[0], atol=2e-2, rtol=2e-2)
        assert torch.allclose(actual[1], expected[1], atol=2e-2, rtol=2e-2)


def _backend_stats(stats: dict[str, object]) -> dict[str, object]:
    rows = stats["backend"]
    assert isinstance(rows, list) and len(rows) == 1
    row = rows[0]
    assert isinstance(row, dict)
    assert row["type"] == "FlashInferBackend"
    return row


def test_flattened_flashinfer_verification_matches_sequential_targets_and_kv(
    cuda,
):
    candidate, candidate_model, _backend, candidate_cache, candidate_pool = (
        _runner(cuda, graph=False)
    )
    reference, reference_model, _backend, reference_cache, reference_pool = (
        _runner(cuda, graph=False)
    )
    candidate_case = _prepare_case(
        candidate_model,
        candidate_pool,
        candidate_cache,
        FIRST_CASE,
        cuda,
    )
    reference_case = _prepare_case(
        reference_model,
        reference_pool,
        reference_cache,
        FIRST_CASE,
        cuda,
    )
    reference.set_batched_verification_enabled(False)
    candidate.verification_execution_stats(reset=True)
    reference.verification_execution_stats(reset=True)

    actual = _execute_tokens(candidate, candidate_case[0], candidate_case[1])
    expected = _execute_tokens(reference, reference_case[0], reference_case[1])
    torch.cuda.synchronize()

    assert actual == expected
    assert all(
        len(actual[request_id]) == target_count
        for request_id, _prompt, _outputs, target_count in FIRST_CASE
    )
    _assert_case_parity(
        candidate_pool,
        candidate_case[2],
        reference_pool,
        reference_case[2],
        FIRST_CASE,
    )

    batched = candidate.verification_execution_stats()
    sequential = reference.verification_execution_stats()
    assert batched["requests"] == len(FIRST_CASE)
    assert batched["positions"] == POSITIONS
    assert batched["model_calls"] == 1
    assert batched["batched_groups"] == 1
    assert batched["sequential_positions"] == 0
    assert batched["graph_groups"] == 0
    assert batched["graph"] is None
    assert _backend_stats(batched) == {
        "type": "FlashInferBackend",
        "plans": 1,
        "runs": LAYERS,
    }
    assert sequential["enabled"] is False
    assert sequential["requests"] == len(FIRST_CASE)
    assert sequential["positions"] == POSITIONS
    assert sequential["model_calls"] == POSITIONS
    assert sequential["batched_groups"] == 0
    assert sequential["sequential_positions"] == POSITIONS
    assert _backend_stats(sequential) == {
        "type": "FlashInferBackend",
        "plans": POSITIONS,
        "runs": POSITIONS * LAYERS,
    }


def test_flashinfer_verification_captures_once_then_replays_new_requests(
    cuda,
):
    candidate, candidate_model, _backend, candidate_cache, candidate_pool = (
        _runner(cuda, graph=True)
    )
    reference, reference_model, _backend, reference_cache, reference_pool = (
        _runner(cuda, graph=False)
    )
    reference.set_batched_verification_enabled(False)
    candidate_cases = [
        _prepare_case(
            candidate_model,
            candidate_pool,
            candidate_cache,
            case,
            cuda,
        )
        for case in (FIRST_CASE, SECOND_CASE)
    ]
    reference_cases = [
        _prepare_case(
            reference_model,
            reference_pool,
            reference_cache,
            case,
            cuda,
        )
        for case in (FIRST_CASE, SECOND_CASE)
    ]
    candidate.verification_execution_stats(reset=True)
    reference.verification_execution_stats(reset=True)

    first = _execute_tokens(
        candidate,
        candidate_cases[0][0],
        candidate_cases[0][1],
    )
    first_reference = _execute_tokens(
        reference,
        reference_cases[0][0],
        reference_cases[0][1],
    )
    second = _execute_tokens(
        candidate,
        candidate_cases[1][0],
        candidate_cases[1][1],
    )
    second_reference = _execute_tokens(
        reference,
        reference_cases[1][0],
        reference_cases[1][1],
    )
    torch.cuda.synchronize()

    assert first == first_reference
    assert second == second_reference
    for index, case in enumerate((FIRST_CASE, SECOND_CASE)):
        _assert_case_parity(
            candidate_pool,
            candidate_cases[index][2],
            reference_pool,
            reference_cases[index][2],
            case,
        )

    stats = candidate.verification_execution_stats()
    assert stats["requests"] == 2 * len(FIRST_CASE)
    assert stats["positions"] == 2 * POSITIONS
    assert stats["model_calls"] == 2
    assert stats["batched_groups"] == 2
    assert stats["sequential_positions"] == 0
    assert stats["graph_groups"] == 2
    assert stats["graph"] == {
        "graph_executions": 2,
        "eager_fallbacks": 0,
        "captured_buckets": (8,),
    }
    graph_backend = candidate._graph._backend
    assert graph_backend.captures == 1
    assert graph_backend.replays == 2
    assert _backend_stats(stats) == {
        "type": "FlashInferBackend",
        # One plan before capture, then one before each real replay.
        "plans": 3,
        # One warmup plus one capture invocation; replay is recorded kernels.
        "runs": 2 * LAYERS,
    }

    sequential = reference.verification_execution_stats()
    assert sequential["model_calls"] == 2 * POSITIONS
    assert _backend_stats(sequential) == {
        "type": "FlashInferBackend",
        "plans": 2 * POSITIONS,
        "runs": 2 * POSITIONS * LAYERS,
    }
