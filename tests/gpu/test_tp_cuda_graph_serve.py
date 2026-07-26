"""Production TP serving captures model/NCCL decode on every rank."""

import pytest
import torch

from kairyu import SamplingParams

transformers = pytest.importorskip("transformers")

pytestmark = pytest.mark.gpu

TINY = dict(
    vocab_size=256,
    hidden_size=256,
    intermediate_size=512,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    max_position_embeddings=512,
)


class _Tokenizer:
    eos_token_id = None

    def encode(self, text: str) -> tuple[int, ...]:
        return tuple(range(9))

    def decode(self, token_ids) -> str:
        return " ".join(f"t{token_id}" for token_id in token_ids)

    def vocab(self) -> list[str]:
        return [f"t{index}" for index in range(TINY["vocab_size"])]


@pytest.fixture(scope="module")
def llama_dir(tmp_path_factory):
    torch.manual_seed(71)
    path = tmp_path_factory.mktemp("tp-graph-serve")
    transformers.LlamaForCausalLM(transformers.LlamaConfig(**TINY)).to(
        torch.float32
    ).eval().save_pretrained(path, safe_serialization=True)
    return str(path)


def _generate(model_dir: str, decode_mode: str):
    from kairyu.engine.kairyu_backend import build_engine_loop

    loop, cache, _scheduler = build_engine_loop(
        model_path=model_dir,
        tokenizer=_Tokenizer(),
        tensor_parallel_size=2,
        num_pages=64,
        page_size=16,
        max_num_batched_tokens=64,
        decode_mode=decode_mode,
        cuda_graph_max_batch=4,
        cuda_graph_max_pages=4,
        cuda_graph_warmup_iters=1,
    )
    try:
        loop.submit(
            f"tp-{decode_mode}",
            "prompt",
            SamplingParams(max_tokens=6, temperature=0.0, ignore_eos=True),
        )
        final = None
        for _ in range(32):
            for _request_id, update in loop.step():
                final = update
            if final is not None and final.finished:
                break
        assert final is not None and final.finished
        rank0_local = loop.tp_launcher.runner._local
        stats = None
        if rank0_local._graph is not None:
            backend = rank0_local._graph._backend
            stats = (backend.captures, backend.replays)
        return final.outputs, stats, cache.num_free_pages
    finally:
        loop.tp_launcher.shutdown()


def test_production_tp_cuda_graph_matches_eager(llama_dir, monkeypatch):
    if not torch.cuda.is_available():  # pragma: no cover - CPU box
        pytest.skip("TP CUDA graph serving needs 2 CUDA devices")
    if torch.cuda.device_count() < 2:  # pragma: no cover - small box
        pytest.skip(f"need 2 CUDA devices, found {torch.cuda.device_count()}")
    monkeypatch.setenv("KAIRYU_ATTENTION_BACKEND", "torch")

    eager, eager_stats, eager_free = _generate(llama_dir, "eager")
    graphed, graph_stats, graph_free = _generate(llama_dir, "cuda_graph")

    assert graphed == eager
    assert eager_stats is None
    assert graph_stats is not None
    captures, replays = graph_stats
    assert captures > 0
    assert replays > captures
    assert graph_free == eager_free - 1
