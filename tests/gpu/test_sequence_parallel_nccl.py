"""Sequence parallelism over a real NCCL group (Megatron TP+SP)."""

import json
import time

import pytest
import torch
import torch.multiprocessing as mp

from tests.dist import dist_targets

pytestmark = pytest.mark.gpu

WORLD_SIZE = 2
SPAWN_TIMEOUT_S = 180

TINY = dict(
    vocab_size=256, hidden_size=256, intermediate_size=512, num_hidden_layers=2,
    num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
)


@pytest.fixture(scope="module")
def llama_dir(tmp_path_factory):
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(71)
    path = tmp_path_factory.mktemp("sp-llama")
    transformers.LlamaForCausalLM(transformers.LlamaConfig(**TINY)).to(
        torch.float32
    ).eval().save_pretrained(path, safe_serialization=True)
    return str(path)


def test_sequence_parallel_matches_plain_tp_on_nccl(tmp_path, llama_dir):
    if not torch.cuda.is_available():  # pragma: no cover - CPU box
        pytest.skip("SP NCCL gate needs 2 CUDA devices")
    if torch.cuda.device_count() < WORLD_SIZE:  # pragma: no cover - small box
        pytest.skip(f"need {WORLD_SIZE} CUDA devices, found {torch.cuda.device_count()}")

    out_dir = tmp_path / "results"
    out_dir.mkdir()
    context = mp.start_processes(
        dist_targets.sequence_parallel_nccl_parity,
        args=(WORLD_SIZE, str(tmp_path / "rdv-sp"), str(out_dir), llama_dir),
        nprocs=WORLD_SIZE,
        join=False,
        start_method="spawn",
    )
    deadline = time.monotonic() + SPAWN_TIMEOUT_S
    while not context.join(timeout=1):
        if time.monotonic() > deadline:  # pragma: no cover - hang guard
            for process in context.processes:
                process.terminate()
            pytest.fail(f"SP NCCL parity timed out after {SPAWN_TIMEOUT_S}s")

    for rank in range(WORLD_SIZE):
        result = json.loads((out_dir / f"rank{rank}.json").read_text())
        assert result["rows"] == 12, rank
        assert result["device"] == f"cuda:{rank}", rank
        # bf16 through a different reduction order; the arithmetic is the same
        assert result["max_error"] < 5e-2, f"rank {rank}: {result['max_error']}"
