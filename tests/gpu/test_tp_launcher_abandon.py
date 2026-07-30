"""A failed NCCL start must not leave a live process group or a live rank.

Review [P1] on #129: the gloo test cannot reach this. gloo's
`destroy_process_group()` never blocks, so the ordering that matters — abort the
communicator BEFORE terminating the peers that would otherwise have to reach a
collective destroy — is only observable with NCCL.
"""

import time

import pytest
import torch

pytestmark = pytest.mark.gpu

WORLD_SIZE = 2
OVERCOMMIT_TP = 4  # the tp the symmetric-failure gate launches
# generous, but far below the 1800 s startup allowance a blocking destroy would
# otherwise sit in; the point is that this returns at all
DEADLINE_S = 120


def _require_devices(count: int) -> None:
    if not torch.cuda.is_available():  # pragma: no cover - CPU box
        pytest.skip("NCCL abandon gate needs CUDA")
    if torch.cuda.device_count() < count:  # pragma: no cover - small box
        pytest.skip(f"need {count} CUDA devices, found {torch.cuda.device_count()}")


@pytest.fixture(scope="module")
def indivisible_model(tmp_path_factory):
    """2 KV heads: tp=2 loads, tp=4 fails inside build_tp_runner — i.e. AFTER the
    spawn and AFTER the group is formed, which is the case that leaked."""
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(71)
    config = transformers.LlamaConfig(
        vocab_size=256, hidden_size=256, intermediate_size=512, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
    )
    model = transformers.LlamaForCausalLM(config)
    path = tmp_path_factory.mktemp("abandon-llama")
    model.to(torch.float32).eval().save_pretrained(path, safe_serialization=True)
    return str(path)


def test_failed_nccl_start_leaves_nothing_behind(indivisible_model):
    import torch.distributed as dist

    from kairyu.engine.core.worker import DistTPLauncher

    _require_devices(OVERCOMMIT_TP)
    assert not dist.is_initialized()

    started = time.monotonic()
    with pytest.raises(
        ValueError,
        match=r"num_key_value_heads=2 not divisible by tp=4",
    ):
        DistTPLauncher(
            indivisible_model,
            tp=4,
            num_pages=64,
            page_size=4,
            vocab=[f"t{i}" for i in range(256)],
        )
    elapsed = time.monotonic() - started

    # the original error came back, rather than the constructor blocking in a
    # collective destroy no peer can reach
    assert elapsed < DEADLINE_S, f"abandon took {elapsed:.1f}s"
    assert not dist.is_initialized(), "process group survived a failed NCCL start"


def test_the_process_can_still_launch_afterwards(indivisible_model):
    """The real consequence of the leak: the next launcher could not rendezvous."""
    import torch.distributed as dist

    from kairyu.engine.core.worker import DistTPLauncher

    _require_devices(OVERCOMMIT_TP)
    with pytest.raises(
        ValueError,
        match=r"num_key_value_heads=2 not divisible by tp=4",
    ):
        DistTPLauncher(
            indivisible_model, tp=4, num_pages=64, page_size=4,
            vocab=[f"t{i}" for i in range(256)],
        )

    launcher = DistTPLauncher(
        indivisible_model, tp=WORLD_SIZE, num_pages=64, page_size=4,
        vocab=[f"t{i}" for i in range(256)],
    )
    launcher.shutdown()
    assert not dist.is_initialized()


def test_a_rank_local_failure_still_returns_and_reaps(indivisible_model, monkeypatch):
    """The case the symmetric gate cannot reach (review [P1] on #129).

    `num_key_value_heads` fails identically on every rank, so the peers are already gone
    when rank 0 tears down. Patching `build_tp_runner` in THIS process only is
    rank-local by construction — the spawned children import the module fresh —
    so the workers survive, reach their handshake broadcast, and sit there while
    rank 0 unwinds. If teardown waits on a collective, this never returns.
    """
    import torch.distributed as dist

    from kairyu.engine.core import worker as worker_module
    from kairyu.engine.core.worker import DistTPLauncher

    _require_devices(WORLD_SIZE)
    assert not dist.is_initialized()

    def _rank0_only_failure(*args, **kwargs):
        raise RuntimeError("injected rank-local failure")

    monkeypatch.setattr(worker_module, "build_tp_runner", _rank0_only_failure)

    started = time.monotonic()
    with pytest.raises(RuntimeError, match="injected rank-local failure"):
        DistTPLauncher(
            indivisible_model,
            tp=WORLD_SIZE,
            num_pages=64,
            page_size=4,
            vocab=[f"t{i}" for i in range(256)],
        )
    elapsed = time.monotonic() - started

    assert elapsed < DEADLINE_S, f"rank-local abandon took {elapsed:.1f}s"
    assert not dist.is_initialized(), "process group survived a rank-local failure"

    # every child reaped, and the process can still form a group
    monkeypatch.undo()
    launcher = DistTPLauncher(
        indivisible_model, tp=WORLD_SIZE, num_pages=64, page_size=4,
        vocab=[f"t{i}" for i in range(256)],
    )
    launcher.shutdown()
    assert not dist.is_initialized()
