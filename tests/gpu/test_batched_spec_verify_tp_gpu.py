"""Real-NCCL TP=2 gate for batched speculative target verification (#215)."""

from __future__ import annotations

import gc

import pytest
import torch

transformers = pytest.importorskip("transformers")

pytestmark = pytest.mark.gpu

WORLD_SIZE = 2
NUM_PAGES = 96
PAGE_SIZE = 8
VOCAB_SIZE = 256
VOCAB = [f"token-{index}" for index in range(VOCAB_SIZE)]
TINY = dict(
    vocab_size=VOCAB_SIZE,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    max_position_embeddings=512,
)

ACCEPT_PROMPT = (1, 2, 3, 4, 5, 6, 7)
REJECT_PROMPT = (17, 18, 19, 20, 21, 22, 23, 24, 25)


def _require_two_gpus() -> None:
    if not torch.cuda.is_available():  # pragma: no cover - CPU host
        pytest.skip("TP speculative verification needs CUDA")
    if torch.cuda.device_count() < WORLD_SIZE:  # pragma: no cover - small host
        pytest.skip(
            f"TP speculative verification needs {WORLD_SIZE} GPUs; "
            f"found {torch.cuda.device_count()}"
        )


@pytest.fixture(scope="module")
def llama_dir(tmp_path_factory: pytest.TempPathFactory) -> str:
    torch.manual_seed(215)
    model = transformers.LlamaForCausalLM(transformers.LlamaConfig(**TINY))
    path = tmp_path_factory.mktemp("tp-batched-spec-verify")
    model.to(torch.float32).eval().save_pretrained(path, safe_serialization=True)
    return str(path)


class _ControlledDraft:
    """Accept one request and force the other through correction/rollback."""

    def __init__(
        self,
        *,
        accepted: tuple[tuple[int, ...], tuple[int, ...]],
        rejected: tuple[tuple[int, ...], tuple[int, ...]],
    ) -> None:
        self._accepted_prompt, self._accepted_reference = accepted
        self._rejected_prompt, self._rejected_reference = rejected

    def propose(self, context, max_draft: int) -> list[int]:
        context = tuple(context)
        if context[: len(self._accepted_prompt)] == self._accepted_prompt:
            produced = len(context) - len(self._accepted_prompt)
            return list(
                self._accepted_reference[produced : produced + max_draft]
            )
        if context[: len(self._rejected_prompt)] == self._rejected_prompt:
            produced = len(context) - len(self._rejected_prompt)
            wrong = (self._rejected_reference[produced] + 1) % VOCAB_SIZE
            return [wrong] * min(1, max_draft)
        raise AssertionError(f"unexpected draft context: {context!r}")


def _plain_references(runner, cache) -> dict[str, tuple[int, ...]]:
    from kairyu.engine.core.engine_core import EngineCore
    from kairyu.engine.core.sampling_types import EngineSampling
    from kairyu.engine.core.scheduler import EngineRequest, Scheduler

    scheduler = Scheduler(
        cache,
        max_num_batched_tokens=64,
        max_num_seqs=4,
        page_size=PAGE_SIZE,
    )
    engine = EngineCore(scheduler, runner)
    engine.add_request(
        EngineRequest(
            "plain-accept",
            ACCEPT_PROMPT,
            max_new_tokens=4,
            ignore_eos=True,
            sampling=EngineSampling(),
        )
    )
    engine.add_request(
        EngineRequest(
            "plain-reject",
            REJECT_PROMPT,
            max_new_tokens=3,
            ignore_eos=True,
            sampling=EngineSampling(),
        )
    )
    outputs = engine.run_to_completion()
    runner.release("plain-accept")
    runner.release("plain-reject")
    return outputs


def test_tp2_batches_variable_width_spec_verify_and_preserves_rollback_parity(
    llama_dir: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kairyu.engine.core.engine_core import token_ids
    from kairyu.engine.core.radix_kv import RadixKVCache
    from kairyu.engine.core.sampling_types import EngineSampling
    from kairyu.engine.core.scheduler import EngineRequest, Scheduler
    from kairyu.engine.core.spec_runner import SpeculativeRunner
    from kairyu.engine.core.worker import DistTPLauncher

    _require_two_gpus()
    monkeypatch.setenv("KAIRYU_ATTENTION_BACKEND", "torch")

    cache = RadixKVCache(num_pages=NUM_PAGES, page_size=PAGE_SIZE)
    launcher = DistTPLauncher(
        llama_dir,
        tp=WORLD_SIZE,
        num_pages=NUM_PAGES,
        page_size=PAGE_SIZE,
        vocab=VOCAB,
    )
    try:
        topology = launcher.runner.sampling_ownership_metadata()
        assert [row["rank"] for row in topology] == [0, 1]
        assert [row["device"] for row in topology] == ["cuda:0", "cuda:1"]
        assert [row["model_backend"] for row in topology] == ["nccl", "nccl"]

        reference = _plain_references(launcher.runner, cache)
        accept_reference = reference["plain-accept"]
        reject_reference = reference["plain-reject"]

        scheduler = Scheduler(
            cache,
            max_num_batched_tokens=64,
            max_num_seqs=4,
            page_size=PAGE_SIZE,
            speculative_tokens=3,
        )
        scheduler.add_request(
            EngineRequest(
                "spec-accept",
                ACCEPT_PROMPT,
                max_new_tokens=4,
                ignore_eos=True,
                sampling=EngineSampling(),
            )
        )
        scheduler.add_request(
            EngineRequest(
                "spec-reject",
                REJECT_PROMPT,
                max_new_tokens=3,
                ignore_eos=True,
                sampling=EngineSampling(),
            )
        )
        runner = SpeculativeRunner(
            launcher.runner,
            draft_source=_ControlledDraft(
                accepted=(ACCEPT_PROMPT, accept_reference),
                rejected=(REJECT_PROMPT, reject_reference),
            ),
        )

        prefill = scheduler.schedule().scheduled
        assert [(chunk.request_id, chunk.is_prefill) for chunk in prefill] == [
            ("spec-accept", True),
            ("spec-reject", True),
        ]
        prefill_sampled = runner.execute(prefill, scheduler.states)
        assert scheduler.update(token_ids(prefill_sampled)) == ()
        assert scheduler.output_tokens("spec-accept") == accept_reference[:1]
        assert scheduler.output_tokens("spec-reject") == reject_reference[:1]

        launcher.runner.verification_execution_stats(reset=True)
        verify = scheduler.schedule().scheduled
        assert [
            (chunk.request_id, chunk.num_tokens, chunk.position)
            for chunk in verify
        ] == [
            ("spec-accept", 3, 1),
            ("spec-reject", 2, 1),
        ]

        local = launcher.runner._local
        original_packet = local.make_sampling_token_packet
        packets: list[tuple[int, ...]] = []

        def capture_packet(scheduled, states, sampled=None):
            packet = original_packet(scheduled, states, sampled=sampled)
            packets.append(tuple(packet.detach().cpu().tolist()))
            return packet

        monkeypatch.setattr(local, "make_sampling_token_packet", capture_packet)
        verified = runner.execute(verify, scheduler.states)
        monkeypatch.setattr(
            local,
            "make_sampling_token_packet",
            original_packet,
        )

        accepted = tuple(
            token.token_id for token in verified["spec-accept"]
        )
        corrected = tuple(
            token.token_id for token in verified["spec-reject"]
        )
        assert accepted == accept_reference[1:4]
        assert corrected == reject_reference[1:2]
        assert runner.draft_proposed == 3
        assert runner.draft_accepted == 2

        assert len(packets) == 1
        packet = packets[0]
        assert len(packet) == 5
        assert all(0 <= token_id < VOCAB_SIZE for token_id in packet)
        assert packet[:3] == accept_reference[1:4]
        assert packet[3] == reject_reference[1]

        rows = launcher.runner.verification_execution_stats()
        assert [row["rank"] for row in rows] == [0, 1]
        assert [row["device"] for row in rows] == ["cuda:0", "cuda:1"]
        for row in rows:
            assert row["world_size"] == WORLD_SIZE
            stats = row["stats"]
            assert stats["enabled"] is True
            assert stats["capability_gap"] is None
            assert stats["requests"] == 2
            assert stats["positions"] == 5
            assert stats["model_calls"] == 1
            assert stats["batched_groups"] == 1
            assert stats["sequential_positions"] == 0
            assert stats["graph_groups"] == 0

        assert scheduler.update(token_ids(verified)) == ("spec-accept",)
        assert scheduler.output_tokens("spec-accept") == accept_reference
        assert scheduler.output_tokens("spec-reject") == reject_reference[:2]

        rollback = scheduler.schedule().scheduled
        assert [
            (chunk.request_id, chunk.num_tokens, chunk.position)
            for chunk in rollback
        ] == [("spec-reject", 1, 2)]
        final = runner.execute(rollback, scheduler.states)
        assert scheduler.update(token_ids(final)) == ("spec-reject",)
        assert scheduler.output_tokens("spec-reject") == reject_reference
        assert not scheduler.has_unfinished()

        runner.release("spec-accept")
        runner.release("spec-reject")
    finally:
        launcher.shutdown()
        gc.collect()
        torch.cuda.empty_cache()
