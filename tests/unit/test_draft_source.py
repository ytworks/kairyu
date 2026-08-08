"""m17 D3 gates: DraftSource seam + model-draft e2e greedy equivalence."""

import torch

from kairyu.engine.core.draft import (
    EagleDraftAdapter,
    ModelDraftSource,
    MtpDraftAdapter,
    NGramDraftSource,
)
from kairyu.engine.core.spec_decode import propose_ngram


class TestNGramDraftSource:
    def test_matches_m8_propose_ngram(self):
        context = (1, 2, 3, 9, 1, 2, 3)
        source = NGramDraftSource()
        assert source.propose(context, 4) == propose_ngram(context, max_draft=4)


class _EchoDraftModel:
    """Deterministic toy: next token = (last token + 1) % 32."""

    def draft_next(self, token_ids: tuple[int, ...]) -> torch.Tensor:
        logits = torch.zeros(32)
        logits[(token_ids[-1] + 1) % 32] = 1.0
        return logits


class TestModelDraftSource:
    def test_autoregressive_rollout(self):
        source = ModelDraftSource(_EchoDraftModel())
        assert source.propose((5,), 3) == [6, 7, 8]
        assert source.propose((31,), 2) == [0, 1]
        assert source.propose((5,), 0) == []


def test_eagle_adapter_commits_only_authoritative_verification_rows():
    class _Target(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.embed_tokens = torch.nn.Embedding(64, 4)

    class _Head:
        def __init__(self):
            self.history_widths = []
            self.history_ptrs = []

        def fuse(self, aux):
            assert not torch.is_grad_enabled()
            return aux[:, :4]

        def rollout_cached(self, shifted, hidden, embed_fn, k):
            assert not torch.is_grad_enabled()
            self.history_widths.append((shifted.shape[0], hidden.shape[0]))
            self.history_ptrs.append(hidden.data_ptr())
            embed_fn(7)
            return list(range(10, 10 + k))

    head = _Head()
    source = EagleDraftAdapter(head, _Target())
    source.capture_target_rows(
        "r",
        (0, 1),
        torch.zeros(2, 4),
        aux_hidden=torch.arange(24, dtype=torch.float32).view(2, 12),
    )
    assert source.propose_for_request("r", (1, 2, 3), 2) == [10, 11]

    source.capture_target_rows(
        "r",
        (2, 3, 4),
        torch.zeros(3, 4),
        aux_hidden=torch.arange(36, dtype=torch.float32).view(3, 12),
        staged=True,
    )
    source.commit_verification("r", accepted_draft_tokens=1)
    assert source.propose_for_request("r", (1, 2, 3, 10, 19), 1) == [10]
    assert head.history_widths == [(2, 2), (4, 4)]
    assert head.history_ptrs[0] == head.history_ptrs[1]


def test_eagle_adapter_degrades_when_radix_hit_omits_hidden_prefix():
    class _Target(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.embed_tokens = torch.nn.Embedding(64, 4)

    class _Head:
        def fuse(self, aux):
            return aux[:, :4]

    source = EagleDraftAdapter(_Head(), _Target())
    source.capture_target_rows(
        "cached",
        (4,),
        torch.zeros(1, 4),
        aux_hidden=torch.zeros(1, 12),
    )
    assert source.propose_for_request("cached", (1, 2, 3, 4, 5, 6), 3) == []


def test_mtp_adapter_shifts_tokens_against_target_hidden_rows():
    class _Target(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.rotary_emb = object()

    class _Head:
        def rollout_cached(self, token_ids, hidden, rotary_emb, k):
            assert token_ids.tolist() == [2, 3]
            assert torch.equal(hidden, torch.tensor([[1.0], [2.0]]))
            assert rotary_emb is target.model.rotary_emb
            return [20, 21][:k]

    target = _Target()
    source = MtpDraftAdapter(_Head(), target)
    source.capture_target_rows(
        "r",
        (0, 1),
        torch.tensor([[1.0], [2.0]]),
    )
    assert source.propose_for_request("r", (1, 2, 3), 2) == [20, 21]


def test_spec_runner_with_model_draft_matches_plain_greedy():
    """A model draft through the FULL spec pipeline == plain greedy (m17 D3)."""
    import tests.unit.test_spec_runner as harness
    from kairyu.engine.core.engine_core import EngineCore
    from kairyu.engine.core.radix_kv import RadixKVCache
    from kairyu.engine.core.scheduler import EngineRequest, Scheduler
    from kairyu.engine.core.spec_runner import SpeculativeRunner
    from kairyu.engine.core.torch_runner import TinyAttentionLM, TorchPagedRunner

    torch.manual_seed(11)
    prompt = tuple(int(x) for x in torch.randint(0, 128, (18,)))
    plain = harness._plain_outputs(prompt, max_new=12, seed=3)

    model = TinyAttentionLM(seed=3)  # same weights as the plain reference

    class _SelfDraft:
        """Draft = the target model itself run densely (perfect drafts)."""

        def draft_next(self, token_ids: tuple[int, ...]) -> torch.Tensor:
            with torch.no_grad():
                keys, values = model.kv_for(torch.tensor(token_ids))
                return model.logits_for_last(int(token_ids[-1]), keys, values)

    cache = RadixKVCache(num_pages=128, page_size=harness.PAGE)
    scheduler = Scheduler(
        cache, max_num_batched_tokens=64, page_size=harness.PAGE, speculative_tokens=3
    )
    runner = SpeculativeRunner(
        TorchPagedRunner(model, num_pages=128, page_size=harness.PAGE),
        draft_source=ModelDraftSource(_SelfDraft()),
    )
    engine = EngineCore(scheduler, runner)
    engine.add_request(EngineRequest("a", prompt, max_new_tokens=12))
    outputs = engine.run_to_completion()["a"]
    assert list(outputs) == list(plain)
    assert runner.mean_accepted > 0.9  # perfect drafts: near-total acceptance
