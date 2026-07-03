"""m17 D3 gates: DraftSource seam + model-draft e2e greedy equivalence."""

import torch

from kairyu.engine.core.draft import ModelDraftSource, NGramDraftSource
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
