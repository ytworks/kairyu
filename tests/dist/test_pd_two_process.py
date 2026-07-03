"""m18 D5 flagship: disaggregated P-D across two REAL processes over TCP —
outputs match a single engine AND the transferred KV bytes match."""

import pytest
import torch

from tests.dist import dist_targets
from tests.dist.test_distributed import TINY_LLAMA, _single_process_greedy

transformers = pytest.importorskip("transformers")

pytestmark = pytest.mark.dist


@pytest.fixture(scope="module")
def llama_dir(tmp_path_factory):
    torch.manual_seed(97)
    model = transformers.LlamaForCausalLM(transformers.LlamaConfig(**TINY_LLAMA))
    path = tmp_path_factory.mktemp("pd-llama")
    model.to(torch.float32).eval().save_pretrained(path, safe_serialization=True)
    return str(path)


def test_pd_two_process_matches_single_engine(spawn2, llama_dir):
    torch.manual_seed(101)
    prompt = torch.randint(0, 256, (11,)).tolist()  # 2 full pages + tail
    reference = _single_process_greedy(llama_dir, prompt, max_new=10)

    results = spawn2(dist_targets.pd_two_process, llama_dir, prompt, 10)
    prefill, decode = results

    # (1) output parity: token 0 came from prefill, the rest decoded remotely
    assert decode["outputs"] == reference
    # (2) byte parity: every transferred page arrived bit-identical
    assert prefill["hashes"] == decode["hashes"]
    assert len(prefill["hashes"]) == 3  # 2 full pages + tail
    # (3) all non-cached pages were injected into the decode pool
    assert decode["injected"] == 3
