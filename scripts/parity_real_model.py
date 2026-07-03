#!/usr/bin/env python
"""Pre-deploy manual gate (m12 D6 secondary): real-model greedy parity on CPU.

Downloads a small real checkpoint from the HF Hub and compares 20-token greedy
through the full Kairyu engine vs transformers.generate. Catches config-parsing
gaps that tiny synthetic configs miss (rope variants, tied embeddings, eos
lists). Run manually before deploy day:

    uv run python scripts/parity_real_model.py [--model Qwen/Qwen2.5-0.5B-Instruct]
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--prompt", default="The capital of Japan is")
    parser.add_argument("--max-new-tokens", type=int, default=20)
    args = parser.parse_args()

    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from kairyu.engine.core.engine_core import EngineCore
    from kairyu.engine.core.kv_pool import PagedKVPool
    from kairyu.engine.core.model_runner import PagedModelRunner
    from kairyu.engine.core.radix_kv import RadixKVCache
    from kairyu.engine.core.sampler import Sampler
    from kairyu.engine.core.sampling_types import EngineSampling
    from kairyu.engine.core.scheduler import EngineRequest, Scheduler
    from kairyu.models.loader import load_model

    print(f"downloading {args.model} ...")
    path = snapshot_download(args.model, allow_patterns=["*.json", "*.safetensors"])

    hf_tokenizer = AutoTokenizer.from_pretrained(path)
    prompt_ids = hf_tokenizer(args.prompt, return_tensors="pt").input_ids

    print("loading transformers oracle (fp32) ...")
    oracle = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval()
    reference = oracle.generate(
        prompt_ids,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        pad_token_id=hf_tokenizer.pad_token_id or 0,
    )[0, prompt_ids.shape[1]:].tolist()
    del oracle

    print("loading kairyu DenseDecoder ...")
    model, config, generation = load_model(path)
    cache = RadixKVCache(num_pages=512, page_size=16)
    scheduler = Scheduler(cache, max_num_batched_tokens=256, page_size=16)
    pool = PagedKVPool.for_cache(cache, config)
    runner = PagedModelRunner(model, pool, sampler=Sampler(), cache=cache)
    engine = EngineCore(scheduler, runner)
    engine.add_request(
        EngineRequest(
            "parity",
            tuple(prompt_ids[0].tolist()),
            max_new_tokens=args.max_new_tokens,
            eos_token_id=generation.eos_token_id,
            stop_token_ids=generation.stop_token_ids,
            sampling=EngineSampling(temperature=0.0),
        )
    )
    ours = list(engine.run_to_completion()["parity"])

    print(f"reference: {reference}")
    print(f"kairyu:    {ours}")
    print(f"reference text: {hf_tokenizer.decode(reference)!r}")
    print(f"kairyu text:    {hf_tokenizer.decode(ours)!r}")
    if ours == reference[: len(ours)] and len(ours) >= min(len(reference), 1):
        print("PARITY OK")
        return 0
    print("PARITY FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
