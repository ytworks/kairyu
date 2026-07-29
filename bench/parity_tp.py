"""Gate A1/A2 harness: greedy token parity across TP degrees (goal G2, m5 §6).

Runs the same fixed prompts through the kairyu engine at each TP degree and
compares outputs pairwise against the base degree, with the overlap pipeline
ON and OFF.

Two harnesses, and the result records which one produced it:

- default: the toy CPU forward over FakeCommunicator ranks. Deterministic, so
  parity must be EXACT (A1).
- ``--model-path``: the runners a deployment actually gets — the single-process
  path at tp=1, the spawned ``DistTPLauncher`` group above it, each placed from
  the hardware probe. On a GPU host the reduction order varies with the TP
  degree, so A2's >=99% match rate applies instead of bit-exactness.

Run:
  uv run python bench/parity_tp.py --tp 1,2 --num-prompts 64
  uv run python bench/parity_tp.py --tp 1,2,4,8 --model-path /models/qwen3-32b \\
      --out bench/results/parity-tp-<date>.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from kairyu.engine.core.comm import FakeCommunicator
from kairyu.engine.core.engine_core import EngineCore
from kairyu.engine.core.overlap import OverlapEngineCore
from kairyu.engine.core.radix_kv import RadixKVCache
from kairyu.engine.core.sampling_types import SampledToken
from kairyu.engine.core.scheduler import EngineRequest, Scheduler
from kairyu.engine.core.tp_runner import TPModelRunner, validate_tp_degree

_VOCAB = 50_000
_SEED = 20260702


class _ToyRunner:
    """Deterministic CPU forward (kairyu_backend's toy runner)."""

    def __init__(self, *, sampling_owner: bool = True) -> None:
        self._sampling_owner = sampling_owner
        self._future_tokens: dict[str, dict[int, int]] = {}

    def execute(self, scheduled, states):
        if not self._sampling_owner:
            raise RuntimeError("only rank 0 may sample in tensor-parallel execution")
        sampled = {}
        for chunk in scheduled:
            state = states[chunk.request_id]
            if not chunk.is_prefill or state.prefill_done:
                seed = sum(state.request.prompt_token_ids)
                sampled[chunk.request_id] = (SampledToken((seed + 31 * chunk.position) % _VOCAB),)
        return sampled

    def execute_passive(self, scheduled, states):
        return {}

    def make_sampling_token_packet(self, scheduled, states, sampled=None):
        import torch

        packet = torch.full((len(scheduled),), -1, dtype=torch.int64)
        if sampled is None:
            return packet
        for index, chunk in enumerate(scheduled):
            state = states[chunk.request_id]
            if not chunk.is_prefill or state.prefill_done:
                packet[index] = sampled[chunk.request_id][0].token_id
        return packet

    def adopt_sampling_token_packet(self, scheduled, states, packet):
        for index, chunk in enumerate(scheduled):
            state = states[chunk.request_id]
            if not chunk.is_prefill or state.prefill_done:
                self._future_tokens.setdefault(chunk.request_id, {})[
                    chunk.position
                ] = int(packet[index])


def _fixed_prompts(count: int, max_new: int) -> list[EngineRequest]:
    """Toy prompts. `max_new` is honoured rather than hardcoded: the config
    recorded the CLI value while the run produced 16 tokens regardless, so the
    result described a run that never happened."""
    if count <= 0:
        raise SystemExit(f"--num-prompts must be positive, got {count}")
    rng = random.Random(_SEED)
    return [
        EngineRequest(
            request_id=f"p{i}",
            prompt_token_ids=tuple(rng.randrange(_VOCAB) for _ in range(rng.randrange(8, 96))),
            max_new_tokens=max_new,
        )
        for i in range(count)
    ]


def _make_runner(tp: int):
    if tp == 1:
        return _ToyRunner()
    validate_tp_degree(tp)
    comms = FakeCommunicator.create_group(tp)
    return TPModelRunner(
        rank_runners=tuple(
            _ToyRunner(sampling_owner=rank == 0) for rank in range(tp)
        ),
        comms=comms,
    )


def _model_vocab_size(model_path: str) -> int:
    return int(json.loads((Path(model_path) / "config.json").read_text())["vocab_size"])


def _check_vocab_agreement(model_path: str, tokenizer_vocab: int) -> None:
    """The same guard `build_engine_loop` applies, because bypassing it here
    turns a config mismatch into a device-side assert several layers down."""
    model_vocab = _model_vocab_size(model_path)
    if tokenizer_vocab > model_vocab:
        raise SystemExit(
            f"tokenizer vocab ({tokenizer_vocab}) exceeds the model's vocab_size "
            f"({model_vocab}); the embedding lookup would run out of bounds"
        )


#: Fixed natural-language prompts (A1 pins the prompt set).
#:
#: Random token ids will NOT do here, even though the toy harness uses them. Fed
#: a random id sequence a trained model is far out of distribution, its next-token
#: distribution goes flat, and argmax is decided by near-ties — so bf16 reduction
#: order flips tokens constantly and the gate measures the prompts, not the
#: sharding. Measured on Qwen3-32B: random ids gave 87.5% parity with half the
#: divergences at token 0.
_TEXT_PROMPTS: tuple[str, ...] = (
    "The capital of France is",
    "Water boils at a temperature of",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
    "The three primary colors are red, blue, and",
    "In 1969, humans first walked on the",
    "The chemical symbol for gold is",
    "A right triangle with legs 3 and 4 has a hypotenuse of length",
    "The largest planet in our solar system is",
    "Photosynthesis converts carbon dioxide and water into glucose and",
    "The first ten prime numbers are 2, 3, 5, 7, 11,",
    "SELECT name FROM users WHERE age >",
    "The author of Pride and Prejudice is",
    "To reverse a list in Python you can write",
    "The speed of light in a vacuum is approximately",
    "An octagon has how many sides? The answer is",
    "The Pacific Ocean is the largest ocean on",
    "In object-oriented programming, inheritance allows a class to",
    "The mitochondria are often described as the powerhouse of the",
    "Given f(x) = 2x + 1, the value of f(5) is",
    "HTTP status code 404 means the requested resource was",
    "The longest river in Africa is the",
    "A binary search runs in time proportional to",
    "The freezing point of water in Fahrenheit is",
    "Shakespeare wrote a play about a Danish prince named",
    "The result of 12 multiplied by 12 is",
    "In git, the command to create a new branch is",
    "The atomic number of carbon is",
    "A leap year occurs every four years except when the year is",
    "The tallest mountain above sea level is",
    "To read a file line by line in Python, you would use",
    "The currency of Japan is the",
    "A stack is a data structure that follows the principle of",
    "The boiling point of water at sea level in Celsius is",
    "A palindrome is a word that reads the same",
    "The command to list files in a directory on Linux is",
    "Photosynthesis takes place in the part of the plant cell called the",
    "If a train travels 60 km in 45 minutes, its average speed is",
    "The Great Wall of China was built primarily to",
    "In Python, a dictionary maps keys to",
    "The smallest prime number greater than 20 is",
    "Mount Everest is located on the border between Nepal and",
    "An algorithm with O(n log n) complexity is typically associated with",
    "The human body has how many pairs of chromosomes? The answer is",
    "To create a virtual environment in Python you run",
    "The Declaration of Independence was signed in the year",
    "A JSON object begins with the character",
    "The square root of 144 is",
    "Photons are the elementary particles of",
    "In SQL, the clause used to filter grouped rows is",
    "The largest desert in the world by area is the",
    "A linked list differs from an array because it does not require",
    "The inventor of the telephone is generally credited as",
    "In Git, the command that shows the commit history is",
    "The chemical formula for table salt is",
    "A regular hexagon has interior angles measuring",
    "The capital city of Australia is",
    "In machine learning, overfitting means the model has learned",
    "The Fibonacci sequence begins 0, 1, 1, 2, 3, 5,",
    "TCP differs from UDP mainly because TCP guarantees",
    "The element with atomic number 1 is",
    "To sort a list in reverse order in Python, pass the argument",
    "The Amazon rainforest is located primarily in the country of",
    "A byte consists of how many bits? The answer is",
    "The Pythagorean theorem relates the sides of a",
)



def _real_prompts(model_path: str, count: int, max_new: int) -> list[EngineRequest]:
    """Fixed prompts tokenized by the model's own tokenizer (A1 pins the set)."""
    from kairyu.engine.tokenizer import resolve_tokenizer

    resolved = resolve_tokenizer(model_path)
    _check_vocab_agreement(model_path, len(resolved.vocab()))
    if count <= 0:
        raise SystemExit(f"--num-prompts must be positive, got {count}")
    if count > len(_TEXT_PROMPTS):
        raise SystemExit(
            f"--num-prompts {count} exceeds the {len(_TEXT_PROMPTS)} fixed prompts; "
            "add prompts to _TEXT_PROMPTS rather than repeating them, or a repeat "
            "would inflate the match rate with duplicate rows"
        )
    return [
        EngineRequest(
            request_id=f"p{index}",
            prompt_token_ids=tuple(resolved.encode(text)),
            max_new_tokens=max_new,
        )
        for index, text in enumerate(_TEXT_PROMPTS[:count])
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _weight_digests(shards: list[Path]) -> list[dict]:
    """Full SHA-256 of every weight file.

    An earlier version hashed only each shard's safetensors HEADER — the JSON
    map of tensor names, dtypes, shapes and byte offsets — plus the file size.
    That identifies the LAYOUT, not the weights: a base checkpoint and a
    fine-tune of it share both, so they produced the same digest while every
    weight value differed. Rewriting one payload byte without changing the file
    size left the digest untouched.

    Reading the bytes costs about a minute over a 32B checkpoint, against a run
    that loads the same bytes onto the GPU anyway. hashlib releases the GIL, so
    the shards hash in parallel.

    A safetensors shard's SHA-256 is also the oid Hugging Face publishes for
    that LFS blob, so these digests identify the checkpoint against an upstream
    immutable revision rather than against a path on one machine.
    """
    with ThreadPoolExecutor(max_workers=min(8, len(shards))) as pool:
        digests = list(pool.map(_sha256, shards))
    return [
        {"file": shard.name, "bytes": shard.stat().st_size, "sha256": digest}
        for shard, digest in zip(shards, digests, strict=True)
    ]


def _checkpoint_provenance(model_path: str) -> dict:
    """Identify the checkpoint by content, not by the path it was read from.

    G2 §8 requires that a number a decision rests on be reviewable next to the
    config that produced it. A machine-local `model_path` does not survive the
    machine, so the evidence records what the weights and tokenizer actually
    were: a per-shard header digest, the file sizes, and a hash of the config
    and tokenizer files.
    """
    root = Path(model_path)
    if not root.is_dir():
        raise SystemExit(f"--model-path is not a directory: {model_path}")

    shards = sorted(root.glob("*.safetensors"))
    if not shards:
        raise SystemExit(f"no *.safetensors under {model_path}; cannot record provenance")

    weights = _weight_digests(shards)
    # one digest over every shard identity, so evidence can be compared at a glance
    rollup = hashlib.sha256(
        json.dumps(weights, sort_keys=True).encode()
    ).hexdigest()

    metadata = {}
    for name in ("config.json", "generation_config.json"):
        candidate = root / name
        if candidate.is_file():
            metadata[name] = _sha256(candidate)

    tokenizer = {}
    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
        candidate = root / name
        if candidate.is_file():
            tokenizer[name] = _sha256(candidate)
    if not tokenizer:
        raise SystemExit(f"no tokenizer files under {model_path}; cannot record provenance")

    name = None
    architecture = None
    config_file = root / "config.json"
    if config_file.is_file():
        parsed = json.loads(config_file.read_text())
        name = parsed.get("_name_or_path") or parsed.get("model_type")
        architecture = {
            key: parsed.get(key)
            for key in (
                "model_type",
                "hidden_size",
                "intermediate_size",
                "num_hidden_layers",
                "num_attention_heads",
                "num_key_value_heads",
                "vocab_size",
                "torch_dtype",
            )
        }

    return {
        "name": name or root.name,
        "architecture": architecture,
        "weights_sha256": rollup,
        "weights": weights,
        "metadata_sha256": metadata,
        "tokenizer_sha256": tokenizer,
        # kept for debugging only — it identifies nothing on another machine
        "read_from": str(root),
    }


def _code_provenance() -> dict:
    """The commit that produced the numbers, and whether it was modified."""
    def _git(*argv: str) -> str | None:
        try:
            done = subprocess.run(
                ["git", *argv],
                cwd=Path(__file__).resolve().parent,
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError:
            return None
        return done.stdout.strip() if done.returncode == 0 else None

    commit = _git("rev-parse", "HEAD")
    tracked = _git("status", "--porcelain", "--untracked-files=no")
    untracked = _git("ls-files", "--others", "--exclude-standard")
    source = "git"
    # Production benchmark images intentionally omit git. In that environment
    # the caller supplies the already-resolved host values; requiring both the
    # full commit and dirty flag prevents a partial fallback from looking clean.
    if commit is None:
        supplied_commit = os.environ.get("KAIRYU_BENCH_COMMIT")
        supplied_dirty = os.environ.get("KAIRYU_BENCH_DIRTY")
        if (
            supplied_commit is not None
            and len(supplied_commit) == 40
            and all(character in "0123456789abcdef" for character in supplied_commit)
            and supplied_dirty in {"0", "1"}
        ):
            commit = supplied_commit
            tracked = "" if supplied_dirty == "0" else "<supplied dirty>"
            supplied_untracked = os.environ.get("KAIRYU_BENCH_UNTRACKED_FILES")
            untracked = (
                "\n" * int(supplied_untracked)
                if supplied_untracked is not None and supplied_untracked.isdigit()
                else None
            )
            source = "environment"
        else:
            source = None
    return {
        "commit": commit,
        # a modified tracked file means the commit alone does not describe the run
        "dirty": None if tracked is None else bool(tracked),
        # untracked files are reported separately rather than folded into `dirty`:
        # scratch dirs and agent worktrees sit in the tree during a run without
        # changing the code that produced the numbers, but a reader still gets to
        # see that something unversioned was present.
        "untracked_files": None if untracked is None else len(untracked.splitlines()),
        "source": source,
    }


def _gpu_runtime_provenance() -> dict:
    """Record the CUDA/NCCL runtime and knobs that can change a TP result."""
    import torch

    driver = None
    topology = None
    try:
        done = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,pci.bus_id,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if done.returncode == 0:
            driver = [line.strip() for line in done.stdout.splitlines() if line.strip()]
        topo = subprocess.run(
            ["nvidia-smi", "topo", "-m"],
            capture_output=True,
            text=True,
            check=False,
        )
        if topo.returncode == 0:
            topology = topo.stdout.strip().splitlines()
    except OSError:
        pass

    nccl_version = None
    if torch.distributed.is_nccl_available():
        raw_nccl_version = torch.cuda.nccl.version()
        nccl_version = (
            ".".join(str(part) for part in raw_nccl_version)
            if isinstance(raw_nccl_version, tuple)
            else str(raw_nccl_version)
        )
    environment = {
        name: os.environ[name]
        for name in (
            "CUDA_VISIBLE_DEVICES",
            "KAIRYU_DIST_BACKEND",
            "NCCL_ALGO",
            "NCCL_DEBUG",
            "NCCL_IB_DISABLE",
            "NCCL_P2P_DISABLE",
            "NCCL_PROTO",
        )
        if name in os.environ
    }
    return {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "nccl": nccl_version,
        "device_capabilities": [
            list(torch.cuda.get_device_capability(index))
            for index in range(torch.cuda.device_count())
        ],
        "nvidia_smi": driver,
        "nvidia_smi_topology": topology,
        "environment": environment,
    }


def _compare(base: dict, candidate: dict) -> dict:
    """Compare two runs' outputs prompt by prompt.

    Reports the counts the verdict uses, plus the FIRST disagreeing request and
    position. Aggregate rates alone cannot distinguish two runs that diverge on
    different prompts but happen to land on the same totals, so the identity of
    the first divergence is recorded rather than only its median depth.
    """
    matches = 0
    token_total = 0
    token_matched = 0
    first_divergence: list[int] = []
    first_mismatch = None
    for rid, expected in base.items():
        actual = candidate.get(rid, ())
        if actual == expected:
            matches += 1
        token_total += len(expected)
        agreed = 0
        for position, token in enumerate(expected):
            if position < len(actual) and actual[position] == token:
                agreed += 1
            else:
                first_divergence.append(position)
                if first_mismatch is None:
                    first_mismatch = {
                        "request_id": rid,
                        "position": position,
                        "expected": token,
                        "actual": actual[position] if position < len(actual) else None,
                    }
                break
        else:
            agreed = len(expected)
        token_matched += agreed
    return {
        "prompts": len(base),
        "exact_match": matches,
        # match_rate is DISPLAY only: 20000/20001 rounds to 1.0 at four places
        # and would clear an `== 1.0` gate. The verdict compares the counts.
        "match_rate": round(matches / len(base), 4) if base else 0.0,
        "tokens": token_total,
        "token_match_rate": round(token_matched / token_total, 4) if token_total else 0.0,
        "median_first_divergence": (
            sorted(first_divergence)[len(first_divergence) // 2]
            if first_divergence
            else None
        ),
        "first_mismatch": first_mismatch,
    }


def _real_runner(model_path: str, tp: int, num_pages: int, page_size: int):
    """The runner a deployment actually gets at this TP degree.

    tp=1 is the single-process path; tp>1 is the spawned DistTPLauncher group.
    Both place themselves from the hardware probe, so on a GPU host this is a
    real-ranks comparison rather than the toy CPU one.
    """
    from kairyu.engine.tokenizer import resolve_tokenizer

    resolved = resolve_tokenizer(model_path)
    vocab = list(resolved.vocab())
    if tp > 1:
        from kairyu.engine.core.worker import DistTPLauncher

        launcher = DistTPLauncher(
            model_path, tp=tp, num_pages=num_pages, page_size=page_size, vocab=vocab
        )
        return launcher.runner, launcher.shutdown

    import torch

    from kairyu.engine.core.attention import select_backend
    from kairyu.engine.core.hw_profile import probe
    from kairyu.engine.core.kv_pool import PagedKVPool
    from kairyu.engine.core.model_runner import PagedModelRunner
    from kairyu.engine.core.sampler import Sampler
    from kairyu.models.loader import load_model

    profile = probe()
    gpu = profile.arch == "cuda"
    device = "cuda:0" if gpu else "cpu"
    dtype = torch.bfloat16 if gpu else torch.float32
    model, config, _ = load_model(
        model_path, dtype=dtype, attention_backend=select_backend(profile)
    )
    model = model.to(device)
    pool = PagedKVPool(
        num_layers=config.num_hidden_layers,
        num_pages=num_pages,
        page_size=page_size,
        num_kv_heads=config.kv_cache_num_heads,
        head_dim=config.kv_cache_head_dim,
        dtype=dtype,
        device=device,
    )
    runner = PagedModelRunner(
        model, pool, sampler=Sampler(vocab_provider=lambda: vocab)
    )

    def teardown() -> None:
        """Drop this degree's weights. The CALLER must clear its own references
        first; this only releases the ones the builder still holds."""
        nonlocal model, pool, runner
        del runner, pool, model
        import gc

        gc.collect()
        if gpu:
            torch.cuda.empty_cache()

    return runner, teardown


def _run(
    tp: int,
    overlap: bool,
    prompts: list[EngineRequest],
    model_path: str | None = None,
    num_pages: int = 4096,
    page_size: int = 16,
) -> dict[str, tuple[int, ...]]:
    kv = RadixKVCache(num_pages=num_pages, page_size=page_size)
    scheduler = Scheduler(kv, max_num_batched_tokens=2048, page_size=page_size)
    if model_path is None:
        runner, teardown = _make_runner(tp), (lambda: None)
    else:
        runner, teardown = _real_runner(model_path, tp, num_pages, page_size)
    try:
        core = OverlapEngineCore(scheduler, runner) if overlap else EngineCore(scheduler, runner)
        for request in prompts:
            core.add_request(request)
        outputs = core.run_to_completion()
    finally:
        # the caller's own references keep the model alive, so teardown cannot
        # free it on its own — a degree sweep otherwise carries every previous
        # degree's weights (observed: 67 GB still on cuda:0 from the tp=1 pass
        # while tp=4 ran)
        core = None
        runner = None
        scheduler = None
        kv = None
        teardown()
    # A request the scheduler REJECTED (KV exhaustion) comes back as an empty
    # tuple, and two empty tuples compare equal — so a run that inferred nothing
    # at all scored a perfect match. Verified: --num-pages 1 --page-size 1 gave
    # match_rate 1.0 with tokens 0 and exit 0 (review [P1] on #130).
    short = {
        request.request_id: len(outputs.get(request.request_id, ()))
        for request in prompts
        if len(outputs.get(request.request_id, ())) < request.max_new_tokens
    }
    if short:
        raise SystemExit(
            f"{len(short)} of {len(prompts)} requests produced fewer than "
            f"max_new_tokens at tp={tp} (overlap={'on' if overlap else 'off'}): "
            f"{dict(list(short.items())[:5])}... — the KV pool is too small or the "
            "scheduler rejected them; a parity score over these is meaningless"
        )
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tp", default="1,2", help="comma-separated TP degrees; first is base")
    parser.add_argument("--num-prompts", type=int, default=64)
    parser.add_argument(
        "--model-path",
        help="checkpoint dir; runs the REAL runners (single-process at tp=1, the "
        "spawned DistTPLauncher group above it). Omit for the toy CPU harness.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--num-pages", type=int, default=4096)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument(
        "--no-overlap-sweep",
        action="store_true",
        help="overlap OFF only; halves the run for a large real model",
    )
    parser.add_argument("--out", type=Path, help="write the JSON result here too")
    args = parser.parse_args()
    degrees = [int(value) for value in args.tp.split(",")]
    if len(degrees) < 2:
        raise SystemExit("need at least a base degree and one comparison degree")

    if args.model_path:
        prompts = _real_prompts(args.model_path, args.num_prompts, args.max_new_tokens)
    else:
        prompts = _fixed_prompts(args.num_prompts, args.max_new_tokens)

    # A1 requires overlap ON and OFF. The ON half used to be skipped here:
    # `PagedModelRunner` read `state.outputs[position - 1]`, which an overlap
    # snapshot is one short of, so a real runner raised IndexError. The in-flight
    # token buffer fixes that, so both halves run.
    overlap_modes = (False,) if args.no_overlap_sweep else (False, True)
    overlap_note = None
    results = {}
    runs: dict[tuple[int, bool], dict] = {}
    for overlap in overlap_modes:
        base = _run(
            degrees[0], overlap, prompts, args.model_path, args.num_pages, args.page_size
        )
        runs[(degrees[0], overlap)] = base
        for degree in degrees[1:]:
            candidate = _run(
                degree, overlap, prompts, args.model_path, args.num_pages, args.page_size
            )
            runs[(degree, overlap)] = candidate
            results[f"tp{degree}_vs_tp{degrees[0]}_overlap_{'on' if overlap else 'off'}"] = (
                _compare(base, candidate)
            )

    # A1 wants parity with the pipeline ON *and* OFF, and the loop above cannot
    # give it: each mode is compared against its own TP1 base, and the outputs
    # are dropped when the next mode overwrites them. Two modes can land on the
    # same exact_match and token_match_rate while diverging on DIFFERENT prompts
    # -- equal aggregates are not sequence equality. So the ON-vs-OFF comparison
    # is made directly, per TP degree, and recorded.
    overlap_equality = {}
    if True in overlap_modes and False in overlap_modes:
        for degree in degrees:
            overlap_equality[f"tp{degree}_overlap_on_vs_off"] = _compare(
                runs[(degree, False)], runs[(degree, True)]
            )

    # the harness label is part of the evidence: a CPU-toy 1.0 and a real-ranks
    # 1.0 are not the same claim, and G2 §8 forbids a number without its config
    if args.model_path:
        from kairyu.engine.core.hw_profile import probe

        profile = probe()
        harness = f"real-model ranks on {profile.arch}"
        hardware = {
            "arch": profile.arch,
            "device_name": profile.device_name,
            "device_count": profile.device_count,
            "sm": profile.sm,
            "runtime": _gpu_runtime_provenance(),
        }
    else:
        harness = "cpu-toy (pass --model-path for real ranks)"
        hardware = {"arch": "cpu"}

    config = {
        "harness": harness,
        "hardware": hardware,
        "checkpoint": _checkpoint_provenance(args.model_path) if args.model_path else None,
        "code": _code_provenance(),
        "tp_degrees": degrees,
        "num_prompts": len(prompts),
        "max_new_tokens": args.max_new_tokens,
        "overlap_modes": ["on" if mode else "off" for mode in overlap_modes],
        "overlap_on_gap": overlap_note,
        "seed": _SEED,
        "sampling": {
            "method": "greedy",
            "temperature": None,
            "top_p": None,
            "top_k": None,
        },
    }
    prompt_evidence = [
        {
            "request_id": prompt.request_id,
            "text": _TEXT_PROMPTS[index] if args.model_path else None,
            "token_ids": list(prompt.prompt_token_ids),
        }
        for index, prompt in enumerate(prompts)
    ]
    continuations = {
        f"tp{degree}_overlap_{'on' if overlap else 'off'}": {
            request_id: list(token_ids)
            for request_id, token_ids in sorted(run.items())
        }
        for (degree, overlap), run in sorted(runs.items())
    }
    payload = {
        "config": config,
        "prompts": prompt_evidence,
        "continuations": continuations,
        "parity": results,
        "overlap_equality": overlap_equality,
    }
    print(json.dumps(payload, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"written: {args.out}")
    worst = min(entry["match_rate"] for entry in results.values())
    all_exact = all(
        entry["exact_match"] == entry["prompts"] for entry in results.values()
    )
    # This one IS a gate, on both harnesses. Cross-TP differences are reduction
    # order; overlap ON vs OFF is the SAME ranks in the same order, so any
    # difference is the pipeline changing an answer. Reported as a verdict
    # because the claim "the pipeline changes no output" is only earned by
    # comparing the outputs, not by two aggregate rows agreeing.
    overlap_exact = all(
        entry["exact_match"] == entry["prompts"] for entry in overlap_equality.values()
    )
    for name, entry in overlap_equality.items():
        status = "OK" if entry["exact_match"] == entry["prompts"] else "FAIL"
        detail = ""
        if entry["first_mismatch"] is not None:
            first = entry["first_mismatch"]
            detail = (
                f" first mismatch: {first['request_id']} at position "
                f"{first['position']} ({first['expected']} -> {first['actual']})"
            )
        print(
            f"overlap ON vs OFF {name}: {entry['exact_match']}/{entry['prompts']} "
            f"exact -> {status}{detail}"
        )
    if overlap_equality and not overlap_exact:
        print("overlap ON changed the output -- the pipeline is not transparent")
        return 1

    if args.model_path:
        # G2 §7 (amended 2026-07-25): free-running greedy sequence equality is NOT
        # a correctness gate. One flipped token fails a whole prompt and every
        # token after it, so these rates cannot separate a broken shard from a
        # moved near-tie — `bench/parity_hf.py` is what gates. Reported for
        # orientation; no verdict is claimed.
        print(
            f"worst free-running match rate: {worst:.4f} (orientation only — "
            "the correctness bar is the teacher-forced agreement in "
            "bench/parity_hf.py)"
        )
        return 0
    # The toy harness IS deterministic, so exactness here is a real invariant:
    # sharding over FakeCommunicator ranks must not change a single token.
    verdict = "OK" if all_exact else "FAIL"
    print(f"worst match rate: {worst:.4f} (toy harness must be exact) -> {verdict}")
    return 0 if all_exact else 1


if __name__ == "__main__":
    raise SystemExit(main())
