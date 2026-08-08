"""CPU/structured and CUDA sampling share one full-width RNG (#353/#354)."""

from __future__ import annotations

import pytest
import torch

from kairyu.bench.profiling import profile_scope
from kairyu.engine.core.sampler import Sampler
from kairyu.engine.core.sampling_types import EngineSampling, mix_seed
from kairyu.kernels.sampling_gpu import (
    _stateless_random_words,
    stateless_gumbel_argmax,
)

pytestmark = pytest.mark.gpu

_VOCAB_SIZE = 257


@pytest.fixture(scope="module", autouse=True)
def _require_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("sampling RNG parity requires CUDA")


def _logits(*, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Fixed, tie-free logits whose filter boundaries have comfortable gaps."""
    indices = torch.arange(_VOCAB_SIZE, dtype=torch.float32)
    values = torch.linspace(-2.75, 2.25, _VOCAB_SIZE)
    values = values + 0.173 * torch.sin(indices * 0.37)
    return values.to(dtype=dtype)


def _sample_pair(
    sampling: EngineSampling,
    position: int,
    logits: torch.Tensor,
):
    cpu = Sampler().sample(
        "cpu",
        sampling,
        position,
        logits,
    )
    cuda = Sampler().sample_device(
        "cuda",
        sampling,
        position,
        logits.to(device="cuda"),
    )
    return cpu, cuda


@pytest.mark.parametrize(
    ("case", "options"),
    (
        ("temperature", {"temperature": 0.73}),
        ("min-p", {"temperature": 0.83, "min_p": 0.12}),
        ("top-k", {"temperature": 0.91, "top_k": 23}),
        ("top-p", {"temperature": 0.77, "top_p": 0.82}),
        (
            "combined",
            {
                "temperature": 0.64,
                "min_p": 0.02,
                "top_k": 41,
                "top_p": 0.90,
            },
        ),
    ),
    ids=lambda value: value if isinstance(value, str) else None,
)
@pytest.mark.parametrize(
    ("seed", "position"),
    ((0, 0), (1, 3), (71, 11), (2**31 + 17, 29)),
)
def test_cpu_and_cuda_sampling_match_for_every_filter_combination(
    case: str,
    options: dict[str, float | int],
    seed: int,
    position: int,
) -> None:
    del case
    sampling = EngineSampling(seed=seed, **options)

    cpu, cuda = _sample_pair(sampling, position, _logits())

    assert int(cuda.token_id.cpu()) == cpu.token_id


def test_stateless_gumbel_primitive_matches_on_cpu_and_cuda() -> None:
    log_weights = torch.log_softmax(_logits(), dim=-1)

    for base_seed, position in ((0, 0), (1, 7), (353, 19), (2**31 + 9, 31)):
        seed = mix_seed(base_seed, position)
        cpu = stateless_gumbel_argmax(log_weights, seed)
        cuda = stateless_gumbel_argmax(log_weights.cuda(), seed)

        assert cpu.device.type == "cpu"
        assert cuda.device.type == "cuda"
        assert int(cuda.cpu()) == int(cpu)


def test_full_width_counter_words_match_exactly_on_cpu_and_cuda() -> None:
    offsets = torch.tensor([0, 1, 2, 3, 7, 31, 257, 151_935], dtype=torch.int64)

    for seed in (0, 1, 1 << 32, 1 << 48, 1 << 63, (1 << 64) - 1):
        cpu = _stateless_random_words(offsets, seed)
        cuda = _stateless_random_words(offsets.cuda(), seed)
        assert torch.equal(cuda.cpu(), cpu)


def test_gumbel_primitive_keeps_result_on_cuda_without_scalar_sync() -> None:
    log_weights = torch.log_softmax(_logits().cuda(), dim=-1)
    stateless_gumbel_argmax(log_weights, mix_seed(354, 0))

    with profile_scope(
        enabled=True,
        activities=("cpu", "cuda"),
        acc_events=False,
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as trace:
        token = stateless_gumbel_argmax(log_weights, mix_seed(354, 1))

    scalar_reads = sum(
        event.count for event in trace.key_averages() if event.key == "aten::_local_scalar_dense"
    )
    assert token.device.type == "cuda"
    assert scalar_reads == 0


def test_request_derived_seed_matches_across_cpu_and_cuda() -> None:
    sampling = EngineSampling(temperature=0.81, top_p=0.91)
    logits = _logits()

    for position in (0, 3, 17):
        cpu = Sampler().sample("same-public-id", sampling, position, logits)
        cuda = Sampler().sample_device(
            "same-public-id",
            sampling,
            position,
            logits.cuda(),
        )

        assert int(cuda.token_id.cpu()) == cpu.token_id


def test_bfloat16_inputs_use_the_same_canonical_draw() -> None:
    sampling = EngineSampling(
        temperature=0.68,
        min_p=0.015,
        top_k=53,
        top_p=0.93,
        seed=821,
    )
    logits = _logits(dtype=torch.bfloat16)

    for position in (0, 5, 23):
        cpu, cuda = _sample_pair(sampling, position, logits)
        assert int(cuda.token_id.cpu()) == cpu.token_id


def test_penalties_and_min_tokens_preserve_cpu_cuda_rng_parity() -> None:
    sampling = EngineSampling(
        temperature=0.79,
        min_p=0.01,
        top_k=47,
        top_p=0.92,
        repetition_penalty=1.17,
        presence_penalty=0.45,
        frequency_penalty=0.22,
        seed=997,
    )
    logits = _logits()
    logits[3] = 12.0  # EOS would dominate without the minimum-token mask.
    logits[5] = 11.0  # The explicit stop ID must be masked as well.
    prompt = (1, 7, 9, 7)
    outputs = (7, 8, 7)
    position = 4

    cpu = Sampler().sample(
        "penalty-cpu",
        sampling,
        position,
        logits,
        prompt=prompt,
        outputs=outputs,
        pending_outputs=(10,),
        history_epoch=0,
        eos_token_id=3,
        stop_token_ids=(5,),
        min_tokens=5,
    )
    cuda = Sampler().sample_device(
        "penalty-cuda",
        sampling,
        position,
        logits.cuda(),
        prompt=prompt,
        outputs=outputs,
        pending_outputs=(torch.tensor(10, device="cuda"),),
        history_epoch=0,
        eos_token_id=3,
        stop_token_ids=(5,),
        min_tokens=5,
    )

    assert cpu.token_id not in (3, 5)
    assert int(cuda.token_id.cpu()) == cpu.token_id


class _PermissiveEnforcer:
    """Stateful matcher whose support is identical across CPU and CUDA."""

    def __init__(self) -> None:
        self.accepted: list[int] = []

    def mask_logits(self, logits: torch.Tensor) -> torch.Tensor:
        return logits

    def accept(self, token_id: int) -> bool:
        self.accepted.append(token_id)
        return True

    def is_terminated(self) -> bool:
        return False


class _EvenEnforcer(_PermissiveEnforcer):
    """Matcher stand-in that keeps only even token IDs."""

    @staticmethod
    def mask_logits(logits: torch.Tensor) -> torch.Tensor:
        logits[1::2] = float("-inf")
        return logits


def test_permissive_structured_matcher_matches_on_cpu_and_cuda() -> None:
    sampling = EngineSampling(
        temperature=0.72,
        min_p=0.01,
        top_k=61,
        top_p=0.94,
        seed=353,
    )
    logits = _logits()
    cpu_sampler = Sampler()
    enforcer = _PermissiveEnforcer()
    cpu_sampler._state_for("structured", sampling).enforcer = enforcer

    cpu = cpu_sampler.sample("structured", sampling, 13, logits)
    cuda_sampler = Sampler()
    cuda_enforcer = _PermissiveEnforcer()
    cuda_sampler._state_for("structured-cuda", sampling).enforcer = cuda_enforcer
    cuda = cuda_sampler.sample_device("structured-cuda", sampling, 13, logits.cuda())

    assert int(cuda.token_id.cpu()) == cpu.token_id
    assert enforcer.accepted == [cpu.token_id]
    assert cuda_enforcer.accepted == [cpu.token_id]


def test_restrictive_structured_mask_matches_on_cpu_and_cuda() -> None:
    sampling = EngineSampling(
        temperature=0.69,
        min_p=0.01,
        top_k=59,
        top_p=0.93,
        seed=733,
    )
    logits = _logits()
    cpu_sampler = Sampler()
    enforcer = _EvenEnforcer()
    cpu_sampler._state_for("structured-even", sampling).enforcer = enforcer
    cuda_sampler = Sampler()
    cuda_enforcer = _EvenEnforcer()
    cuda_sampler._state_for("structured-even-cuda", sampling).enforcer = cuda_enforcer

    cpu = cpu_sampler.sample("structured-even", sampling, 9, logits)
    cuda = cuda_sampler.sample_device(
        "structured-even-cuda",
        sampling,
        9,
        logits.cuda(),
    )

    assert cpu.token_id % 2 == 0
    assert int(cuda.token_id.cpu()) == cpu.token_id
    assert enforcer.accepted == [cpu.token_id]
    assert cuda_enforcer.accepted == [cpu.token_id]


def test_raw_logprobs_are_unchanged_while_selected_tokens_match() -> None:
    sampling = EngineSampling(
        temperature=0.57,
        min_p=0.02,
        top_k=37,
        top_p=0.88,
        seed=1_009,
        logprobs=7,
    )
    logits = _logits()
    expected = torch.log_softmax(logits, dim=-1)
    expected_values, expected_indices = torch.topk(expected, 7)

    cpu, cuda = _sample_pair(sampling, 17, logits)

    assert int(cuda.token_id.cpu()) == cpu.token_id
    assert cpu.logprob == pytest.approx(float(expected[cpu.token_id]), abs=1e-6)
    assert float(cuda.logprob.cpu()) == pytest.approx(
        float(expected[cpu.token_id]), abs=1e-5
    )
    assert cpu.top_logprobs is not None
    assert [token_id for token_id, _ in cpu.top_logprobs] == expected_indices.tolist()
    assert [value for _, value in cpu.top_logprobs] == pytest.approx(
        expected_values.tolist(), abs=1e-6
    )
    assert cuda.top_indices is not None
    assert cuda.top_logprobs is not None
    assert cuda.top_indices.cpu().tolist() == expected_indices.tolist()
    assert cuda.top_logprobs.cpu().tolist() == pytest.approx(
        expected_values.tolist(), abs=1e-5
    )


def test_cuda_canonical_gumbel_trajectory_uses_full_width_issue_354_sequence() -> None:
    sampling = EngineSampling(
        temperature=0.64,
        min_p=0.02,
        top_k=41,
        top_p=0.90,
        seed=353,
    )
    logits = _logits().cuda()
    sampler = Sampler()

    actual = [
        int(sampler.sample_device("golden", sampling, position, logits).token_id.cpu())
        for position in range(12)
    ]

    assert actual == [228, 240, 237, 231, 254, 247, 247, 253, 244, 231, 232, 229]
