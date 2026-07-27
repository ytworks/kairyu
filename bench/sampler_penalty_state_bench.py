#!/usr/bin/env python3
"""Long-context sampler penalty A/B benchmark for issue #216.

Examples:
    uv run python bench/sampler_penalty_state_bench.py --device cpu
    uv run python bench/sampler_penalty_state_bench.py --device cuda:0 \
        --vocab-size 151936 --history-length 32768 --pending-depth 2
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable

import torch

from kairyu.engine.core.sampler import Sampler
from kairyu.engine.core.sampling_types import EngineSampling


def _legacy_apply(
    logits: torch.Tensor,
    sampling: EngineSampling,
    prompt: tuple[int, ...],
    outputs: tuple[int, ...] | list[int],
    pending: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    """The exact pre-#216 CPU/device rebuild paths."""
    work = logits.clone()
    device = logits.device
    vocab = logits.shape[-1]
    if device.type == "cpu":
        effective_outputs = tuple(outputs) + tuple(
            int(token.item()) for token in pending
        )
        if sampling.repetition_penalty != 1.0:
            seen = torch.tensor(
                sorted(set(prompt) | set(effective_outputs)),
                dtype=torch.long,
            )
            if seen.numel():
                values = work[seen]
                work[seen] = torch.where(
                    values > 0,
                    values / sampling.repetition_penalty,
                    values * sampling.repetition_penalty,
                )
    else:
        if sampling.repetition_penalty != 1.0:
            seen_mask = torch.zeros(
                vocab,
                dtype=torch.bool,
                device=device,
            )
            host_seen = tuple(sorted(set(prompt) | set(outputs)))
            seen_mask.scatter_(
                0,
                torch.as_tensor(
                    host_seen,
                    dtype=torch.long,
                    device=device,
                ),
                True,
            )
            for token in pending:
                seen_mask.scatter_(0, token.view(1), True)
            work.copy_(
                torch.where(
                    seen_mask,
                    torch.where(
                        work > 0,
                        work / sampling.repetition_penalty,
                        work * sampling.repetition_penalty,
                    ),
                    work,
                )
            )
    if (
        sampling.frequency_penalty != 0.0
        or sampling.presence_penalty != 0.0
    ):
        if device.type == "cpu":
            counts = torch.bincount(
                torch.tensor(effective_outputs, dtype=torch.long),
                minlength=vocab,
            ).to(work.dtype)
        else:
            counts = torch.zeros(
                vocab,
                dtype=torch.float32,
                device=device,
            )
            indices = torch.as_tensor(
                outputs,
                dtype=torch.long,
                device=device,
            )
            counts.scatter_add_(
                0,
                indices,
                torch.ones_like(indices, dtype=torch.float32),
            )
            for token in pending:
                index = token.to(device=device, dtype=torch.long).view(1)
                counts.scatter_add_(
                    0,
                    index,
                    torch.ones(1, device=device),
                )
        work.sub_(sampling.frequency_penalty * counts)
        work.sub_(
            sampling.presence_penalty * (counts > 0).to(work.dtype)
        )
    return work


class _OverlayState:
    """Committed dense counts plus a transient pending-token overlay."""

    def __init__(
        self,
        prompt: tuple[int, ...],
        outputs: tuple[int, ...],
        vocab: int,
        device: torch.device,
    ) -> None:
        self.device = device
        indices = torch.as_tensor(
            outputs,
            dtype=torch.long,
            device=device,
        )
        self.counts = torch.zeros(
            vocab,
            dtype=torch.float32,
            device=device,
        )
        self.counts.scatter_add_(
            0,
            indices,
            torch.ones_like(indices, dtype=torch.float32),
        )
        self.output_seen = self.counts > 0
        self.repetition_seen = self.output_seen.clone()
        self._one = torch.ones(1, device=device)
        prompt_indices = torch.as_tensor(
            prompt,
            dtype=torch.long,
            device=device,
        )
        self.repetition_seen[prompt_indices] = True
        self.output_indices = torch.tensor(
            sorted(set(outputs)),
            dtype=torch.long,
            device=device,
        )
        self.repetition_indices = torch.tensor(
            sorted(set(prompt) | set(outputs)),
            dtype=torch.long,
            device=device,
        )
        self._cpu_output_ids = (
            set(outputs) if device.type == "cpu" else None
        )
        self._cpu_repetition_ids = (
            set(prompt) | set(outputs)
            if device.type == "cpu"
            else None
        )
        self._cpu_output_positions = (
            {
                int(token): index
                for index, token in enumerate(self.output_indices.tolist())
            }
            if device.type == "cpu"
            else None
        )
        self._cpu_repetition_positions = (
            {
                int(token): index
                for index, token in enumerate(self.repetition_indices.tolist())
            }
            if device.type == "cpu"
            else None
        )
        self._cpu_output_len = self.output_indices.numel()
        self._cpu_repetition_len = self.repetition_indices.numel()

    @staticmethod
    def _append_cpu_index(
        indices: torch.Tensor,
        positions: dict[int, int],
        length: int,
        token_id: int,
    ) -> tuple[torch.Tensor, int]:
        if token_id in positions:
            return indices, length
        if length == indices.numel():
            grown = torch.empty(
                max(8, indices.numel() * 2),
                dtype=torch.long,
            )
            grown[:length].copy_(indices[:length])
            indices = grown
        indices[length] = token_id
        positions[token_id] = length
        return indices, length + 1

    def commit(
        self,
        token_id: int,
        pending_token: torch.Tensor | None,
    ) -> None:
        if self.device.type == "cpu":
            self.counts[token_id] += 1
            self.output_seen[token_id] = True
            self.repetition_seen[token_id] = True
            self._commit_cpu_index(token_id)
            return
        index = torch.tensor(
            [token_id],
            dtype=torch.long,
            device=self.device,
        ) if pending_token is None else pending_token.view(1)
        self.counts.scatter_add_(
            0,
            index,
            self._one,
        )
        self.output_seen[index] = True
        self.repetition_seen[index] = True

    def _commit_cpu_index(self, token_id: int) -> None:
        if self._cpu_output_ids is not None:
            assert self._cpu_repetition_ids is not None
            assert self._cpu_output_positions is not None
            assert self._cpu_repetition_positions is not None
            if token_id not in self._cpu_output_ids:
                self._cpu_output_ids.add(token_id)
                self.output_indices, self._cpu_output_len = (
                    self._append_cpu_index(
                        self.output_indices,
                        self._cpu_output_positions,
                        self._cpu_output_len,
                        token_id,
                    )
                )
            if token_id not in self._cpu_repetition_ids:
                self._cpu_repetition_ids.add(token_id)
                self.repetition_indices, self._cpu_repetition_len = (
                    self._append_cpu_index(
                        self.repetition_indices,
                        self._cpu_repetition_positions,
                        self._cpu_repetition_len,
                        token_id,
                    )
                )

    def apply(
        self,
        logits: torch.Tensor,
        sampling: EngineSampling,
        pending: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        work = logits.clone()
        if work.device.type == "cpu":
            return self._apply_cpu(work, sampling, pending)
        indices = None
        counts = None
        pending_values = None
        if sampling.repetition_penalty != 1.0:
            work.copy_(
                torch.where(
                    self.repetition_seen,
                    torch.where(
                        work > 0,
                        work / sampling.repetition_penalty,
                        work * sampling.repetition_penalty,
                    ),
                    work,
                )
            )
        if pending and sampling.repetition_penalty != 1.0:
            indices, counts = torch.unique(
                torch.stack(pending),
                return_counts=True,
            )
            values = work[indices]
            penalized = torch.where(
                values > 0,
                values / sampling.repetition_penalty,
                values * sampling.repetition_penalty,
            )
            work[indices] = torch.where(
                self.repetition_seen[indices],
                values,
                penalized,
            )
        if pending:
            if indices is None:
                indices, counts = torch.unique(
                    torch.stack(pending),
                    return_counts=True,
                )
            pending_values = work[indices].clone()
        if sampling.frequency_penalty != 0.0:
            work.sub_(sampling.frequency_penalty * self.counts)
        if sampling.presence_penalty != 0.0:
            work.sub_(
                sampling.presence_penalty
                * self.output_seen.to(work.dtype)
            )
        if pending:
            assert indices is not None
            assert counts is not None
            assert pending_values is not None
            if sampling.frequency_penalty != 0.0:
                pending_values.sub_(
                    sampling.frequency_penalty
                    * (
                        self.counts[indices]
                        + counts.to(work.dtype)
                    )
                )
            if sampling.presence_penalty != 0.0:
                pending_values.sub_(sampling.presence_penalty)
            work[indices] = pending_values
        return work

    def _apply_cpu(
        self,
        work: torch.Tensor,
        sampling: EngineSampling,
        pending: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        output_indices = self.output_indices[: self._cpu_output_len]
        repetition_indices = self.repetition_indices[
            : self._cpu_repetition_len
        ]
        if sampling.repetition_penalty != 1.0:
            values = work[repetition_indices]
            work[repetition_indices] = torch.where(
                values > 0,
                values / sampling.repetition_penalty,
                values * sampling.repetition_penalty,
            )
        indices = None
        counts = None
        pending_values = None
        if pending:
            indices, counts = torch.unique(
                torch.stack(pending),
                return_counts=True,
            )
            if sampling.repetition_penalty != 1.0:
                values = work[indices]
                penalized = torch.where(
                    values > 0,
                    values / sampling.repetition_penalty,
                    values * sampling.repetition_penalty,
                )
                work[indices] = torch.where(
                    self.repetition_seen[indices],
                    values,
                    penalized,
                )
            pending_values = work[indices].clone()
        if output_indices.numel():
            values = work[output_indices]
            if sampling.frequency_penalty != 0.0:
                values.sub_(
                    sampling.frequency_penalty
                    * self.counts[output_indices]
                )
            if sampling.presence_penalty != 0.0:
                values.sub_(sampling.presence_penalty)
            work[output_indices] = values
        if pending:
            assert indices is not None
            assert counts is not None
            assert pending_values is not None
            if sampling.frequency_penalty != 0.0:
                pending_values.sub_(
                    sampling.frequency_penalty
                    * (
                        self.counts[indices]
                        + counts.to(work.dtype)
                    )
                )
            if sampling.presence_penalty != 0.0:
                pending_values.sub_(sampling.presence_penalty)
            work[indices] = pending_values
        return work


def _measure(
    call: Callable[[], torch.Tensor],
    device: torch.device,
    iterations: int,
    repeats: int,
) -> dict[str, float | int]:
    for _ in range(min(10, iterations)):
        call()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    samples: list[float] = []
    for _ in range(repeats):
        if device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(iterations):
                call()
            end.record()
            end.synchronize()
            elapsed = start.elapsed_time(end) / 1_000
        else:
            started = time.perf_counter()
            for _ in range(iterations):
                call()
            elapsed = time.perf_counter() - started
        samples.append(elapsed / iterations)
    return {
        "median_us": statistics.median(samples) * 1e6,
        "min_us": min(samples) * 1e6,
        "peak_allocated_bytes": (
            torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else 0
        ),
    }


def _measure_append(
    mode: str,
    logits: torch.Tensor,
    sampling: EngineSampling,
    prompt: tuple[int, ...],
    base_outputs: tuple[int, ...],
    extra_tokens: tuple[int, ...],
    pending_windows: tuple[tuple[torch.Tensor, ...], ...],
    device: torch.device,
    repeats: int,
) -> float:
    samples: list[float] = []
    for _ in range(repeats):
        sampler = Sampler()
        state = sampler._state_for("append-benchmark", sampling)
        overlay = _OverlayState(
            prompt,
            base_outputs,
            logits.shape[-1],
            device,
        )
        warm = logits.clone()
        sampler._apply_incremental_penalties(
            state,
            warm,
            sampling,
            prompt,
            base_outputs,
            pending_windows[0],
            history_epoch=0,
        )
        live_outputs = list(base_outputs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        else:
            started = time.perf_counter()
        iterations = len(pending_windows) - 1
        for index, token_id in enumerate(extra_tokens[:iterations]):
            live_outputs.append(token_id)
            pending = pending_windows[index + 1]
            if mode == "legacy_full_history":
                _legacy_apply(
                    logits,
                    sampling,
                    prompt,
                    live_outputs,
                    pending,
                )
            elif mode == "committed_plus_overlay":
                committed_pending = (
                    pending_windows[index][0]
                    if pending_windows[index]
                    else None
                )
                overlay.commit(token_id, committed_pending)
                overlay.apply(logits, sampling, pending)
            else:
                work = logits.clone()
                sampler._apply_incremental_penalties(
                    state,
                    work,
                    sampling,
                    prompt,
                    live_outputs,
                    pending,
                    history_epoch=0,
                )
        if device.type == "cuda":
            end.record()
            end.synchronize()
            elapsed = start.elapsed_time(end) / 1_000
        else:
            elapsed = time.perf_counter() - started
        samples.append(elapsed / (len(pending_windows) - 1))
    return statistics.median(samples) * 1e6


def _validate_append(
    logits: torch.Tensor,
    sampling: EngineSampling,
    prompt: tuple[int, ...],
    base_outputs: tuple[int, ...],
    extra_tokens: tuple[int, ...],
    pending_windows: tuple[tuple[torch.Tensor, ...], ...],
) -> None:
    """Check every measured commit/pending transition outside the timer."""
    sampler = Sampler()
    state = sampler._state_for("append-validation", sampling)
    overlay = _OverlayState(
        prompt,
        base_outputs,
        logits.shape[-1],
        logits.device,
    )
    warm = logits.clone()
    sampler._apply_incremental_penalties(
        state,
        warm,
        sampling,
        prompt,
        base_outputs,
        pending_windows[0],
        history_epoch=0,
    )
    live_outputs = list(base_outputs)
    for index, token_id in enumerate(
        extra_tokens[: len(pending_windows) - 1]
    ):
        live_outputs.append(token_id)
        pending = pending_windows[index + 1]
        expected = _legacy_apply(
            logits,
            sampling,
            prompt,
            live_outputs,
            pending,
        )
        committed_pending = (
            pending_windows[index][0]
            if pending_windows[index]
            else None
        )
        overlay.commit(token_id, committed_pending)
        overlay_actual = overlay.apply(logits, sampling, pending)
        incremental_actual = logits.clone()
        sampler._apply_incremental_penalties(
            state,
            incremental_actual,
            sampling,
            prompt,
            live_outputs,
            pending,
            history_epoch=0,
        )
        for name, actual in (
            ("committed_plus_overlay", overlay_actual),
            ("incremental_effective_counts", incremental_actual),
        ):
            if not torch.equal(actual, expected):
                different = torch.nonzero(
                    actual != expected,
                    as_tuple=False,
                ).flatten()
                sample = different[:8].tolist()
                raise AssertionError(
                    f"{name} differs from the legacy oracle at append "
                    f"transition {index}; first token IDs: {sample}"
                )


def _validate_duplicate_pending(device: torch.device) -> None:
    """Pin exact rounding when pending repeats an already committed token."""
    sampling = EngineSampling(
        repetition_penalty=1.2,
        presence_penalty=0.4,
        frequency_penalty=0.7,
    )
    prompt = (1, 2)
    outputs = (3, 3, 4)
    pending = (
        torch.tensor(3, device=device),
        torch.tensor(3, device=device),
    )
    logits = torch.randn(
        16,
        generator=torch.Generator(device=device).manual_seed(11),
        device=device,
    )
    expected = _legacy_apply(
        logits,
        sampling,
        prompt,
        outputs,
        pending,
    )
    overlay = _OverlayState(
        prompt,
        outputs,
        logits.shape[-1],
        device,
    )
    overlay_actual = overlay.apply(logits, sampling, pending)
    sampler = Sampler()
    state = sampler._state_for("duplicate-validation", sampling)
    incremental_actual = logits.clone()
    sampler._apply_incremental_penalties(
        state,
        incremental_actual,
        sampling,
        prompt,
        outputs,
        pending,
        history_epoch=0,
    )
    for name, actual in (
        ("committed_plus_overlay", overlay_actual),
        ("incremental_effective_counts", incremental_actual),
    ):
        if not torch.equal(actual, expected):
            raise AssertionError(
                f"{name} differs from the duplicate-pending legacy oracle"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--vocab-size", type=int, default=151_936)
    parser.add_argument("--history-length", type=int, default=32_768)
    parser.add_argument("--pending-depth", type=int, default=2)
    parser.add_argument(
        "--penalties",
        choices=("all", "repetition"),
        default="all",
    )
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if min(
        args.vocab_size,
        args.history_length,
        args.iterations,
        args.repeats,
    ) < 1:
        parser.error(
            "vocab-size, history-length, iterations, and repeats must be positive"
        )
    if args.pending_depth < 0:
        parser.error("pending-depth must be non-negative")

    device = torch.device(args.device)
    _validate_duplicate_pending(device)
    sampling = (
        EngineSampling(repetition_penalty=1.2)
        if args.penalties == "repetition"
        else EngineSampling(
            repetition_penalty=1.2,
            presence_penalty=0.4,
            frequency_penalty=0.7,
        )
    )
    prompt = tuple(index % args.vocab_size for index in range(4_096))
    outputs = tuple(
        (index * 17 + 11) % args.vocab_size
        for index in range(args.history_length)
    )
    pending = tuple(
        torch.tensor(
            (index * 31 + 19) % args.vocab_size,
            dtype=torch.long,
            device=device,
        )
        for index in range(args.pending_depth)
    )
    logits = torch.randn(
        args.vocab_size,
        generator=torch.Generator(device=device).manual_seed(7),
        device=device,
    )

    sampler = Sampler()
    state = sampler._state_for("benchmark", sampling)
    warm = logits.clone()
    sampler._apply_incremental_penalties(
        state,
        warm,
        sampling,
        prompt,
        outputs,
        pending,
        history_epoch=0,
    )
    penalties = next(iter(state.penalty_states.values()))
    overlay = _OverlayState(
        prompt,
        outputs,
        args.vocab_size,
        device,
    )
    modes = {
        "legacy_full_history": lambda: _legacy_apply(
            logits,
            sampling,
            prompt,
            outputs,
            pending,
        ),
        "committed_plus_overlay": lambda: overlay.apply(
            logits,
            sampling,
            pending,
        ),
        "incremental_effective_counts": lambda: _incremental_call(
            sampler,
            state,
            logits,
            sampling,
            prompt,
            outputs,
            pending,
        ),
    }
    expected = modes["legacy_full_history"]()
    for name, call in modes.items():
        if not torch.equal(call(), expected):
            raise AssertionError(
                f"{name} does not match the legacy penalty oracle"
            )
    measured = {
        name: _measure(call, device, args.iterations, args.repeats)
        for name, call in modes.items()
    }
    extra_tokens = tuple(
        (args.history_length * 17 + index * 37 + 23)
        % args.vocab_size
        for index in range(args.iterations + args.pending_depth)
    )
    pending_tokens = tuple(
        torch.tensor(token, dtype=torch.long, device=device)
        for token in extra_tokens
    )
    pending_windows = tuple(
        pending_tokens[
            index : index + args.pending_depth
        ]
        for index in range(args.iterations + 1)
    )
    _validate_append(
        logits,
        sampling,
        prompt,
        outputs,
        extra_tokens,
        pending_windows,
    )
    append_measured = {
        name: _measure_append(
            name,
            logits,
            sampling,
            prompt,
            outputs,
            extra_tokens,
            pending_windows,
            device,
            args.repeats,
        )
        for name in modes
    }
    incremental = measured["incremental_effective_counts"]["median_us"]
    result = {
        "device": str(device),
        "vocab_size": args.vocab_size,
        "history_length": args.history_length,
        "pending_depth": args.pending_depth,
        "penalties": args.penalties,
        "iterations": args.iterations,
        "repeats": args.repeats,
        "validated_bitwise": True,
        "state_bytes": sum(
            tensor.numel() * tensor.element_size()
            for tensor in (
                penalties.prompt_seen,
                penalties.repetition_seen,
                penalties.output_counts,
                penalties.output_seen,
            )
        ),
        "modes": measured,
        "append_modes_median_us": append_measured,
        "legacy_speedup": (
            measured["legacy_full_history"]["median_us"]
            / incremental
        ),
        "overlay_speedup": (
            measured["committed_plus_overlay"]["median_us"]
            / incremental
        ),
        "append_legacy_speedup": (
            append_measured["legacy_full_history"]
            / append_measured["incremental_effective_counts"]
        ),
        "append_overlay_speedup": (
            append_measured["committed_plus_overlay"]
            / append_measured["incremental_effective_counts"]
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


def _incremental_call(
    sampler: Sampler,
    state,
    logits: torch.Tensor,
    sampling: EngineSampling,
    prompt: tuple[int, ...],
    outputs: tuple[int, ...],
    pending: tuple[torch.Tensor, ...],
) -> torch.Tensor:
    work = logits.clone()
    sampler._apply_incremental_penalties(
        state,
        work,
        sampling,
        prompt,
        outputs,
        pending,
        history_epoch=0,
    )
    return work


if __name__ == "__main__":
    main()
