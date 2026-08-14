# Copyright 2023 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Small library facade over Google's official IFEval checker.

This is adapted from ``instruction_following_eval/evaluation_main.py`` at
google-research/google-research commit
``066e1eda43f4785922e3994e95429e496080231f``. Kairyu removes the absl CLI and
file I/O and exposes the same strict and loose checks as pure functions. The
eight loose-response candidates and instruction construction order are kept
identical to upstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from evals._vendor.ifeval import instructions_registry


@dataclass(frozen=True)
class IFEvalInput:
    """One pinned IFEval prompt and its executable instruction descriptions."""

    prompt: str
    instruction_id_list: tuple[str, ...]
    kwargs: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class IFEvalResult:
    """Strict and loose per-instruction booleans for one response."""

    strict: tuple[bool, ...]
    loose: tuple[bool, ...]

    @property
    def strict_prompt(self) -> bool:
        return all(self.strict)

    @property
    def loose_prompt(self) -> bool:
        return all(self.loose)


def _instruction(inp: IFEvalInput, index: int):
    instruction_id = inp.instruction_id_list[index]
    instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
    instruction = instruction_cls(instruction_id)
    instruction.build_description(**inp.kwargs[index])
    args = instruction.get_instruction_args()
    if args and "prompt" in args:
        instruction.build_description(prompt=inp.prompt)
    return instruction


def _loose_candidates(response: str) -> tuple[str, ...]:
    lines = response.split("\n")
    remove_first = "\n".join(lines[1:]).strip()
    remove_last = "\n".join(lines[:-1]).strip()
    remove_both = "\n".join(lines[1:-1]).strip()
    revised = response.replace("*", "")
    return (
        response,
        revised,
        remove_first,
        remove_last,
        remove_both,
        remove_first.replace("*", ""),
        remove_last.replace("*", ""),
        remove_both.replace("*", ""),
    )


def evaluate_response(inp: IFEvalInput, response: str) -> IFEvalResult:
    """Run the official strict and loose instruction checks for one response."""

    if len(inp.instruction_id_list) != len(inp.kwargs):
        raise ValueError("instruction_id_list and kwargs must have equal lengths")
    unknown = set(inp.instruction_id_list) - set(instructions_registry.INSTRUCTION_DICT)
    if unknown:
        raise ValueError(f"unknown IFEval instruction IDs: {sorted(unknown)}")

    strict: list[bool] = []
    loose: list[bool] = []
    candidates = _loose_candidates(response)
    for index in range(len(inp.instruction_id_list)):
        strict_instruction = _instruction(inp, index)
        strict.append(
            bool(response.strip() and strict_instruction.check_following(response))
        )

        # Upstream constructs a fresh instruction for the loose arm.
        loose_instruction = _instruction(inp, index)
        loose.append(
            any(
                candidate.strip() and loose_instruction.check_following(candidate)
                for candidate in candidates
            )
        )

    return IFEvalResult(strict=tuple(strict), loose=tuple(loose))
