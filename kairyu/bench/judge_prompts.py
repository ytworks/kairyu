"""Pinned judge prompt templates. Changing these changes scores — treat as frozen.

The HLE template follows the official HLE judging format (extracted answer,
then a strict `correct: yes|no` field the client parses). The CharXiv
template adapts the official grading rubric to the same verdict field.
"""

from __future__ import annotations

import hashlib


def _ordered_blocks(header: str, first: str, second: str, rubric: str) -> str:
    """Compose a rubric while making the two calibrated block positions explicit."""
    return "\n\n".join((header, first, second, rubric))


_HLE_HEADER = """\
Judge whether the following [response] to [question] is correct or not based on
the precise and unambiguous [correct_answer] below.

[question]: {question}"""
_HLE_RESPONSE_BLOCK = "[response]: {response}"
_HLE_EXPECTED_BLOCK = "[correct_answer]: {expected}"
_HLE_RUBRIC = """\
Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response].
Put 'None' if there is no exact, final answer to extract from the response.

reasoning: Explain why the extracted_final_answer is correct or incorrect based
on [correct_answer], focusing only on if there are meaningful differences.
Do not comment on any background to the problem, do not attempt to solve the
problem, do not argue for any answer different than [correct_answer].

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer]
given above, or is within a small margin of error for numerical problems.
Answer 'no' otherwise.
"""

HLE_JUDGE_TEMPLATE = _ordered_blocks(
    _HLE_HEADER,
    _HLE_RESPONSE_BLOCK,
    _HLE_EXPECTED_BLOCK,
    _HLE_RUBRIC,
)

_CHARXIV_HEADER = """\
You are grading an answer to a chart-reasoning question.

Question: {question}"""
_CHARXIV_EXPECTED_BLOCK = "Ground-truth answer: {expected}"
_CHARXIV_RESPONSE_BLOCK = "Model response: {response}"
_CHARXIV_RUBRIC = """\
Grade the response ONLY on whether its final answer matches the ground truth.
Numerical answers may differ by rounding in the last shown digit; text answers
must match in meaning (case and articles do not matter). Extraneous
explanation does not make a correct answer wrong.

Reply in exactly this format:

reasoning: <one or two sentences>
correct: <yes or no>
"""

CHARXIV_JUDGE_TEMPLATE = _ordered_blocks(
    _CHARXIV_HEADER,
    _CHARXIV_EXPECTED_BLOCK,
    _CHARXIV_RESPONSE_BLOCK,
    _CHARXIV_RUBRIC,
)

# Calibration renders the same labelled fields in the opposite reference /
# response order. The production templates above remain unchanged; comparing
# their verdicts with these counterbalanced variants measures whether block
# position alone flips the judge.
HLE_CALIBRATION_SWAPPED_TEMPLATE = _ordered_blocks(
    _HLE_HEADER,
    _HLE_EXPECTED_BLOCK,
    _HLE_RESPONSE_BLOCK,
    _HLE_RUBRIC,
)

CHARXIV_CALIBRATION_SWAPPED_TEMPLATE = _ordered_blocks(
    _CHARXIV_HEADER,
    _CHARXIV_RESPONSE_BLOCK,
    _CHARXIV_EXPECTED_BLOCK,
    _CHARXIV_RUBRIC,
)


def judge_templates() -> dict[str, str]:
    """Return every production grading template from one live registry."""
    return {
        "hle": HLE_JUDGE_TEMPLATE,
        "charxiv-reasoning": CHARXIV_JUDGE_TEMPLATE,
    }


def judge_template(name: str, *, counterbalanced: bool = False) -> str:
    """Resolve a production template or its calibration-only order variant."""
    if not counterbalanced:
        try:
            return judge_templates()[name]
        except KeyError as error:
            raise ValueError(f"unknown judge template {name!r}") from error
    swapped = {
        "hle": HLE_CALIBRATION_SWAPPED_TEMPLATE,
        "charxiv-reasoning": CHARXIV_CALIBRATION_SWAPPED_TEMPLATE,
    }
    try:
        return swapped[name]
    except KeyError as error:
        raise ValueError(f"unknown judge template {name!r}") from error


def judge_template_identity(name: str, *, counterbalanced: bool = False) -> dict[str, str]:
    """Content identity recorded by runs and calibration artifacts."""
    template = judge_template(name, counterbalanced=counterbalanced)
    return {
        "name": name,
        "variant": "counterbalanced" if counterbalanced else "production",
        "sha256": hashlib.sha256(template.encode("utf-8")).hexdigest(),
    }


def judge_prompt_identities() -> dict[str, dict[str, str]]:
    """Content identities for all production judge prompts."""
    return {name: judge_template_identity(name) for name in judge_templates()}
