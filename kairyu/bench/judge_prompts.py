"""Pinned judge prompt templates. Changing these changes scores — treat as frozen.

The HLE template follows the official HLE judging format (extracted answer,
then a strict `correct: yes|no` field the client parses). The CharXiv
template adapts the official grading rubric to the same verdict field.
"""

HLE_JUDGE_TEMPLATE = """\
Judge whether the following [response] to [question] is correct or not based on
the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

[correct_answer]: {expected}

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

CHARXIV_JUDGE_TEMPLATE = """\
You are grading an answer to a chart-reasoning question.

Question: {question}

Ground-truth answer: {expected}

Model response: {response}

Grade the response ONLY on whether its final answer matches the ground truth.
Numerical answers may differ by rounding in the last shown digit; text answers
must match in meaning (case and articles do not matter). Extraneous
explanation does not make a correct answer wrong.

Reply in exactly this format:

reasoning: <one or two sentences>
correct: <yes or no>
"""
