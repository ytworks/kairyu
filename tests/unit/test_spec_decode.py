from kairyu.engine.core.spec_decode import propose_ngram, verify_greedy


def test_repeating_context_proposes_continuation():
    # suffix (2, 3) occurred earlier, followed by 4, 5
    context = (1, 2, 3, 4, 5, 9, 2, 3)
    assert propose_ngram(context, max_draft=2, max_ngram=2) == (4, 5)


def test_prefers_longest_matching_ngram():
    # suffix (2, 3) matches at two places; 3-gram (9, 2, 3) disambiguates to 7
    context = (2, 3, 4, 9, 2, 3, 7, 8, 9, 2, 3)
    assert propose_ngram(context, max_draft=1, max_ngram=3) == (7,)


def test_no_repetition_proposes_nothing():
    assert propose_ngram((1, 2, 3, 4, 5), max_draft=4) == ()


def test_short_context_is_safe():
    assert propose_ngram((), max_draft=4) == ()
    assert propose_ngram((1,), max_draft=4) == ()


def test_verify_full_acceptance_includes_bonus_token():
    result = verify_greedy(draft=(4, 5, 6), target_tokens=(4, 5, 6, 7))
    assert result.accepted == 3
    assert result.tokens == (4, 5, 6, 7)


def test_verify_partial_acceptance_stops_at_first_mismatch():
    result = verify_greedy(draft=(4, 9, 6), target_tokens=(4, 5, 6, 7))
    assert result.accepted == 1
    assert result.tokens == (4, 5)  # accepted draft + target's correction


def test_spec_decode_output_equals_plain_greedy_decoding():
    """The core invariant: speculation never changes the generated sequence."""

    def greedy_target(prefix: tuple[int, ...]) -> int:
        # deterministic fake LM with enough structure for n-gram hits
        return (prefix[-1] * 31 + len(prefix)) % 7

    def plain_decode(prompt: tuple[int, ...], steps: int) -> tuple[int, ...]:
        sequence = prompt
        for _ in range(steps):
            sequence = (*sequence, greedy_target(sequence))
        return sequence[len(prompt):]

    def spec_decode(prompt: tuple[int, ...], steps: int) -> tuple[int, ...]:
        sequence = prompt
        while len(sequence) - len(prompt) < steps:
            draft = propose_ngram(sequence, max_draft=3)
            targets = []
            scoring_prefix = sequence
            for token in (*draft, None):  # target scores draft positions + bonus
                targets.append(greedy_target(scoring_prefix))
                if token is None or targets[-1] != token:
                    break
                scoring_prefix = (*scoring_prefix, token)
            result = verify_greedy(draft[: len(targets) - 1], tuple(targets))
            sequence = (*sequence, *result.tokens)
        return sequence[len(prompt) : len(prompt) + steps]

    prompt = (3, 1, 4, 1, 5, 9, 2, 6)
    assert spec_decode(prompt, steps=40) == plain_decode(prompt, steps=40)
