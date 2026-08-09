# Long-context accuracy sweep (issue #374)

Status: implemented

## Decision

Kairyu has a separate `long-context` benchmark suite whose six rows are one
deterministic single-key needle-retrieval task at 4K, 8K, 16K, 32K, 64K, and
128K. Length is the row dimension, so the existing scoreboard is the requested
accuracy-vs-length curve. No special curve artifact, runner, history format, or
comparison implementation is added.

This is RULER-style, not an implementation or result of the official 13-task
NVIDIA RULER suite. It adopts RULER's configurable sequence-length principle
and its standard 4K-to-128K reporting points, while keeping one deliberately
small retrieval task. Reports state that boundary and must not compare these
scores with official RULER numbers.

## Population and scoring

Every length contains 20 items. Target needle depths are the midpoint of each
5% interval (2.5%, 7.5%, ..., 97.5%), covering early, middle, and late
positions without random population drift. Keys and values are distinct,
deterministic SHA-256-derived identifiers. One value occurs once in a repeated
text haystack, the query repeats only its key, and the model is instructed to
return only the value. Stripped exact equality is the binary score.

The local generator uses `o200k_base`, constructs the full user-message
content, and recounts it before accepting exactly the requested token length.
The synthetic source revision, generator/scorer source, shared runner, and
aggregation code are content-bound by the existing cache and run-fingerprint
contracts. The actual needle offset and requested depth are retained in each
normalized row. Generation failure is dataset-unavailable evidence, never an
approximate-length fallback.

`o200k_base` measures user-message content only. An endpoint's model tokenizer
and chat-template overhead cannot be attested through the OpenAI-compatible
chat API, so every pair discloses that limitation. The curve is suitable for
same-target/config regression diagnosis; it is not tokenizer-independent proof
of an advertised context window.

## Execution and gates

`BenchTarget.max_context_tokens`, including the public
`--max-context-tokens` target override, gates whole rows. Points above the
declared limit are skipped and never truncated. An absent declaration attempts
all six points. The short-answer budget is capped at 32 tokens.

A full run makes 120 requests per target through the ordinary retry,
concurrency, result, and history paths. `--limit`, `--smoke`, and installed
offline fixtures retain the existing subset/fixture incomparability rules.
Clean full runs are directly usable by `compare-runs`; paired configuration
gates use the existing `compare` command with tolerances for the six row names.
This makes FP8-KV calibration and RoPE-scaling changes measurable without
creating another evidence protocol.
