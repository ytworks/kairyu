# Discarded issue #333 v1 trial

This complete work directory is retained for transparency only. It is not the
issue result and supports no process/GIL-contention classification because the
v1 manifest has `evidence_valid: false`.

- Source commit: `ad11a322b77337547ac34dc1717586c40f76fd8b`
- Invalidating v1 check: `completion_output_hash_parity_across_all_four_cells`
- Observed paired-median process/in-process TTFT-p99 ratio: `0.9454156693989547`
- Raw SHA-256: `6b726d0076f55226b9d50370a27f8f06e486ca9c53088192f003891507b95a13`
- Manifest SHA-256: `15526fc3efec8b5a12ae5dba3e7c5c54c9cc3729987e005dd5a838cf8f94d393`

The two same-arm measurement repeats agreed on only 29/128 and 41/128 output
hashes. In ABBA order, all six pair counts were 00–01: 38/128, 00–02: 39/128,
00–03: 29/128, 01–02: 41/128, 01–03: 20/128, and 02–03: 23/128; only 7/128
sequences matched across all four cells. The v2 amendment therefore replaces
all-four measurement parity with
arm-neutral binding checks and reports all six measurement pair rates without
making them binding. The performance interpretation line remains unchanged.

Replay this artifact with `bench/issue_333_proc_http_bench.py verify` from the
source commit above. Replay succeeds and reproduces the invalid manifest;
`--assert-integrity` is expected to fail. The retained raw rows, shards,
configuration, state, and logs must never be rewritten, resumed, or reused as
v2 evidence. Exactly one entirely fresh v2 ABBA run is the issue result,
regardless of its performance direction.
