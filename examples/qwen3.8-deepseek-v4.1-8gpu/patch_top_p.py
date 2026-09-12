"""Pinned example-only split top-p safeguard; never changes the forcing logit.

Mirror the monolithic sampler's cutoff guard: rounding a cutoff to the maximum
(or producing NaN) must not mask every token under the strict > comparison.
"""

import hashlib
from pathlib import Path

TARGET = "/usr/local/lib/python3.12/dist-packages/vllm/v1/sample/ops/topk_topp_triton.py"
SOURCE_SHA256 = "22c526541d72f3b66f4f10b989c103c0869ea501aae2ac05151782dd69469ee1"
PATCHED_SHA256 = "13ef1b8ec91340ad7a9402e9ec66f605a7117b13c4ce287df0de661e77f0346f"
ANCHOR = """    logZ = tl.log(Z)
    pivot_logit = tl.log(pivot) + logZ + M
    # numdup/numkeep are exact small integers held in fp32.
"""
REPLACEMENT = """    logZ = tl.log(Z)
    pivot_logit = tl.log(pivot) + logZ + M
    # Match the monolithic guard: strict > must retain a candidate even when
    # a large forced logit rounds the cutoff to M (or the cutoff is NaN).
    if not (pivot_logit < M):
        pivot_logit = -float("inf")
    # numdup/numkeep are exact small integers held in fp32.
"""


def digest(source: bytes) -> str:
    return hashlib.sha256(source).hexdigest()


def transform(source: str) -> str:
    if source.count(ANCHOR) != 1:
        raise ValueError("Expected exactly one pinned split top-p cutoff anchor")
    return source.replace(ANCHOR, REPLACEMENT, 1)


def patch(path: Path = Path(TARGET)) -> bool:
    source = path.read_bytes()
    actual = digest(source)
    if actual == PATCHED_SHA256:
        return False
    if actual != SOURCE_SHA256:
        raise ValueError(f"Unrecognized split top-p source: {actual}")
    output = transform(source.decode()).encode()
    if digest(output) != PATCHED_SHA256:
        raise ValueError("Unexpected patched split top-p source hash")
    path.write_bytes(output)
    return True


if __name__ == "__main__":
    changed = patch()
    print(f"split top-p guard: {'patched' if changed else 'already patched'} {PATCHED_SHA256}")
