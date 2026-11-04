# Gemma 4 31B example: serve an edited chat checkpoint

Status: **Plan approved 2026-11-04 (owner), including the verification plan.**
Based on `main` only. Part of the owner's weight-space programming (WSP)
benchmark; the benchmark itself is built in kairyu-bench (plan
`docs/superpowers/plans/2026-11-04-wsp-benchmark.md` there).

## Request

- A common benchmark that measures, in 10, 30 and 60 minutes of GPU time, what
  an edit of a model's weights gained and lost. Time reference (owner): Gemma 4
  31B on eight GPUs, i.e. this example's layout (four TP2 chat replicas, MTP 4,
  the example's vLLM v0.33.0 overlay, official sampling, thinking on).
- An edited Gemma 4 31B checkpoint must therefore start in exactly that layout.
  Owner decision (2026-11-04): add one setting to `gemma-4-31b-8gpu` that
  replaces the chat checkpoint's location.

## Facts

1. The chat replicas load `/models/gemma-4-31b-it`, the pinned download inside
   the example's model volume (`compose.yaml` `x-gemma`). `run.sh up` downloads
   or checks it against `example.json` `model.tree_sha256` and refuses anything
   else (`control.py` `_MODEL_PROGRAM`, `_ensure_models`).
2. vLLM serves any complete Hugging Face checkpoint directory; the served name,
   Kairyu's pool (`kairyu.yaml`) and the Chat UI only know `gemma-4-31b`.
3. Speculative decoding verifies every drafted token, so the official MTP
   assistant keeps the edited model's output distribution (only its speed can
   change).
4. Precedent pair on this host: `chinhnc/Quyet-1.0-Large` is Gemma-4-31B-it
   with a merged rank-16 LoRA, attested at tree
   `8f287d92…fb332a05` under the Quyet example's model volume.

## Change (example-owned; `kairyu/` unchanged)

- `CHAT_CHECKPOINT_DIR=<absolute directory under /mnt/nvme>`: when set, the four
  chat replicas load that directory instead of the pinned checkpoint. It must
  hold a complete checkpoint (`config.json`, safetensors, tokenizer, chat
  template, `generation_config.json`; the served sampling defaults come from
  it). The MTP assistant, EmbeddingGemma 2, the served name, the layout and all
  L1 flags stay as they are.
- Identity: the pinned hash check is skipped for that directory only; `run.sh up`
  computes its tree hash with the attestation's algorithm, writes
  `model-volumes/gemma-4-31b-8gpu/chat-checkpoint.json` (path, tree hash, files)
  and prints the hash. Unset, the example behaves exactly as today.
- Mechanics: every chat replica mounts its checkpoint read-only at
  `/checkpoint`; the source defaults to the pinned directory.
- Review amendment (2026-11-04, owner-approved): Compose does not recreate a
  replica whose mount, command and image are unchanged, so new weights in the
  same directory kept the old ones on the GPUs while the record named the new
  hash. Every chat replica now carries the tree hash as the label
  `kairyu.chat-checkpoint-tree` (Compose recreates it when the hash changes),
  and `run.sh up` writes the record only after each running replica carries
  that hash. `verify.sh long-context` reads `tokenizer.json` from the recorded
  served checkpoint instead of the pinned download, which may be absent.
- Docs: README section; FN-D9 Gemma amendment; PROGRESS entry.
- Tests: none added. The mount and the record are validated by `run.sh up`
  (absolute NVMe path, directory with `config.json` and safetensors); the
  end-to-end check is the GPU step below.

GPU-stage amendment (2026-11-04, owner-approved): with Quyet-1.0-Large's
weights the replicas started and carried the new label, but `run.sh up` failed
its readiness check: the arithmetic probe (exact `323`, normal end, 8 requests)
hit the 8,192-token cap on at least one request; a manual request answered
`323` after garbled reasoning. Readiness checked the model's answers, which is
what the benchmark measures for an edited checkpoint. With
`CHAT_CHECKPOINT_DIR` set, readiness is now serving only (every chat replica
returns a reply, any content or finish; embeddings as usual), and
`verify.sh long-context` judges acceptance and token counts at every length,
recording the returned key. The official checkpoint keeps every check.

## Verification (GPU, after review)

1. `run.sh up` with `CHAT_CHECKPOINT_DIR` pointing at a directory that holds the
   official non-weight files and Quyet-1.0-Large's safetensors (hard links on
   the same NVMe): every chat replica becomes healthy, the record's file hashes
   for the two safetensors equal Quyet's attestation, and one chat request is
   answered. The WSP benchmark's KL diagnostic against the official run is
   non-zero (the edited weights are served).
2. With the setting, replace the safetensors in that directory and run
   `run.sh up` again: the four chat replicas are recreated, and their labels
   and the record carry the new tree hash. `verify.sh long-context` passes
   with the edited checkpoint served: every length is accepted with the
   server's token count at its target (the returned keys are recorded).
3. `run.sh up` without the setting: the pinned checkpoint is checked and served
   again (the official record), `serving` gate passes. The first `up` after
   this change recreates the chat replicas once (their configuration gains
   the label).

Verification amendment (2026-11-04, owner): only what was fixed is checked.
Steps 2 and 3 and the KL diagnostic are dropped.

- Edited checkpoint (the serving-only readiness fixed after the GPU-stage
  failure): `run.sh up` with `CHAT_CHECKPOINT_DIR` holding Quyet-1.0-Large's
  safetensors finished in 189 s with every chat replica healthy and
  answering; the record's safetensors hashes (`7e06d4a9…`, `f06a5e11…`) equal
  Quyet's attestation, and the replicas' labels carry the record's tree hash.
- Client disconnect (#670, fixed in this PR): with the fixed server, a
  request to `/v1/chat/completions`, `/v1/messages` and `/v1/responses`
  dropped by the client after 15 s left no generation running on any replica
  10 s later (before the fix it ran until the reply was complete).
