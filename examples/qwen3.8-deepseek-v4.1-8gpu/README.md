# DeepSeek-V4.1-Flash (6 GPUs) + Qwen3.8-27B (2 GPUs): judged five routes, DeepSeek-led ensemble

One OpenAI-compatible API and one Chat UI over eight RTX PRO 6000 Blackwell
cards. Kairyu's existing route judge (Qwen, non-thinking) picks one of five
routes per request; the ensemble route is a DeepSeek-led critical process
that produces exactly one published answer. Kairyu, the sibling examples, and
the shared scripts are unchanged: everything here is example-owned
configuration plus launch/verification tooling.

```
                       ┌──────── Kairyu L3 (kairyu-auto-max, kairyu-ensemble-max, embed-small)
Open WebUI ── API ─────┤
                       │  L2: route judge (Qwen) -> profile
                       │      qwen_direct | qwen_think_medium | deepseek_direct | deepseek_think | primary
                       │
        ┌──────────────┴──────────────┐
  qwen-0 (GPU 6)  qwen-1 (GPU 7)      deepseek (GPUs 0-5, one vLLM service)
  Qwen3.8-27B-FP8 TP1 x 2             DeepSeek-V4.1-Flash, TP1 x attention-DP6, EP6 (candidate 3)
```

## Status

**GPU gates in progress (2026-09-14).** CPU contracts pass. The DeepSeek
runtime is this example's own overlay image (`vllm-sm120.Dockerfile` on the
sibling `deepseek-v4.1-flash-8gpu` SM120 overlay): PR #598's masked
sparse-KV zero-row fix and seeded top-p terminator fix, which `run.sh up`
builds from the files here when the tag is absent. Without it every six-GPU
topology corrupts output from the second request per engine onward
(`MEASUREMENTS.md`, candidates 1–5). Topology: TP2 × attention-DP3, EP6,
Engram tables in pinned host memory, 0.90 utilization, 16384 batched tokens,
32 sequences — the configuration PR #598 measured. Gate results are recorded
in `MEASUREMENTS.md` as they complete.

## The five routes (unchanged judge, DTO-D13)

| Judge label | Profile | Worker | Thinking | Sampling | Output limit |
|---|---|---|---|---|---|
| `QWEN` | `qwen_direct` | Qwen | off | 0.7 / 0.8 / top_k 20 / presence 1.5 | caller's `max_tokens`, else the model's remaining context |
| `QWEN_THINK` | `qwen_think_medium` | Qwen | medium (spec `high`) | 1.0 / 0.95 / top_k 20 | caller's `max_tokens`, else the model's remaining context |
| `DEEPSEEK` | `deepseek_direct` | DeepSeek | off (`enable_thinking=false`) | 1.0 / 0.95 | min(caller, 393216) |
| `DEEPSEEK_THINK` | `deepseek_think` | DeepSeek | caller's effort (default high = 75) | 1.0 / 0.95 | min(caller, 393216) |
| `ENSEMBLE` | `primary` | both | see below | per role | caller's `max_tokens` shared by head + remainder |

Judge timeout, backend error, or an unparseable verdict fall back to
`primary`. Both pools accept images, so every route is offered on image
requests.

The Qwen routes no longer declare a fixed 131072-token `max_tokens`
(Issue #599): with a fixed value, a long tool-bearing conversation whose
rendered input already exceeded 131,072 tokens made vLLM reject the request
(input + 131,072 > 262,144) and Kairyu returned 502. Without a role cap,
Kairyu sends the caller's `max_tokens` when one is given and omits the field
otherwise, and vLLM fits generation to the remaining context. The Qwen
medium route keeps its official sampling, medium thinking, and template; the
change is only the fixed cap. This deviates from the "131072" figure in the
original requirement on purpose; see V41T-D4 in the design document.

## The ensemble route (`primary`, V41T-D2)

| Wave | Role | Worker | Thinking | Reads | Cap (tokens) |
|---|---|---|---|---|---|
| 1 | `head` | Qwen | off | conversation | 256, streamed to the user from t=0 |
| 1 | `requirements` | DeepSeek | caller's effort, default high | conversation | 16384 / 32768 / 65536 by effort |
| 1 | `independent` | DeepSeek | caller's effort | conversation, tools, images only | 16384 / 32768 / 65536 |
| 1 | `policies` | DeepSeek | caller's effort | conversation | 8192 / 8192 / 16384 (bounded so the Qwen answerers' input fits; see below) |
| 2 | `answer_1..4` | Qwen (2 replicas, 2 + 2) | medium | conversation + policies, one `POLICY n` each | 16384 |
| 3 | `synthesis` | DeepSeek | caller's effort | conversation + all 5 candidates | 16384 / 65536 / 131072 |
| 4 | `final` | DeepSeek | caller's effort | conversation + all 5 candidates + proposal + committed head | caller's limit minus the head |
| 4 | `audit` | DeepSeek (verifier) | caller's effort | head + final + checklist (the only reader of the checklist) | 8192 / 16384 / 32768 |

- `requirements` emits the PR #595 checklist: a bare JSON array of
  `{id, priority, requirement, acceptance_criterion, source}` objects with
  `minimum`/`optional` priorities, literal strings and numbers preserved. It
  is verification data: only the `audit` reads it (owner decision 2026-09-15);
  no role that produces the answer sees it, so the answer is judged against
  criteria it was not written to. The audit recovers the requirements from
  the request when the checklist is empty (a failed upstream call renders its
  slot empty). `requirements` stays a scheduling dependency of `final` only
  because Kairyu runs the verifier inline after its target and requires every
  verifier input to be complete by then.
- The Qwen answerers' input is bounded by configuration, not by a per-request
  calculation (Kairyu has none): rendered conversation + policies text +
  their own 16,384-token budget must fit Qwen's 262,144 context, so the
  `policies` cap is 8,192 (16,384 at `max` effort; DeepSeek completion tokens
  include thinking, so the cap also bounds the text). Guaranteed rendered
  conversation length for the ensemble: about 237,000 tokens at default/low
  effort and about 229,000 at `max` (the head's 256 and the judge's 8 never
  bind first).
- `policies` writes four policies that differ in method, assumptions, and
  evaluation criteria, not wording.
- `synthesis` checks premises, evidence, methods, and the conditions under
  which each candidate's conclusion holds; looks for counterexamples,
  boundary conditions, omissions, and errors common to all five; fixes and
  combines; may take an approach no candidate took; and writes the complete
  proposal followed by an internal `=== DECISION RECORD ===` (reasons and
  references). Agreement among candidates is not evidence.
- `final` re-checks the proposal against the request and all candidates,
  continues directly after the committed opening (or writes the
  complete answer on tool / structured-format turns, where the head is
  disabled), and keeps its adopt/reject decisions in private reasoning.
- `audit` outputs `PASS` or `FAIL` on the first line, then one line per
  requirement ID (`R1 | status | evidence | correction`). Unmet or
  unverifiable minimum requirements fail; optional improvements alone do
  not. `FAIL` text is appended verbatim to `final`'s re-dispatch; at most
  two refinements (`max_refine_depth: 2`), each re-audited by a separate
  call. An unparseable verdict triggers one bounded re-audit that does not
  count as a refinement. After the second failed refinement the last attempt
  is published (Kairyu's existing exhaustion policy); the verdict is kept in
  the trace.
- Budget: `max_steps: 19` = 10 generation calls + 1 empty-output re-dispatch
  + 3 audits + 3 bounded re-audits + 2 refinements. `internal_max_tokens:
  131072` admits the largest role tier; the caller's `max_tokens` still caps
  every internal role (Kairyu contract), so the Chat UI's 65536 is the
  effective ceiling from the UI.

`kairyu-ensemble-max` serves the same DAG without the judge (V41T-D5; it also
omits `public_output_floor`, which Kairyu only accepts when some final unit
declares a `reasoning_close_tag` — in `auto-max.yaml` that is the Qwen
thinking route) so
`verify.sh serving-ensemble` can prove the full flow on every request. It is
API-only; the Chat UI lists `kairyu-auto-max` alone.

## DeepSeek V4.1 generation settings (official sources)

| Setting | Value | Source |
|---|---|---|
| temperature / top_p | 1.0 / 0.95 (thinking and non-thinking alike; the card lists no per-mode difference) | model card: "Recommended sampling parameters: temperature 1.0, top_p 0.95 or 1.0" — https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash |
| reasoning effort | request field `reasoning_effort`; the checkpoint encoder maps `low → 50`, `high → 75` (default), `max → 100` and renders `Reasoning Effort: N` | `encoding/README.md` in the checkpoint — https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/encoding/README.md |
| non-thinking | `chat_template_kwargs.enable_thinking=false` (the encoder's `thinking_mode: chat`, an immediately closed span). Kairyu sends it for every role that declares no effort | same encoder README; pinned vLLM `tokenizers/deepseek_v41.py` reads `thinking or enable_thinking` |
| context / max output | 1,048,576 / 393,216 | model card ("1M context", "384K max output") |

The sibling's SM120 overlay (built by `../deepseek-v4.1-flash-8gpu`) aligns
vLLM's effort aliases with the checkpoint's table; this example's overlay
builds on it. `compose.yaml` sets **no**
`--default-chat-template-kwargs`: the encoder thinks at high on its own,
and a server-side `thinking: true` default would override the
`enable_thinking=false` that non-thinking roles send. `verify.sh native`
renders both modes through `/tokenize` and checks the effort budgets 50/75/100.

## Qwen medium (unchanged) and why the L1 template file exists

Qwen3.8-27B-FP8 at revision `017b9c7a…`, thinking sampling 1.0 / 0.95 /
top_k 20, non-thinking 0.7 / 0.8 / top_k 20 / presence 1.5.

`l1-qwen3.8-27b-vllm-chat-template.jinja` is the L1 (vLLM) chat template of
the Qwen worker — copied byte-for-byte from the tiered example — and has
nothing to do with the Chat UI, which talks only to Kairyu L3. It exists
because of a vocabulary mismatch: the checkpoint's own template accepts
`reasoning_effort` values `low` / `medium` / `xhigh` only and raises on
anything else, while Kairyu can send Qwen only `low` / `high` / `max` (the
API maps `medium` to `high`). Without an example-owned template every Qwen
thinking call would be rejected by vLLM. The copied template maps `high`
to the medium tier (a short "reason carefully" preamble, DTO-D14) and
clamps `max` to `high`; `low` renders the stock prompt unchanged. Every
Qwen example in this repository uses the same `--chat-template` mechanism.

## What the conversation looks like to the roles

Kairyu L3 renders the conversation once: a role-tagged JSON of every message
(system, developer, user, assistant, tool; empty values, prior
`reasoning_content`, tool calls with ids and argument strings, tool results,
images as `<image:N>` markers) followed by a plain-text view of the latest
user turn. Every role prompt places that rendering first (`{query}`, once),
so the DeepSeek service's prefix cache reuses the long-conversation prefill
across its six calls. Images are forwarded as the original media to every
role on both pools; nothing is summarized or replaced by a description.

## Start

```sh
./examples/qwen3.8-deepseek-v4.1-8gpu/run.sh          # up (default) | down | status | logs
```

`run.sh up` checks the eight GPUs, pulls the pinned Qwen image, builds the
sibling's DeepSeek overlay only if the tag is absent (then fails closed on any
image-ID mismatch), attests both checkpoints against the sibling examples'
manifests (no re-download when the attestation matches), starts Compose,
and validates: readiness, the public model set `{kairyu-auto-max,
kairyu-ensemble-max, embed-small}`, both served L2 policies against
`example.json`, the embedding contract, the DeepSeek tokenizer oracle, and
the Chat UI reasoning-effort dropdown (default/low/high/max).

Required: `KAIRYU_RESPONSES_COMPACTION_SECRET` (32+ random bytes),
`/mnt/nvme/kairyu` storage, the sibling checkpoints under
`model-volumes/qwen3.8-27b-1gpu/models` and
`model-volumes/deepseek-v4.1-flash-8gpu/models`, `PUBLIC_HOST` when the
outward interface cannot be detected. Ports: API 8008, Chat UI 3008, DeepSeek
L1 loopback 8009 (baseline rows and the token oracle only).

Effort from the Chat UI: Chat Controls → Valves → Reasoning Effort. `default`
leaves the field unset (DeepSeek roles then use high; Qwen roles keep their
fixed declarations); `low`/`high`/`max` reach every `inherit` role, including
`requirements` (V41T-D3: an explicit `low` is honoured — the DSL has no
"floor at high" setting).

## Verify

```sh
./examples/qwen3.8-deepseek-v4.1-8gpu/verify.sh list
./examples/qwen3.8-deepseek-v4.1-8gpu/verify.sh native            # L1 candidate gates
./examples/qwen3.8-deepseek-v4.1-8gpu/verify.sh serving-auto-max  # judged, generic, c1/8/16/32 x 32
./examples/qwen3.8-deepseek-v4.1-8gpu/verify.sh serving-auto-max-coding
./examples/qwen3.8-deepseek-v4.1-8gpu/verify.sh serving-ensemble  # forced five-candidate flow
./examples/qwen3.8-deepseek-v4.1-8gpu/verify.sh tool-calling | vision | cancellation | restart | long-input | browser
ISSUE_599_REQUEST_PATH=... ./examples/qwen3.8-deepseek-v4.1-8gpu/verify.sh issue-599
```

Every serving row records, from the traces: routes and per-route time to
first visible content and completion, judge time, per-stage queue wait and
duration, per-stage completion tokens against the role caps (a stage that
ends at its cap fails the row, because the trace carries no per-stage
finish reason), audit verdicts and refinement counts, Qwen placement over
the two replicas, GPU/host memory peaks, and usage including discarded
candidates. Each row is paired with a DeepSeek-direct row on the same
service at the same concurrency and effort; the gate is product time to
first visible content (judge included) ≤ 2× the paired row, and the paired
row is only a valid denominator when all 32 requests completed with
`finish_reason: stop` and visible content.

## Known limits (Kairyu `main`, recorded, not hidden)

1. No windowed reading of inputs that exceed a model's context: Qwen-involving
   routes fail with an upstream 400 (surfaced as Kairyu's generic 502) once
   the rendered input exceeds the bound above (about 237,000 tokens for the
   ensemble at default effort, 229,000 at `max`); DeepSeek-only routes reach
   1,048,576.
2. The upstream 400 reason is replaced by "orchestration final unit produced
   no public output" (502); the actual reason is in the gateway log.
3. L3 appends `<image:N>` directly to the user text in the rendered
   conversation and repeats the latest user turn as a plain-text view;
   exact-ending requirements next to an image can be misread by extraction.
   DeepSeek roles also receive the real image, and the audit is told the
   checklist is fallible.
4. If the final unit ends its thinking span without public text twice in a
   row, Kairyu returns 502 (or the head alone when a head was streamed).
   The DTO-D9 floor is inert for the DeepSeek final until the native-chat
   assistant-prefill continuation is GPU-verified; the Qwen thinking route
   keeps it.
5. A client `max_tokens` smaller than the internal role tiers clamps every
   internal role to that value (framework contract); the Chat UI sends 65536.
6. `requirements` output is prompt-constrained JSON, not grammar-constrained.
7. Qwen candidates have no separate thinking budget; the 16384-token cap
   bounds thinking + answer, and a cap hit fails the serving row.

## Reproducibility pins

- Checkpoints: Qwen `Qwen/Qwen3.8-27B-FP8` @ `017b9c7a…` (tree
  `9825ce11…`); DeepSeek `deepseek-ai/DeepSeek-V4.1-Flash` @ `dba1be0a…`
  (tree `d21211ca…`). Both attested by the sibling examples' manifests.
- Images: Qwen `vllm/vllm-openai:v0.23.0@sha256:6d8429e3…`; DeepSeek
  `local/vllm-openai:deepseek-v41-sm120-masked-kv-budget` = this directory's
  `vllm-sm120.Dockerfile` (+ `patch_masked_kv.py`, `patch_top_p.py`, SHA-256
  pinned in `example.json`) on the sibling's `local/vllm-openai:deepseek-v41-sm120`
  (vLLM `179dd0fa9`, FlashInfer `60b49158`). The reference build measured here
  has ID `sha256:18dad57d…`; another host rebuilds the same recipe and records
  its own ID (`run.json` → `runtime`). Open WebUI `v0.11.0-slim@sha256:3698bd4e…`.
- Served-config hash: `verification.py` hashes `example.json`,
  `compose.yaml`, `kairyu.yaml`, `auto-max.yaml`, `ensemble-max.yaml`,
  `router.json`, `l1-qwen3.8-27b-vllm-chat-template.jinja`, `webui-reasoning-effort-filter.py`,
  and `benchmark.py`, `vllm-sm120.Dockerfile`, and both patch scripts into every `run.json`, and attests the running images,
  commands, and mounted files before measuring.
