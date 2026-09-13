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
  Qwen3.8-27B-FP8 TP1 x 2             DeepSeek-V4.1-Flash, TP2 x attention-DP3, EP6
```

## Status

CPU contracts only. **No GPU evidence exists for this example yet**; the
6-GPU DeepSeek topology in `compose.yaml` is the first candidate of the
selection procedure in `MEASUREMENTS.md`, not a verified configuration. Do
not reuse the sibling examples' measurements for this topology.

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
| 2 | `policies` | DeepSeek | caller's effort | conversation + checklist | 8192 / 32768 / 65536 |
| 3 | `answer_1..4` | Qwen (2 replicas, 2 + 2) | medium | conversation + checklist + policies, one `POLICY n` each | 16384 |
| 4 | `synthesis` | DeepSeek | caller's effort | conversation + checklist + all 5 candidates | 16384 / 65536 / 131072 |
| 5 | `final` | DeepSeek | caller's effort | everything above + committed head | caller's limit minus the head |
| 5 | `audit` | DeepSeek (verifier) | caller's effort | head + final + checklist | 8192 / 16384 / 32768 |

- `requirements` emits the PR #595 checklist: a bare JSON array of
  `{id, priority, requirement, acceptance_criterion, source}` objects with
  `minimum`/`optional` priorities, literal strings and numbers preserved. It
  is data for the later roles; the conversation stays authoritative and every
  consumer is told to recover the requirements from the request when the
  checklist is empty (a failed upstream call renders its slot empty).
- `policies` writes four policies that differ in method, assumptions, and
  evaluation criteria, not wording.
- `synthesis` checks premises, evidence, methods, and the conditions under
  which each candidate's conclusion holds; looks for counterexamples,
  boundary conditions, omissions, and errors common to all five; fixes and
  combines; may take an approach no candidate took; and writes the complete
  proposal followed by an internal `=== DECISION RECORD ===` (reasons and
  references). Agreement among candidates is not evidence.
- `final` re-checks the proposal against the request, the checklist, and all
  candidates, continues directly after the committed opening (or writes the
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

The pinned overlay image (built by `../deepseek-v4.1-flash-8gpu`) aligns
vLLM's effort aliases with the checkpoint's table; this example reuses that
image by ID and never patches it. `compose.yaml` sets **no**
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
   the rendered input approaches 262,144 minus the Qwen role caps (about
   245,000 tokens for the ensemble); DeepSeek-only routes reach 1,048,576.
2. The upstream 400 reason is replaced by "orchestration final unit produced
   no public output" (502); the actual reason is in the gateway log.
3. L3 appends `<image:N>` directly to the user text in the rendered
   conversation and repeats the latest user turn as a plain-text view;
   exact-ending requirements next to an image can be misread by extraction.
   DeepSeek roles also receive the real image and are told the checklist is
   fallible.
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
  `local/vllm-openai:deepseek-v41-sm120` ID `sha256:027bf47b…` (vLLM
  `179dd0fa9`, FlashInfer `60b49158`, built by
  `../deepseek-v4.1-flash-8gpu/vllm-sm120.Dockerfile`); Open WebUI
  `v0.11.0-slim@sha256:3698bd4e…`.
- Served-config hash: `verification.py` hashes `example.json`,
  `compose.yaml`, `kairyu.yaml`, `auto-max.yaml`, `ensemble-max.yaml`,
  `router.json`, `l1-qwen3.8-27b-vllm-chat-template.jinja`, `webui-reasoning-effort-filter.py`,
  and `benchmark.py` into every `run.json`, and attests the running images,
  commands, and mounted files before measuring.
