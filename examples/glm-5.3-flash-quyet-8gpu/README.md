# Routed answers: GLM-5.3-Flash (6 GPUs) + two Quyet-1.0-Large replicas (2 GPUs)

One public model, `kairyu-verified-tool`. Quyet reads each conversation and
routes it (VCO-D25). While the next reply must call one of the caller's tools
and the assistant has made no change yet (no call to create, edit, move or
delete data, send, install or commit), the turn takes the verified tool route
(VCO-D22, VCO-D23): GLM lists
candidate calls, Quyet judges them, GLM makes the move, and Quyet checks
the reply's form before it is published. Every other request takes the think
route, where GLM answers at the effort the caller asked for (default
max, GLM's official default).

Design: `docs/design/example-verified-checklist-orchestration.md` (VCO-D21,
VCO-D22, VCO-D23, VCO-D25); framework mechanisms: m1 D8 (checklist verifiers and their
refinement), m1 D9 (the System One route judge). `kairyu/` is not changed
for this example.

| Layer | What runs here |
|---|---|
| L1 | GLM-5.3-Flash (official FP8), one TP2 x DP3 / EP6 replica with MTP 3 on GPUs 0-5, exactly as `glm-5.3-flash-6gpu` selected it (stock vLLM v0.31.0). Two Quyet-1.0-Large replicas served as in `quyet-1.0-large-1gpu` (stock vLLM v0.31.0, bf16, batch-invariant, plus this example's System One adapter that keeps the `quyet` package's prompt and calibration): `quyet-route` on GPU 6 answers only the route judge, `quyet-judge` on GPU 7 the judgments and the form check. |
| L2 | `verified-tool.yaml`: the Quyet route judge, the verified tool route and the think route. |
| L3 | Public model `kairyu-verified-tool`; Open WebUI on :3017. |

## L2: how an answer is made

```text
request
  │
  ▼
profile_judge ── quyet-route (System One), 1 request: a tool call needed, no
  │              change made yet, at most 20 messages out of view? MULTI or SINGLE
  ├─ MULTI
  │    candidates    GLM, the caller's effort: 10-16 candidate tool calls
  │                  for the next move from different viewpoints, one call
  │                  each, in one request
  │    judgments     quyet-judge, 6 questions per candidate (60-96), 32 per
  │                  read, the reads in parallel: needed now? repeats no shown
  │                  step? fits the tools? follows the request exactly? checks
  │                  the way it will be judged? safe now?
  │    answer        GLM, the caller's effort: the caller's conversation
  │                  as native messages, then the candidates and judgments in
  │                  one message; makes the move through the tool-calling
  │                  interface, several independent safe calls at once when the
  │                  conversation allows it
  │    format_check  quyet-judge read strict, 7 questions on the reply (p >= 0.5 each):
  │                  announced calls made, no call written as text, declared
  │                  tools with valid arguments, at least one call, batched
  │                  calls independent, finishing call alone after success,
  │                  short text without internal material; reported in
  │                  kairyu_verification, the reply published as written
  └─ SINGLE ────► glm_think_answer  GLM, the caller's effort
                                     (default max), one call with the
                                     caller's conversation as sent
```

- Quyet reads at most 6,000 state tokens (its documented input; a longer
  object is cut at its end). The route judge reads at most 12,000 characters
  of the conversation and the judgments 8,000. A long run keeps its first
  message, its system and developer messages, its latest user message and its
  newest messages, so an agent's task, protocol and latest results stay
  visible; the middle of a long run is not read.
- The form check's reply cannot be bounded (a call may write a whole file), so
  it reads in Quyet's strict mode (the adapter name `quyet-strict`): a state
  Quyet would cut (the reply, the tools and 6,000 characters of the
  conversation) is refused, and the reply is published unverified rather than
  passed on what fits.
- The judge sends the conversation, the number of messages left out of it
  and whether the caller declared tools, with one choice question; the more
  probable label wins. MULTI only while the assistant has made no change and
  at most 20 messages are left out (a change further back may be out of
  view). On the DeepSWE records (DeepSeek as the writer) that phase took about
  the time of one answer per reply on the verified tool route with far fewer
  calls and input tokens, while every later phase took 1.4-10 times the time
  (VCO-D25). The first change is written on the verified tool route: whether
  the next reply makes a change cannot be read before it is written.
- When quyet-route does not answer within 60 s (down, overloaded), the
  request takes the think route.
- The two Quyet replicas never share a queue: a burst of judgments cannot
  delay the next request's route decision.
- The reply is the next assistant message: the tool call or calls for the move
  needed now, with at most a short text. It is not the task's final result in
  one jump, and not a repeat of a step the tool results already show. The
  finishing move goes alone, once tool results show every step it depends on
  succeeded.
- Each judgment question carries its candidate's tool, arguments and purpose;
  the state holds the conversation and the caller's tools. The answer reads
  every judgment with its probability as evidence, not as a verdict.
- The form check reads the reply as text with its calls written as
  `<tool_call>{...}</tool_call>`, then the tools and the conversation. The
  response carries its outcome in `kairyu_verification` (`guaranteed` true
  when every form item passed, `reason` otherwise, `attempts`); it reports
  the reply's form, not the correctness of the move.
- Both routes pass the caller's tools and response_format to the GLM role
  that publishes, which receives the caller's conversation as native chat
  messages (m1 D8 amendment). The candidates read the conversation as JSON.
- Stage reports are not returned in `reasoning_content`: an agent replays
  them in its next request.

## Run

```sh
./run.sh            # preflight, images, checkpoints, compose up, readiness probes
./verify.sh list    # GPU gates; evidence lands in model-volumes/<env>/results/
./browser-smoke.sh  # Open WebUI in a real browser
./run.sh down
```

`run.sh` pulls the pinned stock vLLM v0.31.0 image that GLM and Quyet share
(registry digest checked), builds this example's Quyet System One adapter on
it whenever its sources change (their hashes are image labels), downloads the
GLM checkpoint (305.8 GiB) and the Quyet checkpoint and checks them against
the pinned trees (Quyet also against the publisher's `MANIFEST.sha256`), and
refuses to start when a GLM TP pair spans two NUMA nodes. `GLM_MODEL_SEED`
may name a local copy of the pinned GLM checkpoint below `/mnt/nvme`; it is
hard-linked and re-hashed against the pin before serving.

## Using it

The OpenAI API listens on `127.0.0.1:8016`. An agent sends its tools with
each turn:

```sh
curl -s http://127.0.0.1:8016/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "kairyu-verified-tool",
  "messages": [{"role": "user", "content": "What is the weather in Paris right now?"}],
  "tools": [{"type": "function", "function": {"name": "get_weather",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                   "required": ["city"]}}}]
}' | jq '.choices[0].message.tool_calls'
```

## GPU gates (`verify.sh`)

| Gate | Checks |
|---|---|
| `l1` | every GLM DP rank (grammar JSON at the default and low effort), each Quyet replica's vLLM and System One, the strict name refusing a state Quyet would cut, one structured tool call from the public model |
| `routing` | `datasets/tool-routing-set.json` under the served rule (the more probable label): ≥ 90 % of the MULTI conversations routed MULTI on the calibration and held-out halves; at most one conversation holding a change routed MULTI |
| `think-route` | everyday requests stream from the think route at the default effort |
| `effort` | the think route gets the caller's effort; the verified tool route's candidates and answer get it too (default max); the route is judged on `quyet-route` |
| `verified-tool-route` | the MULTI agent turns from `datasets/tool-turns.json` take the verified tool route: 10-16 candidates, 6 judgments each on `quyet-judge`, the answer and its 7-question form check on `quyet-judge`; each reply carries a structured call to a declared tool; form results, publish reasons and calls per reply recorded |
| `fallback` | `quyet-route` down: think route; `quyet-judge` down: still routed, answer without judgments, published unverified; both back: judged and form-checked |
| `serving` | agent turns at c1/c4/c8/c16 (8/16/16/32 requests): every reply a structured call; per-route latency and tokens |
| `serving-routed` | the routing set at c1/c4/c8/c16, route mix and per-route latency and tokens |
| `browser` | `browser-smoke.sh`: Open WebUI answers |

Every gate also records each Quyet adapter's reads, states cut at 6,000
tokens, strict reads refused and read times from the adapters' logs. The two datasets are authored
with generic tools (shell, file read and write, HTTP fetch, web search,
calendar, email), labelled by the route's definition: the routing set's MULTI
side holds 27 tool turns with no change yet; its SINGLE side holds 80 chats
without tools, 16 tool-declared turns whose next reply is text and 13 tool
turns that already hold a change (14 conversations are marked `changed`).
`tool-turns.json` marks each agent turn's route (11 MULTI, 13 SINGLE).

## Limits

- Quyet reads at most 6,000 tokens of state: on a long agent run the route
  judge, the judgments and the form check see the task's messages and the
  newest ones, not the middle of the run.
- When the judgments cannot be read (quyet-judge down, or a candidate so long
  that its question leaves no room for the state, about 20,000 characters),
  the answer is written without judgments and published without the form
  check (a run shares one unjudgeable flag, m1 D8).
- A reply whose form-check state exceeds Quyet's 6,000 tokens (a long write,
  many or large tool schemas) is published unverified.
- A tool-declared turn whose next reply is text can still read as MULTI and
  get an answer with a call (routing set on the same Quyet route judge in
  `deepseek-v4.1-quyet-8gpu`, 2026-10-15: 0.95 of the SINGLE conversations
  without a change routed SINGLE).
- Only the turns before the first change, and the first change itself, run
  on the verified tool route. A change is seen only while it stays in the
  judge's view or more than 20 messages are left out (DeepSWE records: 0.3 %
  of the turns after a change routed MULTI).
- The form check does not send a reply back for a fix (VCO-D23 amendment
  2026-10-13): Quyet read some correct one-call replies as failing (F1, F5),
  and in the GPU runs 2 of the 10 replies that reached the fix limit had lost
  their call. A reply that fails the check is published as written with the
  failing items in `kairyu_verification`.
- vLLM v0.31.0 occasionally ends a grammar-constrained request with MTP as an
  internal error ("grammar rejected tokens", once in about 100 candidates
  calls in the GPU gates); that request fails with 502. The GLM pool keeps its
  only replica after one such failure (`unhealthy_after: 3`), so the requests
  waiting for it are not failed with it.
- On the think route GLM may report the task done instead of making the call
  still needed (`dedup-case` in `datasets/tool-turns.json`, a SINGLE turn; under
  VCO-D23's routing it read P(TOOL) 0.025 and failed 1 of 3 serving runs; DeepSeek made the call each time in
  `deepseek-v4.1-quyet-8gpu`).
- The verified tool route is not streamed: the reply is published after its
  form check.
- The candidates share one call capped at 131,072 tokens (the DSL's internal
  maximum).
- Latency: measured in `MEASUREMENTS.md`.
