# Routed answers: DeepSeek-V4.1 (6 GPUs) + two Quyet-1.0-Large replicas (2 GPUs)

One public model, `kairyu-verified-tool`. Quyet reads each conversation and
asks whether the next reply needs to call one of the caller's tools. If it
does, the turn takes the verified tool route (VCO-D22): DeepSeek lists
candidate calls, Quyet judges them, DeepSeek makes the move, and Quyet checks
the reply's form before it is published. Every other request takes the think
route, where DeepSeek answers at the effort the caller asked for.

Design: `docs/design/example-verified-checklist-orchestration.md` (VCO-D21,
VCO-D22); framework mechanisms: m1 D8 (checklist verifiers and their
refinement), m1 D9 (the System One route judge). `kairyu/` is not changed
for this example.

| Layer | What runs here |
|---|---|
| L1 | DeepSeek-V4.1-Flash, one DP6/EP6 replica on GPUs 0-5 (the six-GPU example's L1, no server-wide thinking default). Two Quyet-1.0-Large replicas served as in `quyet-1.0-large-1gpu` (stock vLLM v0.31.0, bf16, batch-invariant, plus this example's System One adapter that keeps the `quyet` package's prompt and calibration): `quyet-route` on GPU 6 answers only the route judge, `quyet-judge` on GPU 7 the judgments and the form check. |
| L2 | `verified-tool.yaml`: the Quyet route judge, the verified tool route and the think route. |
| L3 | Public model `kairyu-verified-tool`; Open WebUI on :3012. |

## L2: how an answer is made

```text
request
  │
  ▼
profile_judge ── quyet-route (System One), 1 request: does the next reply
  │              need a tool call? TOOL or THINK
  ├─ TOOL
  │    candidates    DeepSeek, the caller's effort: 10-16 candidate tool calls
  │                  for the next move from different viewpoints, one call
  │                  each, in one request
  │    judgments     quyet-judge, 6 questions per candidate (60-96), 32 per
  │                  read, the reads in parallel: needed now? repeats no shown
  │                  step? fits the tools? follows the request exactly? checks
  │                  the way it will be judged? safe now?
  │    answer        DeepSeek, the caller's effort: the caller's conversation
  │                  as native messages, then the candidates and judgments in
  │                  one message; makes the move through the tool-calling
  │                  interface, several independent safe calls at once when the
  │                  conversation allows it
  │    format_check  quyet-judge, 7 questions on the reply (p >= 0.5 each):
  │                  announced calls made, no call written as text, declared
  │                  tools with valid arguments, at least one call, batched
  │                  calls independent, finishing call alone after success,
  │                  short text without internal material
  │                  fail → answer writes the reply again (only what the unmet
  │                  items need; the move kept) → checked again, at most twice
  └─ THINK ─────► deepseek_think_answer  DeepSeek, the caller's effort
                                          (default high), one call with the
                                          caller's conversation as sent
```

- Quyet reads at most 6,000 state tokens (its documented input; a longer
  object is cut at its end). Every state is bounded to fit: the route judge
  reads at most 12,000 characters of the conversation, the judgments 8,000
  and the form check 6,000. A long run keeps its first message, its system
  and developer messages, its latest user message and its newest messages,
  so an agent's task, protocol and latest results stay visible; the middle
  of a long run is not read.
- The judge sends the conversation and whether the caller declared tools,
  with one choice question; the most probable label wins. When the
  conversation's instructions require a tool call in every reply or end the
  work with one, every reply up to that final one is TOOL.
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
- Both routes pass the caller's tools and response_format to the DeepSeek role
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

`run.sh` pulls the pinned vLLM image for Quyet (registry digest checked),
builds this example's Quyet System One adapter on it whenever its sources
change (their hashes are image labels), downloads the Quyet checkpoint and
checks it against the publisher's `MANIFEST.sha256` and the pinned tree, and
builds the DeepSeek SM120 overlay when its pinned image is absent.
`DEEPSEEK_MODEL_SEED` may name a local copy of the pinned DeepSeek checkpoint
below `/mnt/nvme`; it is hard-linked and re-hashed against the pin before
serving.

## Using it

The OpenAI API listens on `127.0.0.1:8013`. An agent sends its tools with
each turn:

```sh
curl -s http://127.0.0.1:8013/v1/chat/completions -H 'Content-Type: application/json' -d '{
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
| `l1` | every DeepSeek DP rank (thinking and chat JSON), each Quyet replica's vLLM and System One, one structured tool call from the public model |
| `routing` | `datasets/tool-routing-set.json`: TOOL miss rate < 10 % on the calibration and held-out halves; THINK precision ≥ 90 % |
| `think-route` | everyday requests stream from the think route at the default effort |
| `effort` | the think route gets the caller's effort; the verified tool route's candidates and answer (fixes included) get it too (default high); the route is judged on `quyet-route` |
| `verified-tool-route` | agent turns from `datasets/tool-turns.json` take the TOOL route: 10-16 candidates, 6 judgments each on `quyet-judge`, the answer and its 7-question form check on `quyet-judge`; each reply carries a structured call to a declared tool; fixes, publish reasons and calls per reply recorded |
| `fallback` | `quyet-route` down: think route; `quyet-judge` down: still routed, answer without judgments, published unverified; both back: judged and form-checked |
| `serving` | agent turns at c1/c4/c8/c16 (8/16/16/32 requests): every reply a structured call; per-route latency and tokens |
| `serving-routed` | the routing set at c1/c4/c8/c16, route mix and per-route latency and tokens |
| `browser` | `browser-smoke.sh`: Open WebUI answers |

Every gate also records each Quyet adapter's reads, states cut at 6,000
tokens and read times from the adapters' logs. The two datasets are authored
with generic tools (shell, file read and write, HTTP fetch, web search,
calendar, email). The routing set's THINK side holds 80 chats without tools
and 16 tool-declared turns whose next reply is text.

## Limits

- Quyet reads at most 6,000 tokens of state: on a long agent run the route
  judge, the judgments and the form check see the task's messages and the
  newest ones, not the middle of the run.
- When the judgments cannot be read (quyet-judge down, or a candidate so long
  that its question leaves no room for the state, about 20,000 characters),
  the answer is written without judgments and published without the form
  check (a run shares one unjudgeable flag, m1 D8).
- A fix is the same DeepSeek role at the same effort; it receives the
  previous reply as text. A DeepSeek error during a fix fails the request.
- The verified tool route is not streamed: the reply is published after its
  form check.
- The candidates share one call capped at 131,072 tokens (the DSL's internal
  maximum).
- Latency: measured in `MEASUREMENTS.md`.
