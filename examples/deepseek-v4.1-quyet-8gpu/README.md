# DeepSeek-V4.1 answers decided by Quyet: DeepSeek-V4.1 (6 GPUs) + two Quyet-1.0-Large replicas (2 GPUs)

One public model, `kairyu-verified-tool`. DeepSeek answers every request as
it would alone, at the effort the caller asked for. Quyet then reads the
answer once and asks five questions about whether it would cost the work
more replies than needed. If every question passes, the answer is published
as written; if one fails, DeepSeek writes the reply once more with Quyet's
findings, and that reply is published.

The goal is fewer API calls until a multi-turn task is solved (so fewer input
tokens and less time) without a lower score. The gates prove the behaviour;
the effect is measured on DeepSWE.

Design: `docs/design/example-verified-checklist-orchestration.md` (VCO-D27);
framework mechanism: m1 D8 (checklist verifiers and their refinement).
`kairyu/` is not changed for this example.

| Layer | What runs here |
|---|---|
| L1 | DeepSeek-V4.1-Flash, one DP6/EP6 replica on GPUs 0-5 (the six-GPU example's L1, no server-wide thinking default). Two Quyet-1.0-Large replicas served as in `quyet-1.0-large-1gpu` (stock vLLM v0.31.0, bf16, batch-invariant, plus this example's System One adapter that keeps the `quyet` package's prompt and calibration): `quyet-judge` on GPU 7 answers the decision; `quyet-route` on GPU 6 stays deployed with the L1 and is not read (it served the route judge before VCO-D27). |
| L2 | `verified-tool.yaml`: `answer`, then `decision`, then at most one fix. |
| L3 | Public model `kairyu-verified-tool`; Open WebUI on :3012. |

## L2: how an answer is made

```text
request
  │
  ▼
answer    DeepSeek, the caller's effort (default high): the caller's
  │       conversation, tools and format as sent, nothing added
  ▼
decision  quyet-judge read strict, 5 questions in one read (p >= 0.5 each):
  │         E1 no call needed now left for a later reply
  │         E2 the move's call is made (not text only, not announced
  │            without the call, not written in the text)
  │         E3 every call names a declared tool with valid arguments
  │         E4 no step repeated whose result is shown and unchanged
  │         E5 calls in one reply need no result of each other
  ├─ all pass ──────────────► publish the answer as written
  └─ any fails → answer again (the conversation, the draft and the failing
                 questions with their probabilities; only what they need
                 changes) → decided again for the report → publish the fix
```

- Every request takes this flow; there is no route judge. A conversation
  without tools passes E1-E5 and is published as written.
- The decision reads the answer as text with its calls written as
  `<tool_call>{...}</tool_call>`, then the tools and the conversation (2,000
  characters per message, 6,000 in all: the request's messages and the newest
  ones). The answer cannot be bounded (a call may write a whole file), so the
  read is strict (the adapter name `quyet-strict`): a state Quyet would cut is
  refused and the answer is published unverified rather than passed on what
  fits. The same happens when quyet-judge is down.
- Safety and an early finishing call are not judged: the answer is
  DeepSeek's own. The fix prompt keeps every added call safe and the
  finishing call alone, after tool results show success.
- The response carries the decision in `kairyu_verification` (`guaranteed`
  true when the published reply passed its last read, the failing questions,
  `attempts` 1 or 2). It reports the decision, not the correctness of the
  move.
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

The gates guarantee behaviour, not the accuracy of Quyet's decision. Every
traced request is checked against the decision rule: the answer runs, then
the five-question decision on `quyet-judge`; a passing first decision
publishes the answer with no fix (one attempt); a failing one runs exactly one
fix, decided again (two attempts); an unread decision publishes the answer
unverified.

| Gate | Checks |
|---|---|
| `l1` | every DeepSeek DP rank (thinking and chat JSON), each Quyet replica's vLLM and System One, the strict name refusing a state Quyet would cut, one structured tool call from the public model |
| `decision` | 12 agent turns from `datasets/tool-turns.json`, unary and streamed, efforts none/low/high/max: the decision rule, the caller's effort, a structured call to a declared tool, under 900 s; fixes and failing questions recorded |
| `text` | text conversations from `datasets/text-set.json` (4 without tools, 2 with tools) stream an answer under the decision rule at the default effort |
| `effort` | the caller's effort (default high) reaches the answer and its fix, for a text conversation and an agent turn |
| `fallback` | `quyet-judge` down: the answer is published unverified with no fix, still a structured call; back: decided again |
| `serving` | agent turns at c1/c4/c8/c16 (8/16/16/32 requests): every reply a structured call under 900 s and the decision rule; fix share and stage times recorded |
| `serving-mixed` | agent turns and text conversations mixed at c1/c4/c8/c16: every request answered under the decision rule; per-kind latency, tokens and fixes |
| `browser` | `browser-smoke.sh`: Open WebUI answers |

Every gate also records each Quyet adapter's reads, states cut at 6,000
tokens, strict reads refused and read times from the adapters' logs. The two
datasets are authored with generic tools (shell, file read and write, HTTP
fetch, web search, calendar, email). `text-set.json` holds 80 chats without
tools and 16 tool-declared turns whose next reply is text.

## Limits

- Quyet reads at most 6,000 tokens of state: on a long agent run it sees the
  task's messages and the newest ones, not the middle of the run.
- An answer whose decision state exceeds Quyet's 6,000 tokens (a long write,
  many or large tool schemas) is published unverified.
- Before implementation, Quyet's E1 did not separate replies that left room
  for more calls from replies that did not on 120 labelled DeepSWE replies
  (VCO-D27); whether the fixes cut API calls is measured on DeepSWE.
- A fix is the same DeepSeek role at the same effort; it receives the
  previous reply as text. A DeepSeek error during a fix fails the request.
- The answer is not streamed: it is published after its decision.
- Latency: measured in `MEASUREMENTS.md`.
