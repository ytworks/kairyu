# Routed answers: DeepSeek-V4.1 (6 GPUs) + two Winnow-12B replicas (2 GPUs)

One public model, `kairyu-verified-tool`. Winnow reads each conversation and
asks whether the next reply needs to call one of the caller's tools. If it
does, the turn takes the verified tool route: three waves of DeepSeek and
Winnow (VCO-D19, VCO-D20). Every other request takes the think route, where
DeepSeek answers at the effort the caller asked for.

Design: `docs/design/example-verified-checklist-orchestration.md` (VCO-D18,
VCO-D19, VCO-D20); framework mechanisms: m1 D8 (checklist verifiers, including
the 2026-10-06 wait for a parallel branch), m1 D9 (the System One route judge)
and LCP-D1..D6 (llama.cpp upstream).

| Layer | What runs here |
|---|---|
| L1 | DeepSeek-V4.1-Flash, one DP6/EP6 replica on GPUs 0-5 (the six-GPU example's L1, no server-wide thinking default). Two Winnow-12B Q8_0 replicas (each as in `winnow-12b-q8-1gpu`, chat and System One from one loaded model): `winnow-route` on GPU 6 answers only the route judge, `winnow-judge` on GPU 7 only the judgments. |
| L2 | `verified-tool.yaml`: the Winnow route judge, the verified tool route and the think route. |
| L3 | Public model `kairyu-verified-tool`; Open WebUI on :3012. |

## L2: how an answer is made

```text
request
  │
  ▼
profile_judge ── winnow-route (System One), 1 request: does the next reply
  │              need a tool call? TOOL or THINK
  ├─ TOOL
  │    wave 1 ┬ drafts        DeepSeek, the caller's effort: five candidate next
  │           │               moves D1..D5, each its tool calls and short text,
  │           │               as different as possible (one call)
  │           └ requirements  DeepSeek, thinking at max: what the next move must
  │                           meet, from the request's own words (≤ 16)
  │    wave 2   judgments     winnow-judge, 1 request with the request, the
  │                           conversation and the tools: each draft adoptable
  │                           as the next move? each draft x requirement met?
  │                           (5 + 5 x N probabilities)
  │    wave 3   answer        DeepSeek, the caller's effort: reads drafts,
  │                           requirements and judgments critically, makes the
  │                           best move through the tool-calling interface
  └─ THINK ─────► deepseek_think_answer  DeepSeek, the caller's effort
                                          (default high), one call
```

- The judge sends the whole conversation and whether the caller declared
  tools, with one choice question; the most probable label wins. Each
  message is cut at 4,000 characters and the whole at 120,000; a long run
  keeps its first message, its system and developer messages and its latest
  user message, so an agent's task and protocol stay visible.
- When the conversation's instructions require a tool call in every reply
  or end the work with one, every reply up to that final one is TOOL.
- When winnow-route does not answer within 60 s (down, overloaded,
  unreadable), the request takes the think route.
- The two Winnow replicas never share a queue: a burst of judgments cannot
  delay the next request's route decision.
- The reply is the next assistant message: the tool call or calls for the one
  move needed now, with at most a short text. It is not the task's final
  result in one jump, and not a repeat of a step the tool results already
  show. The finishing move goes alone, once tool results show every step it
  depends on succeeded.
- The requirements are written from the request's own words, for the current
  position: exact names, paths and public entry points; checks taken from the
  request, not bent to fit; verification through the public path, build mode
  and type checks that existing checks use; must-not behaviour; and order and
  finishing. The prompts name no agent, language or test runner.
- Both routes pass the caller's tools and response_format to the DeepSeek
  role that publishes. The drafts and the requirements read the tools, and
  each draft puts its calls in `tool_calls`, never in its text. A call
  DeepSeek writes at the end of its reply in its own format but without its
  marker tokens is still returned as a tool call (m9 D2 amendment).
- Stage reports are not returned in `reasoning_content`: an agent replays
  them in its next request.
- If winnow-judge cannot read the judgments (down, or the drafts exceed its
  65,536-token decision context), the answer is written without them.

## Run

```sh
./run.sh            # preflight, images, checkpoints, compose up, readiness probes
./verify.sh list    # GPU gates; evidence lands in model-volumes/<env>/results/
./browser-smoke.sh  # Open WebUI in a real browser
./run.sh down
```

`run.sh` builds `winnow-server` from its pinned revision with the f072b10
llama.cpp backport (`winnow-patches/`) once for both replicas, and builds the
DeepSeek SM120 overlay when its pinned image is absent. `DEEPSEEK_MODEL_SEED`
may name a local copy of the pinned checkpoint below `/mnt/nvme`; it is
hard-linked and re-hashed against the pin before serving.

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
| `l1` | every DeepSeek DP rank (thinking and chat JSON), chat and System One on each Winnow replica (L1), one structured tool call from the public model |
| `routing` | `datasets/tool-routing-set.json`: TOOL miss rate < 10 % on the calibration and held-out halves; THINK precision ≥ 90 % |
| `think-route` | everyday requests stream from the think route at the default effort |
| `effort` | the think route gets the caller's effort; the verified tool route's drafts and answer get it too (default high); the requirements always think at max; the route is judged on `winnow-route` |
| `verified-tool-route` | agent turns from `datasets/tool-turns.json` take the TOOL route and run the three waves (drafts beside requirements, one `winnow-judge` read of 5 + 5 x N items, the answer); each reply carries a structured call to a declared tool |
| `fallback` | `winnow-route` down: think route; `winnow-judge` down: still routed, answer without judgments; both back: routed and judged |
| `serving` | agent turns at c1/c4/c8/c16 (8/16/16/32 requests), per-route latency and tokens |
| `serving-routed` | the routing set at c1/c4/c8/c16, route mix and per-route latency and tokens |
| `browser` | `browser-smoke.sh`: Open WebUI answers |

The two datasets are authored with generic tools (shell, file read and write,
HTTP fetch, web search, calendar, email). The routing set's THINK side holds 80
chats without tools and 16 tool-declared turns whose next reply is text.

## Limits

- No answer carries a guarantee (`kairyu_verification` is absent): Winnow's
  judgments inform the answer, they do not gate it.
- The drafts share one call capped at 131,072 tokens (the DSL's internal
  maximum).
- Latency: measured in `MEASUREMENTS.md`.
