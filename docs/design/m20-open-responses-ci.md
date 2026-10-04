# M20 companion: Open Responses CI (WP-14)

Parent: `docs/design/m20-responses-compat.md` (D1 conformance infrastructure,
§3 pinned reference contract, §12 CPU gate 3). Gap: G-conformance-3.
Date: 2026-10-05. This file holds the WP-14 records so that the m20 document
stays within its size budget; later changes are dated entries below.

## Gate

| Piece | Value |
|---|---|
| Suite | `openresponses/openresponses@92c12d96d7b61d6d15e2214daa5e9c6000ab6e1c` (spec 2026-04-24), `bin/compliance-test.ts --json`, 17 scenarios |
| Pin and expected failures | `tests/contracts/openresponses/expected-failures.toml` (`[suite]` plus one `[[expected_failure]]` per failing scenario) |
| Server | `tests/contracts/openresponses/kairyu-mock.yaml` (engine `kairyu-mock`, `legacy_chat_models`) on the WP-02b `ScenarioBackend`, served with `kairyu serve`'s uvicorn options (`ws="none"`) by `scripts/open_responses_gate/launcher.py` |
| Runner | `scripts/open_responses_gate/run.py`: shallow fetch at the pin, `bun install --frozen-lockfile --ignore-scripts`, launcher on a free port, suite against `/v1`, verdict |
| CI | `.github/workflows/open-responses.yml`: bun 1.4.2 via `oven-sh/setup-bun` (SHA-pinned); PRs touching `kairyu/**`, dependencies or the gate (blocking); weekly drift job (non-blocking) |

**Verdict** (`scripts/open_responses_gate/verdict.py`). The suite's exit
status and `--filter` are not used (`--filter` still runs every scenario,
upstream issue #55). Every scenario in the JSON report gets one outcome:

- `PASS`: it passed and is not listed.
- `XFAIL`: it failed, is listed, and its errors or error body contain the
  entry's `signature`.
- `FAIL`: it failed and is not listed, failed without its signature, or was
  skipped.
- `XPASS`: it passed although it is listed.
- `MISSING`: it is listed but absent from the report.

Only `PASS` and `XFAIL` are green, so the list stays truthful:
- the WP that fixes a scenario deletes its entry;
- a WP that changes how a listed scenario fails updates its signature;
- a pin move adds or deletes entries in the same PR.

The file is validated when loaded: required `reason`, `gap_id` and `owner_wp`
(the vocabulary of `divergences.toml`), unique ids, and a full commit SHA. The
report must agree with its own summary counts.

**Scripted backend.** The suite runs its scenarios concurrently, so
`launcher.SCRIPT` picks turns by the scenario's user text, never by call
order:
- `tool-calling` gets a `get_weather` call;
- every other HTTP scenario gets fixed text.

No turn carries reasoning. Kairyu follows OpenAI's `response.reasoning_text.*`
event names, which Codex and the SDKs read; the pinned suite's event union
names them `response.reasoning.*` (upstream PR #42, open). The streaming
scenario therefore runs with reasoning off. Reasoning events stay covered by
the OpenAI schema gate (WP-01) and by WP-21.

## Expected failures at the pin (2026-10-05)

| Scenarios | Signature | Gap → owner |
|---|---|---|
| basic-response, assistant-phase, system-prompt, multi-turn, tool-calling, streaming-response | `completed_at: Invalid input` | G-output-object-1 → WP-08b |
| image-input | `'input_image' is not supported` | G-input-items-1 → WP-25 |
| compact-response, compact-missing-model | `Invalid URL (POST /v1/responses/compact)` | G-compaction-2 → WP-36 |
| websocket-response, -sequential-responses, -continuation, -reconnect-store-false-recovery, -previous-response-not-found, -failed-continuation-evicts-cache | `WebSocket connection failed` | G-transport-1 → WP-47 |
| websocket-compact-new-chain | `HTTP 404 from /responses/compact` | G-transport-1 → WP-47 (WP-36 moves the signature) |

Notes on the expected failures:
- **Envelope (WP-08b).** The suite's `ResponseResource` requires 31 keys.
  Five sampling numbers must be non-null numbers, `service_tier` must be a
  string, and `text` must be an object with `format`. OpenAI allows null for
  several of these (upstream issue #63). Echoing the resolved values (R-8)
  satisfies both schemas. In `tool-calling`, the echoed function tool also
  lacks `strict`.
- **Streaming.** All 13 events of `streaming-response` validate. Only the
  terminal envelope fails, so after WP-08b the scenario protects the WP-17/18
  emitter.
- **image-input.** WP-25 accepts message images. The gate then also needs a
  backend that answers multimodal prompts, because `ScenarioBackend` rejects
  them.
- **Outside the suite.** The suite does not test D-h (call finality over "an
  incomplete item must be last") or `previous_response_id` over HTTP. Those
  are covered by the Codex fixtures and the schema gate.

## Weekly drift

`scripts/open_responses_gate/drift.py` resolves upstream `main`. When `main`
is not the pin, it runs that suite against the same server with the pinned
expected failures. It then writes an issue: the compare link, the
`git diff --numstat` of the spec and suite paths, the verdict, and the pin-move
procedure. The workflow opens one issue per upstream commit (the title names
it) and never fails on drift.

**Moving the pin:**
1. Edit `[suite]` in `expected-failures.toml`.
2. Run `python -m scripts.open_responses_gate.run --out <dir>`.
3. Apply its FAIL, XPASS and MISSING rows to the list, in the same PR.

## Local run

    npm install --prefix <dir> bun@1.4.2      # CI's bun, kept out of the global PATH
    python -m scripts.open_responses_gate.run --out <dir>/out \
        --cache-dir <dir>/cache --bun <dir>/node_modules/.bin/bun

`--out` holds `results.json` (the suite report), `verdict.md` and `server.log`.

## Records

*2026-10-05, WP-14 (macOS, bun 1.4.2, base `feat/m20-wp15-ws-none`):*

**Pinned suite.** 1 PASS and 16 XFAIL; the gate is green. The one scenario
that passes, `response-output-phase-schema`, sends no request: it validates a
local fixture. This matches the research prediction (1/17). The HTTP scenarios
reach Kairyu and produce the scripted text and function call. Their failures
are only the envelope gaps, the image rejection and the missing `/compact`
route listed above. Each scenario took ≤ 21 ms.

**Red probes:**
- With an empty list, the gate is RED: 16 FAIL, exit 1.
- With an entry added for the passing scenario, it is RED: XPASS, exit 1.
- Evaluated against that run's report, a wrong signature, a removed entry and
  an unknown id give FAIL, FAIL and MISSING respectively. A malformed list (a
  short commit, a bad owner WP, an empty signature) is rejected at load.
- Drift with a temporary pin at the parent commit `cd31bc2060`: the issue body
  lists the CHANGELOG and `public/openapi` changes, the verdict at `main` is
  green, exit 0. With the real pin, the job reports no drift (`main` is
  `92c12d96d7`).

**Framework boundary.** WP-14 changes nothing under `kairyu/`. It adds:
- the gate scripts;
- the test deployment and list under `tests/contracts/openresponses/`;
- the workflow.

`tests/contracts/divergences.py` exports its gap and owner-WP patterns, so both
lists share one vocabulary. pytest collection is unchanged: the gate is a CI
job, like the WP-05 Codex matrix.
