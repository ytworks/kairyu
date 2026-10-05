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
| CI | `.github/workflows/open-responses.yml`: bun 1.4.2 via `oven-sh/setup-bun` (SHA-pinned); PRs touching `kairyu/**`, dependencies or the gate (blocking); weekly drift run (read-only token) and drift-issue job (non-blocking) |

**Verdict** (`scripts/open_responses_gate/verdict.py`). The suite's exit
status and `--filter` are not used (`--filter` still runs every scenario,
upstream issue #55). Every scenario in the JSON report gets one outcome:

- `PASS`: it passed and is not listed.
- `XFAIL`: it failed, is listed, and reported exactly the entry's errors
  (its `errors` plus its named `[error_sets]`, compared as sets); with
  `body_contains`, its response body also contains that text.
- `FAIL`: it failed and is not listed; it is listed but reported an unlisted
  error, no longer reported a listed one, or lacks the body text; or it was
  skipped.
- `XPASS`: it passed although it is listed.
- `MISSING`: it is listed but absent from the report.

Only `PASS` and `XFAIL` are green, so the list stays truthful and a new
error in a listed scenario (say a `function_call` item losing `call_id`) is
not masked by the errors it is listed for:
- the WP that fixes a scenario deletes its entry;
- a WP that changes how a listed scenario fails updates its errors;
- a pin move adds or deletes entries in the same PR.

`body_contains` exists for HTTP status failures: the suite then reports only
`HTTP 400: [object Object]`, and the reason is in the body.

The file is validated when loaded: required `reason`, `gap_id` and `owner_wp`
(the vocabulary of `divergences.toml`), at least one error per entry, known
and used error sets, no repeated errors, unique ids, and a full commit SHA.
The report must agree with its own summary counts.

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

| Scenarios | Errors | Gap → owner |
|---|---|---|
| basic-response, assistant-phase, system-prompt, multi-turn, tool-calling, streaming-response | error set `envelope` (14 top-level keys, `completed_at: Invalid input` …); tool-calling also `tools.0.strict: Invalid input` | G-output-object-1 → WP-08b |
| image-input | `HTTP 400: [object Object]`, body `'input_image' is not supported` | G-input-items-1 → WP-25 |
| compact-response, compact-missing-model | `HTTP 404: [object Object]`, body `Invalid URL (POST /v1/responses/compact)` | G-compaction-2 → WP-36 |
| websocket-response, -sequential-responses, -continuation, -reconnect-store-false-recovery, -previous-response-not-found, -failed-continuation-evicts-cache | `WebSocket connection failed` | G-transport-1 → WP-47 |
| websocket-compact-new-chain | `HTTP 404 from /responses/compact: {…}` | G-transport-1 → WP-47 (WP-36 changes its errors) |

Notes on the expected failures:
- **Envelope (WP-08b).** The suite's `ResponseResource` requires 31 keys.
  Five sampling numbers must be non-null numbers, `service_tier` must be a
  string, and `text` must be an object with `format`. OpenAI allows null for
  several of these (upstream issue #63). Echoing the resolved values (R-8)
  satisfies both schemas. In `tool-calling`, the echoed function tool also
  lacks `strict`.
- **Streaming.** While the terminal envelope fails its schema, the suite
  reports only those envelope errors and skips its per-event checks
  (`validateResponseData` returns before the `streamingSchema` validator).
  The gate therefore covers no stream event shape until WP-08b fixes the
  envelope; from then on `streaming-response` protects the WP-17/18 emitter.
  A one-off check outside the gate (the suite's `parseSSEStream` against the
  launcher, 2026-10-05) found 10 of 13 events valid; `response.created`,
  `response.in_progress` and `response.completed` fail only on the same
  envelope keys under `response.`.
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
procedure. When the suite cannot run at `main` (a changed lockfile, CLI,
report format or result status), the issue reports that failure instead. The
workflow opens one issue per upstream commit (the title names it) and never
fails on drift.

The `drift` job runs upstream's unpinned code with a read-only token. It hands
the title and body as an artifact to `drift-issue`, the only job with
`issues: write`; that job runs no upstream code and accepts only a title of
the form drift.py writes.

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
- Evaluated against that run's report, wrong listed errors, a removed entry
  and an unknown id give FAIL, FAIL and MISSING respectively. A malformed
  list (a short commit, a bad owner WP, an entry without errors, an unknown
  or unused error set) is rejected at load.
- Drift with a temporary pin at the parent commit `cd31bc2060`: the issue body
  lists the CHANGELOG and `public/openapi` changes, the verdict at `main` is
  green, exit 0. With the real pin, the job reports no drift (`main` is
  `92c12d96d7`).

**Framework boundary.** WP-14 changes nothing under `kairyu/`. It adds:
- the gate scripts;
- the test deployment and list under `tests/contracts/openresponses/`;
- the workflow.

`tests/contracts/divergences.py` exports its gap and owner-WP patterns, so both
lists share one vocabulary. The live gate is a CI job, like the WP-05 Codex
matrix. One pytest case (`tests/unit/test_open_responses_gate.py`) feeds the
verdict a synthetic report: the live suite produces no failure beyond the
listed ones, so it cannot show that an extra or missing error in a listed
scenario turns the gate red.

*2026-10-05, WP-14 review fixes:* a listed scenario now needs exactly its
listed errors (before, one listed substring anywhere let a new error through,
reproduced by appending `output.0.call_id: Required` to `tool-calling`); the
drift run files a "could not run" issue instead of crashing (probe: a failing
`bun install` at a temporary pin, exit 0 with title and body); and the drift
job's token is read-only. The pinned gate stays 1 PASS and 16 XFAIL.
