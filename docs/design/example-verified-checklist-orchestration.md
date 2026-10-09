# Checklist-Verified Answers (DeepSeek-V4.1 six-GPU + OpenJev x 2)

Status: **Accepted 2026-10-01; redesigned 2026-10-02 (VCO-D15); agent turns
verified as steps 2026-10-03 (VCO-D16), replaced by a verified-tool route 2026-10-04 (VCO-D17, PR #619);
rebuilt as Winnow-routed answers without checklists 2026-10-06 (VCO-D18, PR #640), all nine GPU gates pass 2026-10-06;
DeepSeek max requirements and two Winnow replicas 2026-10-07 (VCO-D19 amendment), all nine GPU gates pass 2026-10-07;
next-step replies with structured tool calls 2026-10-07 (VCO-D19 amendment), all nine GPU gates pass 2026-10-07;
one routed verified tool route with prompts tuned on DeepSWE 2026-10-09 (VCO-D20, amended the same day), GPU gates pending**
(evidence: `examples/deepseek-v4.1-winnow-8gpu/MEASUREMENTS.md`; VCO-D1..D17 evidence:
`examples/deepseek-v4.1-openjev-verified-8gpu/MEASUREMENTS.md` at `df109a6b`).
Applies to: `examples/deepseek-v4.1-winnow-8gpu/` (renamed from
`examples/deepseek-v4.1-openjev-verified-8gpu/` by VCO-D18 and from
`examples/deepseek-v4.1-qwen3.8-winnow-8gpu/` by the VCO-D19 amendment of
2026-10-07). Framework
mechanisms: m1 D8 (checklist verifiers, unused since VCO-D18), m1 D9 and the
m11 D8 replica amendment.

## Goal

Return an answer with a guarantee flag that means "the request was met":
every requirement of a checklist that is necessary, sufficient and mutually
exclusive (MECE) with respect to the request passed. Otherwise return the
best available answer with the flag off and the reason.

Roles (owner, 2026-10-01):

- DeepSeek-V4.1-Flash, one DP6/EP6 replica on GPUs 0-5 (the six-GPU
  example's L1): requirement extraction, generator, state builder, repair.
- OpenJev (DiffusionGemma 26B-A4B), one replica each on GPU 6 and GPU 7,
  read through System One: requirement confirmation and per-requirement
  judgments.
- Kairyu L2: Validator, Conductor, transitions, fallback.

## Decisions

### VCO-D1 — DAG

*Amended by VCO-D15 (2026-10-03): the generator writes after adoption, to meet the adopted points; by VCO-D16: the final `answer` role writes that draft itself (no seed).*

`verified.yaml` (DSL only; no Python orchestration in the example):

1. Wave 1: `extract` (DeepSeek, thinking high, JSON grammar) and
   `generator` (DeepSeek, caller effort, sees only the conversation; a caller
   `response_format` constrains it) run in parallel. `extract` is verified by
   `requirements_check` inline (VCO-D2).
2. Wave 2: `answer` is seeded with the generator's draft (no model call) and
   verified by `checklist`: Validator checks, then `state_builder` (DeepSeek,
   non-thinking, JSON claims with verbatim evidence) re-run on the attempt,
   then OpenJev reads, then the Conductor rule. A FAIL is repaired by
   DeepSeek through `answer.refine_prompt` with the failing items, at most
   twice, and the Validator runs again on the repaired version.
3. A rules router with `multi_step_markers: 0` sends every request to the
   DAG: short requests are not exempt from the guarantee.

### VCO-D2 — Requirement set
*Superseded by VCO-D15 (2026-10-02): no rule-based check judges an answer.*


The extractor splits the request (system/developer instructions and the
latest user message) into instruction units U1..Un and writes conditions
R1..Rm, each with a proposition, a kind (`deterministic` with one primitive
from the shared check library, or `semantic`), and its source units.
Identifiers and labels are not instructions; an action that cannot be
performed here becomes "states that it cannot be performed".

`requirements_check` confirms it:

- **Sufficiency**: deterministic coverage (every unit id is cited) and one
  `noul` read per unit ("an answer meeting the conditions citing Ui meets
  Ui"), threshold 0.5.
- **Necessity**: one read per condition ("its source units ask for it").
  Threshold 0 for the verdict; curation drops conditions with p < 0.5.
- **Exclusivity**: one read per pair of conditions sharing a source unit
  ("they require the same thing", expect no), threshold 0.5.

A coverage gap or a duplicate re-extracts once with the problems listed.
Afterwards curation drops low-necessity conditions, merges remaining likely
duplicates (first kept, sources united, propositions joined) and pads every
uncovered unit with "the answer responds to this part of the request: <unit>".

Request-independent requirements have their own ids and never overlap the
extracted ones: G1 groundedness (material-based claims quote the material
verbatim — `G1-excerpts`, deterministic — and, advisory only since
VCO-D11, every claim is supported — OpenJev per claim kind, minimum), G2
execution claims (an "action" claim's
evidence must appear in the conversation, i.e. its tool call and result),
G3 quotations in the answer appear in the conversation.

### VCO-D3 — Validator and Conductor
*Superseded by VCO-D15 (2026-10-02): no rule-based check judges an answer.*


The Validator is the deterministic half of `checklist`: extracted
`deterministic` conditions (an unusable primitive falls back to an OpenJev
read of the proposition), G3 before the state builder, G1-excerpts and G2
after it. Any violation goes to repair without an OpenJev read. The Conductor
passes an attempt only when every item has p >= tau_hi (a check is 1 or 0);
the advisory per-claim G1 items (VCO-D11) are reported but do not gate.
Because the checklist was confirmed sufficient, every item passing means the
request was met.

### VCO-D4 — tau_hi (amended: alpha = 0.10)

tau_hi is calibrated on InFoBench's expert annotation (50 instructions x 5
models, 1,129 labelled requirement judgments). Each decomposed question is
rewritten by DeepSeek as a condition statement and judged through the
production checklist code (same state and question conversion). tau_hi is
the smallest threshold whose accepted judgments have a one-sided 95 %
Clopper-Pearson upper bound on the violation rate <= alpha on the
calibration half (split by instruction, seed 20261001). The held-out half is
reported unchanged (`calibrate.py`).

Result: tau_hi = 0.9966. Calibration: 392 accepted, 29 violations, upper
bound 0.0995. Held-out: 398 accepted, 25 violations (6.3 %), upper bound
0.0866. 54 of 125 held-out answers pass every requirement; 10 of them carry
at least one labelled violation.

Why alpha = 0.10, not 0.05 (owner decision, 2026-10-01): the labels are
noisier than 0.05. When one expert annotator says a requirement is
satisfied, the official label says violated 9.1 % / 10.0 % of the time (the
two experts disagree 5.9-6.6 % of the time). At alpha = 0.05 only p = 1.0
exactly qualified (about 10 % of requirements). Request-form variants did
not lower the violation rate: smaller states, one question per read,
stricter criteria, `steps`/`samples`, atomized requirements, and `think`
all scored equal or worse AUROC (MEASUREMENTS.md).

Limitation: InFoBench has no source material, so G1 was not covered by this
calibration; VCO-D11 measured it separately and made it advisory.

### VCO-D5 — Fallback

*Amended by VCO-D16: on the repair limit the last non-empty attempt is published; an answer that met every point but was not accepted is published unverified (`not_accepted`).*

- Repair limit: the newest attempt whose deterministic checks passed (else
  the draft), `guaranteed: false`, `reason: refinement_limit`.
- Judge unavailable (both OpenJev replicas down or overloaded, in either
  verifier): the generator's draft as-is, `reason: judge_unavailable`.
- Checklist not judgeable (no parseable list, state above 160,000 characters
  for OpenJev's 65,536-token window): the draft, `reason:
  checklist_unavailable`.

### VCO-D6 — Output

The chat response carries `kairyu_verification` (m1 D8) next to the answer;
the answer text is never altered. The example's answer page (nginx on
:3013, Kairyu's API on the same origin) shows the badge, the reason, and the
requirement table with each p.

### VCO-D7 — Jev routing and the think route (2026-10-01)

Two public models: `kairyu-verified` lets Jev (m1 D9) choose between the
verified DAG (VERIFIED) and `deepseek_think`, one thinking DeepSeek answer
(THINK); `kairyu-verified-always` keeps every request on the verified DAG and
backs the answer page and the guarantee gates. VERIFIED criteria: difficult
or specialised questions, many requirements, earlier answers corrected,
a frustrated or pressured user, high-stakes domains, explicit demands for
correctness or sources, output that will be published, sent, signed or
executed, and long multi-part deliverables. Owner decision: accuracy first
(`prefer: VERIFIED` at tau_route; missed accuracy-critical requests < 10 %).
An unjudgeable route falls back to `deepseek_think`, since the verified DAG
could not be judged either. The publisher's private reasoning is withheld
by Kairyu's multi-stage contract on both routes.

### VCO-D8 — Implicit requirements (2026-10-01)

The extractor adds conditions the situation clearly presupposes (origin
`implicit`, attached to the units they serve) next to the stated ones. Each
implicit condition gets its own necessity read ("does the user expect this
even though they did not say it?") and curation drops it below 0.5, so it
enters the guarantee only when Jev judges it expected. Coverage and
exclusivity checks apply unchanged; sufficiency stays on the stated units.

Amendment (2026-10-02): the first full `implicit` gate failed (recall 0.525,
controls kept 0.8 implicit conditions each): the extractor, prompted with
"answering in the user's language" as an example, added that default to
every request (Jev rates it necessary at p 1.0) and missed situational
ones. The extractor now walks the situation (given material to keep, the
audience, the medium's limits, real-world circumstances, produced
artifacts) and leaves out what any ordinary reply meets by default. On a
separate dev set written before the change (12 + 4 controls): recall 0.875
-> 0.917, controls 0.75 -> 0.00. The next gate run reached controls 0.0
but recall 0.600 while extraction plus Jev alone reached 0.825 on the same
set: re-extraction (after a coverage gap or duplicate) rewrote the list
without the implicit guidance and dropped the implicit conditions. The
refine prompt now keeps every implicit condition the problems do not name
and carries the same guidance. The gate's coverage judge now thinks: in
chat mode it marked "every price is kept" uncovered next to three
conditions keeping each price (thinking judge 0.900 vs 0.825 on the same
lists).

Amendment 2 (2026-10-02, owner decision): extraction is two-stage. With
both kinds in one prompt the extractor wrote fewer stated conditions (4.5
vs 5.1 per InFoBench request) and InFoBench gold recall fell to 0.867. Now
`extract` lists stated conditions only (origin `explicit`), and a separate
`implicit` role, in parallel with it (no dependency, so no added latency;
superseded 2026-10-03 by VCO-D15 item 1: `implicit` now reads `extract`),
lists at most four situational conditions with the request words they
belong to; `implicit_check` asks Jev per condition and curation drops it
below 0.5 without blocking the guarantee. The final checklist reads both.
Extraction plus Jev, measured without the full DAG: implicit-set recall
0.925-0.950, dev set 1.000, controls 0.0-0.4 (the planets question listed
the correct answer's content; that is now excluded).

### VCO-D9 — One effort for every DeepSeek step (2026-10-01)

Owner requirement: whatever the route, every DeepSeek role (extract,
generator, repair, state builder, think answer) runs at the caller's
`reasoning_effort` (API field or the Open WebUI dropdown), default high
(75). Open WebUI offers both models and shows the guarantee in the folded
"Verification" section.

### VCO-D10 — Review amendment (PR #616, 2026-10-01)

A requirement checklist still failing sufficiency after re-extraction and
curation blocks the guarantee (`requirements_unconfirmed`); deterministic
conditions are never merged; execution claims need tool-result evidence;
`n > 1` is refused on the verified models.

### VCO-D11 — Per-claim G1 is advisory (2026-10-02)
*Superseded by VCO-D15 (2026-10-02): no rule-based check judges an answer.*


Owner request: calibrate G1 on its own, per claim kind, at alpha = 0.10
(95 %, answer level). `calibrate_g1.py` ran the production state builder
(high effort) and the production G1 questions on 600 human-labelled answers
per kind, split by problem / document / page: RAGTruth for source and
action claims (any hallucination span), PRM800K phase 2 for computation (a
-1 step; 300 + 300 balanced), FEVER for general knowledge (REFUTES).

| Kind | AUROC | accepted at p >= 0.99 | violated |
|---|---:|---:|---:|
| G1-source (RAGTruth) | 0.694 | 157 | 37 (23.6 %) |
| G1-computation (PRM800K) | 0.704 | 185 | 74 (40.0 %) |
| G1-general (FEVER) | 0.893 | 251 | 38 (15.1 %) |

No threshold reaches alpha on source and computation; general reaches it on
the calibration half (tau 0.99926, 89 accepted, 4 violated) but not on the
held-out half (99 accepted, 15 violated, upper bound 0.224). Causes seen in
the data: RAGTruth counts true but unsourced additions as hallucinations
while G1 accepts well-established knowledge; OpenJev does not detect
arithmetic and reasoning errors; OpenJev is confident on false facts.

Owner decision (option A): the per-claim G1 questions stay, split by kind,
with threshold 0 and tag `guarantee: advisory` (since VCO-D12 only
G1-source remains). Their p is reported in
`kairyu_verification`, the Verification section and the answer page, but
they neither repair nor block the guarantee. The guarantee covers the
calibrated requirements (tau_hi) and the deterministic G1-excerpts, G2 and
G3. Also observed: 12 of 1,800 state-builder outputs were truncated JSON
(runaway newlines), which serving reports as `checklist_unavailable`.

### VCO-D12 — Latency target and the slimmer state builder (2026-10-02)
*Superseded by VCO-D15 (2026-10-02): no rule-based check judges an answer.*


Owner target: p50 <= 3 minutes on long InFoBench requests (every DeepSeek
step keeps the caller's effort, repairs stay at most two). Traced breakdown
(8 InFoBench requests, c8): the state builder took 92 s and 10,491 output
tokens per attempt, listing every claim although only source and action
claims feed a guarantee (G1-excerpts, G2); computation and general claims fed
only advisory, uncalibrated questions (VCO-D11). It now lists source and
action claims only; G1-computation and G1-general are gone. The step budget
(16) was below the two-stage worst case (18) and published answers
unverified (`reason: budget`); it is 24. Result: p50 492 s -> 297 s, state
builder 44 s / 5,109 tokens, guaranteed 1/8 -> 3/8. The floor is extraction
(implicit 110 s in parallel with extract 89 s) plus one attempt (answer and
state builder, about 70 s); repairs add about 70 s each.

### VCO-D13 — Instruction units read within the caller's format (2026-10-02)
*Superseded by VCO-D15 (2026-10-02): no rule-based check judges an answer.*


The structured gate's "Pick a European capital and describe it in exactly
the requested JSON" (fields city, country, population_estimate) was not
guaranteed: OpenJev read "describe it" as free prose, called the field
conditions unrequested (necessity p 0.005-0.02) and three fields an
incomplete description (sufficiency 0.01). Owner ruling: under a fixed
format, "describe it" means filling those fields; the answer deserves the
guarantee, and the fix belongs in how the requirement reaches Jev. The
extractor now appends to each unit that asks for content the format bounds
the scope to read it in ("describe it (the caller fixed the answer to a JSON
object with only the fields city, country and population_estimate; this is
done by filling them)"). Probes (5 reads each): all fields covered ->
sufficiency 0.85-0.93; population condition missing -> 0.11-0.18 (still
rejected); free-text requests unchanged. A shorter scope ("within the
caller's fixed JSON fields") did not work (0.03-0.07). Structured gate: two
runs, 4/4 guaranteed.

Amendment (2026-10-02, owner option A): six in-process runs per request
(live L1 / OpenJev, every exchange logged) showed three further causes.
(1) A grammar-constrained draft can degenerate (an integer repeating zeros
to max_tokens); repairing that 93,582-character draft thought to max_tokens
and returned nothing (3/3). The repair prompt now rewrites a cut-off or
degenerate draft from the conversation (3/3 valid). (2) An implicit
condition picked one reading the request leaves open ("a city proper, not
its metropolitan area"), so repair rewrote a correct Tokyo population; the
implicit extractor now lists only what every reasonable reading shares
(Japan request 3/4 -> 6/6 guaranteed). (3) Re-extraction dropped the format
scope; the refine prompt keeps it. Known limit: OpenJev's reading of
"describe it" against three fixed fields still swings with the
extractor's wording (necessity and sufficiency 0.02-0.98), so the European
capital request is guaranteed in 2-5 of 6 runs; a separate format unit and
three-sample reads did not steady it. Also seen once: the state builder
listed a JSON field as an "action" claim, failing G2.


Defect found by the requirements gate (2026-10-02): on a puzzle request
("a 9-digit lockscreen pattern ..."), the implicit extractor tried to solve
the puzzle, thought until max_tokens (32,768) and returned cut-off JSON; the
final checklist could not read it and the whole answer ended
`checklist_unavailable` (1 of 4 reruns). Both extractors are now told never
to work out the answer itself, and implicit_check writes a broken list once
more (max_refinements 1; worst case 20 steps, budget 24).

### VCO-D14 — The caller's tools reach the published draft (2026-10-02, issue #617)

*Amended by VCO-D16: with no seed, the final role writes the draft under the caller's contract.*

DeepSWE (mini-swe-agent, `tools=[bash]`) got HTTP 200 answers whose tool
calls were plain text (`<invoke name="bash">`, `Tool: bash`), so the agent
executed nothing; two turns ended HTTP 502 `EmptyFinalOutput`. Cause: only
the final unit (`answer`) carried the caller's tools, but its attempt 0 is the
generator's draft, which was generated without them; and `latest_checks_passed`
published an empty last repair over a non-empty draft. Fix (m1 D8
amendment): the seed of a seeded final unit is generated under the caller's
tool contract, and exhaustion never chooses an empty attempt when a non-empty
one exists. The repair prompt now tells DeepSeek to emit a tool call through
the tool interface, never as text.

Review amendment (PR #618): an empty repair that passes every item vacuously
is not accepted, so the draft is published as `refinement_limit`. The same
GPU rerun showed every later agent turn (about 42K-token conversations)
falling back to deepseek_think: the route judge's state exceeded OpenJev's
65,536-token context (HTTP 400). The judge now reads at most 120,000
characters of conversation (`max_conversation_chars`, m1 D9 amendment): the
first message plus the newest that fit; a 111-message DeepSWE conversation
bounded this way is 37,716 OpenJev tokens.

Second GPU rerun (5376ec73): no judge 400s, every response a structured tool
call, but long turns on the verified route ended `checklist_unavailable`:
their conversation state was 526,443-651,415 characters against
`max_state_chars: 160000`. Every checklist's conversation section now sets
`max_total_chars: 100000`, leaving 60,000 characters for the other sections.

### VCO-D15 — Points, adoption and coverage, all read by models (2026-10-02, PR #618)

*Amended by VCO-D17: items 5 and 6 (agent turns) no longer apply; tool requests take the verified-tool route.*

*Amended by VCO-D16: an empty implicit list asks nothing and passes (no `on_empty`); the judge reads the reply's text with its calls (the Chat API returns both); a rejection with every point met is not repaired.*

Owner decision. The guarantee was LLM-based to overcome the limits of rules,
yet the Validator added rule-based checks (G1-excerpts, G2, G3, extracted
`deterministic` conditions, S0/S1 coverage). These were not requirements of
any request, and on agent turns G3 misread tool-call JSON as quotations: all
ten verified DeepSWE turns failed it, no OpenJev read ran, and two repairs
were wasted per turn (305-554 s). Every rule-based check is removed from the
example and from the framework (m1 D8 amendment).

The goal is unchanged: the answer meets a requirement set that is MECE with
respect to the request.

1. **Points (mutually exclusive, collectively exhaustive).** `extract`
   lists the explicit points; then `implicit` reads them and lists only
   presupposed points none of them already requires (amendment 2026-10-03,
   owner decision: run in parallel, the two lists overlapped and 23 of 40
   InFoBench requests had a duplicate pair, against the gate's 10 %; the
   cost is the implicit extraction time before adoption). Each point is one
   issue, and together they cover the request. Both prompts name the
   overlaps the requirements gate found (amendment 2026-10-03, owner
   decision, after 6 of 40 requests had a duplicate): a requirement restated
   from another angle (a heading and its section's content, a table's
   columns and rows) is one point, and an implicit point never restates an
   explicit one or its converse. A Jev check of the explicit
   list (necessary per point, sufficient for the list, re-extracting once
   on a failure) was tried and removed (owner decision, 2026-10-03): an
   overlap question failed nearly every point with no duplicate found, and
   on 80 InFoBench requests the sufficiency question did not track gold
   coverage (AUROC 0.41; at 0.5 it re-extracted 20 lists, 18 already
   complete, and passed 7 of the 9 incomplete ones). The first lists
   already covered 96.2 % of the gold requirements; the recall loss came
   from adoption dropping explicit points (item 2). Adoption with no
   implicit point is skipped (`on_empty: pass`); before, it was unavailable
   and the whole run unverified (14 of 66). The requirements gate
   measures coverage of InFoBench's gold questions and duplicate pairs.
2. **Adoption (necessary).** One Jev request asks, for every implicit
   point, "is this point necessary to answer the request?"; explicit points
   are what the user asked for and always stay (amendment 2026-10-03, owner
   decision: judging them too dropped stated requirements, e.g. an
   obituary's name, age and date at p 0.16-0.47, and InFoBench gold recall
   fell from 0.972 to 0.891). The state is the request verbatim
   (system/developer messages plus the latest user message) and `history`, a
   DeepSeek summary at the caller's effort (amendment 2026-10-03, VCO-D9:
   the effort gate found it the one step without it) of every other message
   (earlier turns, tool calls and tool results). Summarizing these instead of
   passing them verbatim keeps the read inside OpenJev's 65,536 tokens.
   Points with p < 0.5 leave their list. `history` waits for both extractors
   so that `adopt`, its verifier, reads both lists in one request.
3. **Coverage.** One Jev request asks, for every adopted point, "does the
   answer fully and correctly do what this point requires?". The state is the
   answer exactly as it will be sent (tool calls as `{text, tool_calls}`, so
   the judge reads them as calls the caller executes), with the request
   verbatim and the history summary. Every p >= tau_hi gives the guarantee. A miss is repaired by DeepSeek
   at most twice; otherwise the last non-empty answer is returned as
   `refinement_limit`.
4. **Draft (amendment, owner decision).** `generator` runs after adoption and
   writes the draft from the conversation plus the adopted points, told to
   meet every point. This replaces VCO-D1's draft "from the conversation
   only": the draft aims at the requirement set it is judged against. The
   cost is the extraction and adoption time before the draft starts.
5. **One unit for every stage (amendment, DeepSWE r2).** The extractors
   list points about the reply given now, but adoption asked whether a point
   was "necessary to answer the request" with only the request and history:
   on agent turns the request is the whole task, so Jev dropped the step's
   points (p 0.0-0.5) and four turns kept none. Adoption now asks "must the
   reply the assistant gives now meet this point?" and also reads the
   caller's tool definitions. A checklist with no item to judge is
   unavailable, never a pass.
6. **Agent turns.** A request with `tools` is answered by one assistant
   message, which may hold several tool calls. The extractors read the tool
   definitions (`{tools}`) and list what this one message must do now, never
   the completion of the task.
7. **Acceptance (amendment 2026-10-03, owner decision).** After the coverage
   read, one more Jev request reads the coverage results (each point's p and
   pass), the original prompt and the answer, and asks whether, reading
   these, the answer can be adopted as the reply the user expects (owner
   wording; "may it be adopted as the official reply" measured AUROC 0.787,
   this 0.809). Its P(yes) against tau_accept decides
   the guarantee; tau_hi only selects the missed points. A rejected answer is
   repaired with the original prompt, the answer and the missed points, with
   an explicit instruction to rewrite it to meet them. The judged answer is
   what the API publishes: with tool calls, the text is dropped and only
   calls the caller's tools and `tool_choice` allow count. Why: on DeepSWE r3
   a point-wise conjunction failed on single misjudged points (a correct
   bash call read at low p) and on points about text the API never
   publishes, costing repairs; the owner wants the adoption decision made
   by the judge over the whole answer, informed by the point results.
   On InFoBench response labels (all labelled requirements met) no
   threshold meets alpha = 0.10 at 95 % for any of six measured question and
   state variants (best calibration-half upper bound 0.259); nor does the
   point-wise conjunction (0.396). The owner set tau_accept = 0.99, the
   lowest measured error: held-out 43 of 125 answers guaranteed, 7 missing a
   labelled requirement (16.3 %, upper bound 28.4 %). The guarantee is
   therefore the judge's adoption at 0.99, not an alpha-bounded claim.
8. **Four denoise passes (amendment 2026-10-03, owner decision).** Every
   Jev checklist read (adopt, coverage, acceptance) sets `steps: 4`:
   OpenJev re-reads each answer slot with the other slots' current answers
   in place, instead of filling every slot at once. With one pass, an agent
   turn with 11 points read a present bash call at p 0.37-0.67 (0.99 with
   four questions); with four passes, 0.90-0.99. Recalibrated on InFoBench:
   tau_hi 0.99894 (calibration upper bound 9.78 %, held-out 7.84 %);
   tau_accept stays 0.99 (held-out 7 of 46, 15.2 %; acceptance AUROC 0.794).

tau_hi is recalibrated on InFoBench for the coverage question and its state
(`calibrate.py`, alpha = 0.10 at 95 %); the per-claim G1 calibration
(`calibrate_g1.py`) is removed with G1. The owner's first form (the answer
alone, "does the answer contain this point?") reached AUROC 0.791 and failed
alpha on the held-out half (upper bound 12.4 %). Five variants were measured on
the same point statements; the owner chose the best, which adds the request
and the history summary to the state and asks the stricter question: AUROC
0.852 (the earlier design: 0.850), tau_hi 0.9895, held-out upper bound 8.6 %,
45 of 125 held-out answers pass every point (MEASUREMENTS.md).

DeepSeek runs with `disable_any_whitespace` (xgrammar): a grammar-constrained
extractor once emitted whitespace inside its JSON until max_tokens, leaving
its list unreadable and the turn `checklist_unavailable` (DeepSWE, PR #618).

### VCO-D16 — An agent turn is verified as one step (2026-10-03, PR #619)

*Superseded by VCO-D17 (2026-10-04): the step route and its verification are removed.*

Owner decision. On closed PR #618's DeepSWE runs, 73 of 83 verified agent
turns ended `refinement_limit` and repairs changed the agent's commands in 51
of 74 repaired turns, once replacing an exploring step with the submission
command. Causes: every stage judged a turn as a finished answer to the task;
the Chat API dropped the reply's text next to its tool calls (fixed in L3,
m1); the repair prompt framed DeepSeek as an editor outside the conversation;
spurious misses (points read at p 0.998 against tau_hi 0.99894, rejections
with no failing point) triggered repairs.

1. **Criterion (A0).** A reply in an ongoing tool-using task is judged as
   the next step: it uses the latest tool results correctly, moves the task
   forward, rests on no wrong premise and takes no destructive action, and
   neither submits nor declares completion while the work is unfinished.
   Completion is never required. Answers without tools keep VCO-D15.
2. **Routing.** The Jev route judge gains the label STEP (tools offered, an
   ongoing task, the reply is the next action) for the `verified_step`
   profile; `kairyu-verified` offers THINK/STEP/VERIFIED and
   `kairyu-verified-always` STEP/VERIFIED (fallback: the answer DAG). The
   VERIFIED floor rises from 0.3 to 0.5 (owner decision), so VERIFIED is
   preferred only when it is also the most probable route and never takes
   a step from STEP; on the routing set no accuracy-critical conversation
   read below 0.987. The routing gate now applies the served three-way
   rule and fails when 10 % or more of its conversations (none with tools)
   go to STEP.
3. **verified_step.** `step_extract` lists the step's points from the task
   and the latest tool results (facts the step must take into account, what
   moves the task forward, and whether submission is allowed — only when the
   conversation shows the work done and verified); `step_implicit` what the
   step presupposes (an environment-appropriate, non-destructive,
   non-repeating action); `step_adopt` keeps the points the next step needs.
   Coverage and acceptance both read the request, the recent conversation
   verbatim (the bounded `query`: first and newest messages), the summary
   of earlier work and the reply, and ask whether the reply, as the next step, does what
   each point requires and is a sound next step.
4. **Repairs (both profiles).** The repair is the same message written
   again in the draft's frame (the conversation first, then every adopted
   point), followed by the draft and its missed points, changing only what
   they require and keeping the rest, tool calls included. For a step, a
   missed point is one read below 0.5 (coverage threshold 0.5); the answer
   profile keeps tau_hi. A rejection with every point met is published
   unverified (`reason: not_accepted`, m1 D8 amendment).
5. **Thresholds.** 164 recorded DeepSWE replies (82 turns, the replayed
   reply and the recorded one) were labelled by the A0 criteria, blind to
   Jev's probabilities: 4 unsound (three patches using a type the latest
   results showed absent, one submission before any work). Acceptance 0.5
   guaranteed 3 of them; 0.99 none, with 24 of 160 sound replies sent to
   repair or published unverified (0.999: 72). Acceptance is 0.99; a missed
   point stays p < 0.5. Asking Jev four direct problem questions instead
   caught only the submission (1 of 4).
6. **Framework.** The DAG keeps to the minimal L2 (m1 D8 amendment): the
   `answer` role writes the draft itself; `adopt` verifies `implicit` and
   curates that one list; `history` runs beside `extract`. The extractors may
   use 65,536 tokens (six of 41 long-conversation turns had an empty implicit
   list).

Measured before a full run: the 83 recorded turns replayed (routing, how
often repairs happen, every changed call mapped to a missed point, no lost
call, no submission introduced by a repair, text present, no empty implicit
list), latency p50 <= 180 s (VCO-D12), the GPU gates, and a DeepSWE subset
where verified scores at least as think with no early submission.

### VCO-D17 — A verified-tool route instead of step verification (2026-10-04, PR #619)

Owner decision. The route judge offers a third route, VERIFIED_TOOL, for a request
that requires a tool call (the caller offers tools and the reply is expected
to call one: an agent loop's turn, or a request to act with the tools). It
is answered by one DeepSeek call at max effort with the caller's tools and is
not verified. `kairyu-verified` offers THINK/VERIFIED_TOOL/VERIFIED and
`kairyu-verified-always` VERIFIED_TOOL/VERIFIED. VCO-D16's step profile and STEP
route are removed. GPU gates: `routing` fails when 10 % or more of its
tool-free conversations go to VERIFIED_TOOL; `verified-tool-routing` (40 conversations that
offer tools, 20 requiring a call) needs at least 90 % of the requiring ones
on VERIFIED_TOOL and under 10 % of the others; `verified-tool-route` needs every requiring
conversation, unary and streamed at any caller effort, to return structured
tool_calls from one DeepSeek call at max effort without verification;
`fallback` adds a tool request with both judges down; `serving-routed` mixes
both sets and bounds the verified-tool route's judge read at p50 2 s.

Why: verifying a correct intermediate agent step against the request's
requirements failed it, and its repair jumped to the final move (DeepSWE,
closed PR #618); step verification did not remove that risk. Unverified
direct tool answers at max effort are the baseline until a verification that
lets correct intermediate steps pass is found.

The verified DAG drops its agent-turn wording (owner decision): the
extractors no longer read `{tools}` or list points for "this one message
of a tool-using loop"; adoption asks again "is this point necessary to
answer the request?" over the request and the summary (no tools); the
summary covers earlier turns; the repair rewrites the reply without
tool-call instructions; the extractors' limit returns to 32,768 tokens
(16,384 low, 65,536 max).

VERIFIED_TOOL's criteria name when a call is needed (a first tool action, a
retry after a tool error, reading or checking more, or a conversation that
requires a call every turn) and exclude a reply that only reports tool
results already in the conversation (owner-approved, 2026-10-04): offline,
tool-free conversations to VERIFIED_TOOL fell from 35 % to 5 %, requiring
ones stayed at 95 %, the routing set's held-out miss stayed 8.3 % and 73 of
the 83 recorded DeepSWE turns (was 65) route to VERIFIED_TOOL.

A second wording (owner-approved, 2026-10-04) frames VERIFIED_TOOL as "the
next reply must return a tool call to make progress" and also excludes a
question the assistant can answer from its own knowledge. Offline, of five
wordings it alone met every bound: tool-free to VERIFIED_TOOL 5 %, requiring
90 %, routing held-out miss 4.2 %, recorded DeepSWE turns 78 of 83.

The route judge picks the most probable label; the VERIFIED floor
(`prefer`) is removed (owner decision, 2026-10-04).

Recalibration after this change (owner decision, 2026-10-04): with every
DeepSeek point statement and summary regenerated, tau_hi is 0.999733
(calibration upper bound 9.66 %, held-out 9.26 %; was 0.99894, 9.78 % /
7.84 %). 79 of 249 regenerated statements differed from the cached ones
(temperature 0 is not reproducible); points with unchanged statements read
the same (median |dp| 0.0002). Held-out responses passing every point fell
from 39 to 23 of 125; acceptance 0.99 guaranteed 39 with 4 violating (10.3 %,
was 46 and 7, 15.2 %).

### VCO-D18 — Winnow-routed answers; the guarantee is rebuilt later (2026-10-06, PR #640)

Owner decision. The example is rebuilt on main as
`examples/deepseek-v4.1-qwen3.8-winnow-8gpu`:

- L1: DeepSeek-V4.1-Flash DP6/EP6 on GPUs 0-5 (unchanged), Qwen3.8-27B FP8
  on GPU 6 (as in `qwen3.8-27b-1gpu`), Winnow-12B Q8_0 on GPU 7 (as in
  `winnow-12b-q8-1gpu`, chat and System One). OpenJev is removed.
- Routing: Winnow replaces OpenJev as the System One route judge with the
  same question and criteria, minus what existed only for the verified-tool
  route; it chooses THINK or VERIFIED (most probable wins, fallback THINK).
- Routes: THINK is `deepseek_think` (the caller's effort, unchanged);
  VERIFIED is, for now, one DeepSeek call at max effort. Both pass the
  caller's tools.
- Removed: the verified-tool route and everything attached to it (supersedes
  VCO-D17); the checklist DAG (extract, history, implicit, adopt, answer,
  coverage, acceptance, repair) and its calibration, so no answer carries
  `kairyu_verification` until the guarantee is rebuilt.
- Qwen is served as an internal pool that no route references yet.
- Open WebUI and the answer page carry over unchanged; the answer page's
  guarantee panel stays empty for now.
- GPU gates are replaced: l1, routing (VERIFIED miss < 10 % on both halves,
  as before), think-route, effort, verified-route, fallback (Winnow down),
  serving and serving-routed (c1/c4/c8/c16 with 8/16/16/32 requests, as
  before), browser.

Why: the owner restarts the verified route from a plain max-effort DeepSeek
answer on an eight-GPU layout that also hosts Qwen and Winnow, keeping the
previous routing conditions and serving levels as the verification baseline.

### VCO-D19 — Verified route in three waves (2026-10-06, PR #641)

Owner decision. The VERIFIED route (both public models) becomes:

- Wave 1, in parallel: DeepSeek writes five complete answers D1..D5 from
  viewpoints as different as possible, in one call at the caller's effort
  (default high; JSON with fixed keys, so each draft sees the others'
  viewpoints); Qwen (thinking at low, whatever the caller sends)
  lists the requirements an answer must meet, necessary and sufficient and
  MECE, at most 16 (JSON).
- Wave 2: one Winnow System One request with the request and the drafts in
  the state: per draft, can it be adopted as is as the final reply (D1..D5);
  per requirement x draft, is it met (Dk-Rn). Winnow never repairs; threshold
  1.0 lists every judgment below certainty with its probability for wave 3.
  An unreadable judgment leaves wave 3 without judgments.
- Wave 3: DeepSeek at the caller's effort reads the request, the drafts, the
  requirements and the judgments, treats all of them critically, and writes
  the best reply (the final unit, with the caller's tools and format).
- No answer carries `kairyu_verification`: the judgments inform the answer.
- Wave-1 parallelism uses the m1 D8 amendment of 2026-10-06 (a verifier may
  wait for a unit running beside its target).
- GPU-verified 2026-10-06: all nine gates pass (example `MEASUREMENTS.md`).
- Gates: effort expects drafts and answer at the caller's effort and Qwen at
  low; verified-route checks the
  four stages, wave-1 overlap and 5 + 5 x N judgment items; all gates re-run.

Why: the owner rebuilds the verified route as diverse drafts judged by
Winnow against independently listed requirements, then a critical synthesis.

**Amendment (2026-10-07, PR #641).** Owner decision: Qwen `requirements`
reads `{conversation_without_reasoning}`; DeepSeek `drafts` and `answer` keep
the replayed `reasoning_content`; Winnow `judgments` reads `request` (no
assistant turn). Why: DeepSWE r1 replayed Kairyu's stage reports in
`reasoning_content` (189,953 of 240,650 Qwen tokens on one turn), so Qwen's
262,144-token context overflowed and 87 of 160 VERIFIED turns had no Winnow
judgment; without it the failed turns measured 30,330-96,837 tokens.

**Amendment (2026-10-07, PR #641): DeepSeek requirements, two Winnow
replicas.** Owner decision; supersedes the amendment above.

- `requirements` is written by DeepSeek at max effort, whatever the caller
  sends (same sampling as the other DeepSeek roles, output cap 131,072), and
  reads `{conversation}` with the replayed reasoning.
- Qwen leaves the example: no worker, pool, compose service or template. The
  example is renamed `examples/deepseek-v4.1-winnow-8gpu`.
- GPU 6 hosts a second Winnow-12B replica. Each replica serves one System One
  use: `winnow-route` (GPU 6, `winnow-route-systemone`) only the route judge,
  `winnow-judge` (GPU 7, `winnow-judge-systemone`) only the judgments. Both
  stay in the internal `winnow-12b` chat pool.
- Gates: effort expects drafts and answer at the caller's effort and
  requirements at max, and the route judged on `winnow-route`;
  verified-route expects the judgments on `winnow-judge`; fallback stops
  each replica alone (route down: think route while judgments still run;
  judge down: still routed, answer without judgments).
- GPU-verified 2026-10-07: all nine gates pass (example `MEASUREMENTS.md`);
  verified-route p50 141.6 s (Qwen layout 162.7 s), serving c1/c16 p50
  98.3/151.4 s (Qwen layout 51.5/118.6 s).

Why: DeepSWE r1 of the three waves (24 of 113 tasks scored, 8 passed) showed
Qwen `requirements` as the wave-1 bottleneck (median 107 s against 36 s for
the drafts). Separate Winnow replicas keep a burst of judgments from queueing
the next request's route decision. With Qwen gone, the m1 D8 placeholder
`{conversation_without_reasoning}` has no user and is withdrawn.

**Amendment (2026-10-07, PR #641): the next step, structured tool calls,
no stage reports.** Owner decision after the content of DeepSWE turns on the
two-Winnow layout (stopped at 0 of 113 scored).

- The drafts, the requirements and the answer define the reply as the next
  assistant message. When the conversation ends with tool results or the
  assistant's own turns, the reply is the next step of that work: the move
  needed now, not the request's final result and not a step whose result
  the conversation already shows. The requirements list what that next step
  must meet; the request's final goal is context.
- Each draft carries `tool_calls` (name, JSON arguments) and reads the
  caller's tools; its text never holds a call. The answer makes calls
  through the tool-calling interface only.
- The judgments read the whole conversation as well (each message cut at
  4,000 characters, the whole at 120,000, as for the route judge) and the
  caller's tools (PR #641 review), and ask whether each draft (text and tool
  calls) can be adopted as the next reply.
- `expose_intermediate_outputs: false`: stage reports leave
  `reasoning_content`; the answer page's "Internal stages" panel is empty.
- GPU-verified 2026-10-07: all nine gates pass; replayed DeepSWE turns return
  structured tool calls (10/10) and next-step requirements (example
  `MEASUREMENTS.md`). Verified-route p50 181.6 s (141.6 s before).

Why: the drafts wrote calls as text (`[Makes bash tool call with ...]`, 35 of
54 verified turns) and the answer copied it in 3 of 47, so the agent rejected
the turn (one 639-second implementation was lost; r1 had 19 of 655). The
requirements read the latest user message, which in an agent conversation is
always the task, so they listed the whole task every turn; the judgments
failed nearly all 85 items and the agent re-explored instead of taking the
next step (owner's problem definition: choose the move needed now). The
replayed stage reports (40,000-130,000 characters per turn) grew every
request and let the next turn read last turn's drafts.

### VCO-D20 — The verified tool route (2026-10-09, PR #641)

Owner decision after DeepSWE `deepswe-verified-3wave-nextstep-4w-20261007-r1`
(stopped at 75 of 113 scored, 35 passed). Plan:
`docs/superpowers/plans/2026-10-09-verified-tool-route.md`.

- One public model, `kairyu-verified-tool` (spec `verified-tool.yaml`);
  `kairyu-verified`, `kairyu-verified-always` and `verified-always.yaml` are
  removed, and so is the answer page (`playground/`). Open WebUI stays.
- The route judge asks whether the next reply needs to call one of the
  caller's tools: `TOOL` takes the three waves (profile `primary`), `THINK`
  takes `deepseek_think`; fallback stays `deepseek_think`. The judge state
  already carries `tool_calling` beside the conversation. The VERIFIED route
  for requests without tools is gone; the think route serves them.
- The DSL stays: the same roles, workers, efforts and schemas. Only prompts
  change, kept generic for any tool-calling agent:
  - drafts and requirements share one reply definition: the call(s) for the
    one move needed now with at most a short text; the finishing move only
    once tool results show the request done;
  - requirements read the tools and, from the request's literal words and
    the current position, cover exact names and public entry points, checks
    taken from the request (the request wins over a check), verification
    through the public path, build mode and type checks that existing checks
    use ("all" checked on more than one case), must-not behaviour, and order
    and finishing (finish alone, after success is shown);
  - drafts: five different moves; near the end, one checks the work against
    the request before finishing; never announce a call without making it;
  - judgments: adoptable means the move needed now, calls that fit the tools
    and the protocol, and text that matches its calls;
  - answer: makes every call it announces; the finishing move goes alone,
    after tool results show success.
- Gates: `routing` uses `datasets/tool-routing-set.json` (40 TOOL turns; 96
  THINK: the 80 earlier chats without tools and 16 tool-declared turns whose
  next reply is text), TOOL miss rate < 10 % on both halves and THINK
  precision >= 90 %; `verified-route` becomes `verified-tool-route` on
  `datasets/tool-turns.json` (24 agent turns), which also needs a structured
  call to a declared tool; `serving` runs those turns and `serving-routed`
  the routing set; `l1` ends with one tool-call answer; the browser gate
  checks Open WebUI only. The InFoBench serving set is dropped.

Why: the 75 scored tasks failed on the public surface differing from what the
request names (about 9), on checks that encode the agent's misreading (about
8), on verification at an internal layer instead of the tests' path or type
check (5), on missing must-not behaviour (3), and on a finishing action
batched with an unseen failing step (2); 9 replies announced a call without
making it. Running tests did not separate passes from failures; what was
checked did. The owner keeps the verified route for tool calls only and
routes every other request to the think route.

**Amendment (2026-10-09, PR #641): the route judge on long agent runs.**
Owner decision after replaying DeepSWE turns before the GPU gates: 20 of 64
real agent turns, each needing a tool call, were routed THINK, almost all
near the end of the run; 4 of 6 concurrent long turns hit the 10 s cut; and
the `routing` gate failed (TOOL miss 15 % on both halves).

- The judge's question adds that when the conversation's instructions
  require a tool call or command in every reply, or end the work with one,
  every reply up to that final one needs a call; THINK applies only when
  the conversation accepts a plain-text reply; TOOL names finishing calls.
- The judge's conversation bound keeps the request's messages (m1 D9
  amendment 2026-10-09), so the task and protocol stay visible.
- `timeout_seconds` 10 -> 60 (the DSL maximum).
- One authored routing item had no edit step before its check; it was fixed
  (label unchanged).
- Measured before the gates: real-turn misses 21 -> 1 of 64; authored set
  TOOL misses 2 of 40 (one per half), THINK sent to TOOL 2 of 96.

Why: the old THINK criterion "final report once the work is done" pulled an
agent's last turns to the think route, whose replies then wrote the submit
command as text or batched it with a commit: the failures this route is for.

**Amendment (2026-10-09, PR #641): calls DeepSeek writes without markers.**
Owner decision after `verified-tool-route` failed 1 of 12 (the type-check
turn, streamed, effort max): the answer wrote DeepSeek's DSML call without
its marker tokens, so vLLM returned it as text. Kairyu now returns such a
trailing block for a declared tool with schema-valid arguments as a tool
call (m9 D2 amendment 2026-10-09). The example's prompts are unchanged.

## Limitations

The items below describe the checklist configuration (VCO-D1..D17), which
VCO-D18 removed.


- A guaranteed answer is not streamed before its checklist finishes (time to
  first token is the whole pipeline).
- The routing set is author-labelled with clear-cut categories; borderline
  requests are not measured by it.
- The 0.5 necessity cut of the adoption read is a default, not calibrated.
- The guarantee covers the adopted points, not the truth of every claim in
  the answer.
- An extractor cut off before its JSON closes leaves its list unreadable and
  the answer unverified (`checklist_unavailable`); there is no rewrite.
- With tools, each assistant message is judged as one step; whether the
  whole agent run solves the task is not part of the flag.
