# Qwen3.8 + DeepSeek V4.1 ensemble on eight GPUs

**GPU validation is in progress; see the measured results and remaining gates below.**
Do not interpret the sibling V4.1 TP8 measurements as evidence for this topology.
See [MEASUREMENTS.md](MEASUREMENTS.md) for the current evidence and
[L1-NOTES.md](L1-NOTES.md) for the fixed-runtime inspection.

This separate example keeps the original five-route L2 judge while assigning
six GPUs to DeepSeek-V4.1-Flash and two GPUs to Qwen3.8-27B-FP8. It ports
Requirement extraction from PR #595 (`31f1adc`) onto DeepSeek with a high
minimum effort: API max is preserved, while omitted/low/high use high.
The original example configuration is preserved. Shared Kairyu changes add an
opt-in paragraph separator and correct stream cleanup after client disconnects;
see V41E-D5/D6 in the design and the regression evidence in MEASUREMENTS.

## Topology and behavior

| Resource | Configuration |
| --- | --- |
| GPU 0–5 | One DeepSeek service: TP2 × internal Attention-DP3, EP6, PP1 |
| GPU 6, 7 | Two Qwen3.8 27B FP8 replicas, TP1 each |
| Public API | `http://HOST:8008/v1`, model `kairyu-auto-max` |
| Embeddings | `embed-small`, CPU FastEmbed |
| Chat UI | `http://HOST:3008`, no-login example UI |
| DeepSeek L1 | `http://127.0.0.1:8009/v1`, host loopback only |

Internal Attention-DP3 is vLLM parallelism inside the one DeepSeek service;
it is not three Kairyu copies of the full six-GPU worker. The two logical
DeepSeek pools select thinking/non-thinking modes against that same endpoint.
Qwen placement retains prefix indexing and the zero queue-depth overload valve.

TP6 is invalid for 64 attention heads. The selected TP2/DP3/EP6 configuration
passes static divisibility/source checks, but memory fit, collectives, kernels
and numerical behavior require hardware validation. The initial GPU-resident
Engram configuration failed memory profiling; the current candidate uses the
pinned runtime's Engram CPU offload (about 189 GiB of host RAM for the tables).
The measured host has 1 TiB RAM; reserve table memory in addition to the models'
loading/serving overhead and other processes. DSpark is disabled because its
128 draft experts do not divide EP6. Qwen MTP remains disabled.

The judge still selects Qwen direct, Qwen thinking-medium, DeepSeek direct,
DeepSeek thinking, or the ensemble, with ensemble fallback on judge failure.
Both models accept images; no image-description bridge is used, and the
DeepSeek profiles are eligible for image requests. The public image envelope
is constrained by Qwen's one-image limit (8 MiB, 2,097,152 pixels).

The ensemble runs these dependencies:

```text
head (public opening) ───────────────────────────────────────┐
draft ─────────────────┐                                    │
requirements ──┬───────┴─> critique (improved draft) ─────────┤
               └─> policies (exactly two) ─> answer_1 ───────┤
                                         └─> answer_2 ───────┤
                                                            v
                                         synthesis -> Qwen audit
                                              ^          |
                                              └─ repair ─┘ (at most twice)
```

The checklist is passed explicitly to policies, answers, critique, synthesis
and audit. Head and the quick draft remain independent roots. Direct profiles
bypass Requirement, matching PR #595's scope. The checklist has stable `R1…`
IDs and `priority`, `requirement`, `acceptance_criterion`, and `source` fields.
It is evidence derived from the original request, not a higher-priority instruction.

| Role | Effort |
| --- | --- |
| DeepSeek Requirement | Native high for omitted/low/high; native max for API max |
| DeepSeek policies, critique, synthesis, thinking-direct | Inherit API effort; omitted means native high |
| DeepSeek direct | Non-thinking |
| Qwen draft, answer_1/2, audit, thinking-direct | Existing fixed medium (DSL `high` → Qwen `medium`) |
| Qwen head, route judge, direct | Existing non-thinking |

The native V4.1 tokenizer/role hook replaces the old inline text scaffold.
It applies the Requirement high floor to the resolved top-level API effort
and writes the result to both top-level and nested template fields. Nested
template kwargs cannot override that decision. The pinned encoder maps high
to official high75 and max to max100. Its total allowance remains 8192
tokens, with thinking capped at `min(4096, total/2)` to reserve checklist output.
The generation schema omits string `minLength` because pinned XGrammar 0.2.6
otherwise rejects quote/backslash/newline escapes. Smoke/quality checks still
reject empty fields; the serving grammar itself now permits them (V41E-D7).
Synthesis and thinking-direct reserve `min(256, total/2)` for public output
through vLLM's thinking budget processor, rather than an unsupported assistant
prefill. These reservations never increase caller or role token limits.
Qwen draft and policy answers also reserve half their existing total allowance
for the candidate body: at most 1024 thinking tokens for draft and 2048 for
each answer. This keeps their effort at medium. The initial GPU run exhausted
one answer's entire allowance in thinking, so an empty candidate must fail the
probe even when the overall request returns successfully. See MEASUREMENTS for
the native enforcement evidence and composed revalidation status.

The head opts into `continuation_separator: "\n\n"`. After exact-prefix and
`NO_CONTINUATION` handling, the publisher inserts this whitespace only when
the head and a nonempty continuation touch without whitespace. It preserves
existing model whitespace, head-only exact answers and headless tool/JSON
responses. The default is empty, so older deployments keep byte-exact joining.
The inserted separator is formatting; generated-token usage remains unchanged.

The audit uses PR #595's verdict/assessment format and at most two refinements.
The existing behavior can publish the last answer after failed/exhausted audit,
and the opening may already have streamed. This is model-based checking, not
a guarantee that every published answer satisfies every minimum requirement.

## Prepare another machine

Use the entire repository checkout at the PR commit; the example references
the sibling V4.1 runtime build and the existing CPU sandbox build. Copying only
this directory is insufficient. Required host facilities are Linux x86-64,
eight RTX PRO 6000 Blackwell Server Edition GPUs with at least 90,000 MiB each,
a compatible NVIDIA driver/container toolkit, Docker Compose v2, Python and uv,
network access for pinned images/models, and writable `/mnt/nvme` storage.
The current preflight requires at least 650 GiB free, including when reusing
cached models. Confirm the host memory/disk envelope before downloading.

From the repository root:

```sh
uv sync --frozen --dev
uv run pytest tests/unit/test_v41_ensemble_*.py --no-cov -q
uv run ruff check examples/qwen3.8-deepseek-v4.1-8gpu tests/unit/test_v41_ensemble_*.py
./examples/qwen3.8-deepseek-v4.1-8gpu/run.sh config
./examples/qwen3.8-deepseek-v4.1-8gpu/verify.sh list
```

`config`, `status`, and `down` do not prepare NVMe storage or generate a key.
They use the new Compose project only. `run.sh up` prepares storage, attests or
downloads models, pulls/builds pinned images, creates a private compaction key,
discovers NUMA affinity, starts the stack, and checks public readiness/routing.
It does not stop any other example to free GPUs. Run it only after all eight
GPUs are available for this experiment.

```sh
export NVME_STORAGE_ROOT=/mnt/nvme/kairyu
export PUBLIC_HOST=your-hostname
./examples/qwen3.8-deepseek-v4.1-8gpu/run.sh up
./examples/qwen3.8-deepseek-v4.1-8gpu/run.sh status
./examples/qwen3.8-deepseek-v4.1-8gpu/run.sh logs
```

Optional endpoint overrides: `API_PORT`, `CHAT_UI_PORT`, `DEEPSEEK_L1_PORT`,
`API_BIND_ADDRESS`, `CHAT_UI_BIND_ADDRESS`, and `WEBUI_URL`. `HF_TOKEN` is passed
only to model download containers when present. Checkpoint revisions, tree
hashes, runtime source and image pins are in `example.json`.

Storage relative to `NVME_STORAGE_ROOT`:

- `model-volumes/deepseek-v4.1-flash-8gpu/models`: sibling-compatible V4.1 model tree.
- `model-volumes/qwen3.8-27b-1gpu/models`: sibling-compatible Qwen model tree.
- `model-volumes/qwen3.8-deepseek-v4.1-8gpu/`: isolated compile caches, placement
  logs, WebUI data, private compaction key, benchmark temporaries and verification.

Set `VERIFY_MODEL=1` to recompute the cached checkpoint tree attestation. The
DeepSeek runtime builds in two layers. The launcher first builds/attests the
sibling V4.1 SM120 image using that sibling's directory as context, then applies
this directory's `patch_masked_kv.py` through its own Dockerfile. The patch checks
the original FlashInfer header hashes before editing. It prevents invalid sparse
KV indices from reading potentially nonfinite data in recycled slot zero.
The child also applies the source-pinned `patch_top_p.py` cutoff guard so seeded
positive-temperature sampling can emit forced thinking terminators (V41E-D8).
The versioned `compile-cache/deepseek-masked-kv-v1` directory keeps old generated
kernels out of the new runtime; FlashInfer uses a versioned subdirectory too.

On another machine a rebuild may produce a different Docker image ID. The
launcher fails closed; inspect the exact parent/child build and attest the parent
in its sibling spec, then update the child's `example.json` and both DeepSeek
descriptors in `kairyu.yaml` together before testing. Record the new image IDs and
source hashes in that machine's evidence. Do not disable the check or claim that
source-equivalent image bytes were already GPU-validated. To preserve the exact
tested images, transfer them with `docker image save` / `docker image load` and
confirm `docker image inspect --format '{{.Id}}'` against both pinned specs.

## GPU validation: execution order

1. **Inventory/provenance.** Record checkout SHA, `git status`, `nvidia-smi -q`,
   `nvidia-smi topo -m`, driver/Docker versions, model attestations, image IDs and
   the rendered Compose configuration. Preserve private state securely; redact
   the compaction key from any recorded environment or configuration output.
2. **DeepSeek alone.** Use the same lifecycle preparation methods to obtain the
   environment and images/models, then start only the `deepseek` service:

   ```sh
   uv run python - <<'PY'
   import importlib.util
   from pathlib import Path
   path = Path('examples/qwen3.8-deepseek-v4.1-8gpu/control.py')
   spec = importlib.util.spec_from_file_location('control', path)
   ctl = importlib.util.module_from_spec(spec)
   spec.loader.exec_module(ctl)
   env = ctl._compose_env(prepare=True)
   ctl._preflight(env)
   for key, name in [('QWEN_VLLM_IMAGE', 'qwen'), ('DEEPSEEK_VLLM_IMAGE', 'deepseek')]:
       ctl._ensure_vllm_image(env, key, ctl.SPEC['vllm'][name])
   ctl._ensure_models(env)
   ctl._prepare_deepseek_cache(env)
   ctl._run(['docker', 'compose', '--project-directory', str(ctl.HERE),
             '-f', str(ctl.HERE / 'compose.yaml'), 'up', '-d', '--wait',
             '--wait-timeout', '7200', 'deepseek'], env=env)
   PY
   ```

   Capture startup logs, per-rank weights/KV allocation and GPU utilization. A
   topology assertion, OOM, collective timeout or kernel failure is a failed
   candidate, not a reason to mark the example ready. Preserve the first failure.
3. **Native contracts.** On loopback port 8009, test default/high/low/max, explicit
   `chat_template_kwargs={"thinking":false,"enable_thinking":false}`, images,
   tool-call/tool-result turns, streaming completion and cancellation. Check that
   Requirement JSON grammar and thinking-budget termination work on V4.1.
4. **Co-residency and L2.** Run normal `run.sh up`; confirm GPUs 6/7 run separate
   Qwen replicas and DeepSeek still owns 0–5. Inspect `/routing`; confirm all five
   profiles and only two policy-answer stages. Test JSON/headless/tool requests
   and native images. For API efforts omitted/low/high/max, correlate the
   Requirement stage with the DeepSeek hook log and verify high for
   omitted/low/high and max for API max.
   Verify that other DeepSeek thinking roles inherit effort and Qwen stays medium.
5. **Protocol and quality diagnostics.** Run the commands below. Requirement
   diagnostics run one three-case pass by default and report protocol contracts
   separately from heuristic/model judgments. Inspect failed/exhausted audits and
   full final answers; an exit code of zero is not independent factual validation.
   If the judge does not select ensemble, the ensemble-specific check should fail;
   record route coverage rather than declaring unexercised roles tested.
6. **Capacity/recovery/performance.** Validate clean restart and key persistence,
   context/retrieval at 8K/32K/128K/256K plus the claimed DS-only 1M envelope,
   concurrency 1/8/16/32, cancellation resource release and sustained operation.
   Only then investigate speculation or tuning, one recorded variable at a time.

```sh
./examples/qwen3.8-deepseek-v4.1-8gpu/verify.sh requirements-quality --no-start
./examples/qwen3.8-deepseek-v4.1-8gpu/verify.sh serving-auto-max --no-start
./examples/qwen3.8-deepseek-v4.1-8gpu/verify.sh serving-auto-max-coding --no-start
```

The native probes do not change running services. Choose a new output directory
for each invocation; retain failures before retrying:

```sh
uv run python examples/qwen3.8-deepseek-v4.1-8gpu/gpu_smoke.py \
  --base-url http://127.0.0.1:8009/v1 --logprobs --output /tmp/v41-native-run
uv run python examples/qwen3.8-deepseek-v4.1-8gpu/capacity.py \
  --container qwen3-8-deepseek-v4-1-8gpu-deepseek-1 \
  --tokenizer /mnt/nvme/kairyu/model-volumes/deepseek-v4.1-flash-8gpu/models/deepseek-v4.1-flash/tokenizer.json \
  --results-dir /tmp/v41-capacity-run
```

`gpu_smoke.py --case '^requirements-'` selects the twelve combinations of
omitted/low/high/max API effort and omitted/low/max nested effort, plus explicit
quote/newline and backslash literal cases. Use `--case '^requirements-.*-nested-'`
for just the effort matrix. Match each
saved `messages_sha256` to the worker's `Kairyu role hook` log to establish
the effective high floor and max preservation; a valid JSON response alone
cannot establish it. A conflicting nested value must match the resolved
top-level effort after the hook (top-level max wins over nested low, while
top-level low or omitted uses high even if nested says max).
The script distinguishes completed answers from output-limit truncation and
client stream closure from server-side cancellation cleanup.

Use `--data-parallel-rank 0`, `1`, or `2` to check each native DP rank. The
`native-thinking-budget` case requires exactly 16 reported reasoning tokens
before a completed public answer; a shorter/missing count is `not_exercised`,
not proof of forced termination. `--suite l2 --base-url http://127.0.0.1:8008/v1`
checks observed public routes, images, tools, JSON and cancel/recovery traces.
Add `--l2-effort-matrix --case '^l2-route-primary'` for omitted/low/high/max
public effort. A route targeted by the fixture but not chosen by the judge is
reported as a coverage gap; there is no public force-route override.

`capacity.py` checks the running DeepSeek container's image, topology, startup
configuration hash, published loopback endpoint and tokenizer pin. It reports
fixed 256-token throughput (including reasoning) separately from exact-key
retrieval with a completed answer. Use `--cases serving retrieval-8k` or the
individual `retrieval-32k`, `retrieval-128k`, `retrieval-256k`,
`retrieval-near1m` cases to stage the work. The native probes complement the
public L2/requirements checks; they do not establish ensemble behavior.

Run `gpu_cancellation.py --suite native --output NEW_DIRECTORY` in an idle
window to verify all three native DP ranks: require visible output and a running
request, close the unfinished stream, observe running/waiting gauges return to
zero, then verify a recovery answer. For the composed API use `--suite public
--public-request REQUEST_JSON`, with three `--worker-metrics` arguments:
`deepseek=http://127.0.0.1:8009/metrics`,
`qwen-0=docker://qwen3-8-deepseek-v4-1-8gpu-qwen-0-1`, and the analogous qwen-1
container. This requires overlapping DeepSeek/Qwen activity before cancellation
and checks all three workers afterwards. It records worker cleanup separately
from primary-route coverage, since cancelling prevents the final full trace.

Use `--suite public-audit` with the same request/metrics arguments and both
`--audit-log qwen-0=docker://qwen3-8-deepseek-v4-1-8gpu-qwen-0-1` and the analogous
qwen-1 argument to cancel specifically during deferred audit. The probe requires
an idle baseline, exactly one new audit hook, matching Qwen-only activity, a
later pending-stream keepalive and another running snapshot immediately before
close. It saves the observed audit message hash, raw hook, metrics and recovery.
A short audit that completes before the required observation cannot pass. Run
one probe at a time with fresh directories and no unrelated inference; initialize
all native DP ranks first so the strict rank-inventory metrics are available.
Activation/recovery timeouts are bounded to 600 seconds. Native/early-public
cancellation success does not replace the deferred-audit check.

Results default to the new example's `verification-results` directory.
`VERIFICATION_RESULTS_ROOT` and `--run-id` can select a persistent run location.
Manifests hash the new configuration, hooks, fixtures and shared runtime sources.
Even with `--no-start`, verification checks all four running model/API containers'
startup configuration hashes and the pinned vLLM image IDs, then records actual
container image IDs. A stale deployment is rejected before measurements begin.
The coding comparison renders the native non-thinking DeepSeek direct role over
the same tasks, so V4.1's thinking-high default cannot silently change the baseline.
It requires a newly measured paired baseline; there is no old-model fallback.
Rows routed exclusively to thinking-direct are reported as TTFT-gate N/A.

For a failing candidate, retain logs and exact configuration first. Investigate
memory placement/Engram, EP collectives, then cache/kernel shape support according
to the first failure. A revised context limit, CPU offload, PP layout, DSpark or
L1 patch is a new configuration needing its own evidence; document the decision
before retrying. Stop only this example using `run.sh down` when finished, and
retain persistent models/caches/evidence. Update `MEASUREMENTS.md` and the PR's
remaining-task list with measured outcomes and exact hashes.
