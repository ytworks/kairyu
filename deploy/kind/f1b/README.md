# F1b kind rolling rollout

The F1b gate reuses the production-shaped F1a gateway, EndpointSlice
discovery, headless Service, and static Go mock.  It changes only the
StatefulSet rollout policy and replica count:

- `overlays/formal`: 100 replicas, `RollingUpdate`, initial partition 100;
- `overlays/smoke`: 4 replicas, `RollingUpdate`, initial partition 4.

The benchmark starts retry-free traffic, runs `kubectl rollout restart`, and
then lowers the partition one ordinal at a time.  Before each partition change
it calls the old Pod's `/admin/drain` endpoint and waits for readiness and
gateway membership withdrawal.  It waits for the replacement UID and rollout
revision to become ready and eligible before continuing.  No operator step is
required.

The formal profile retains a five-second withdrawal bound and a 60-second
per-replacement bound. Its 1,500-second whole-rollout deadline is only a
stuck-run safety cap sized for 100 sequential replacements and shared-runner
jitter; it is not an additional rollout-latency acceptance target.

Run either profile from one clean source commit:

```bash
bash scripts/kind_rollout_gate.sh --formal
bash scripts/kind_rollout_gate.sh --smoke
```

Pull requests always run the smoke profile; the formal profile is reachable
only through `workflow_dispatch`, which defaults to formal.  The
`pull_request` trigger checks out the pull request head SHA, so the gate runs
on the proposed commit even before this workflow exists on the default branch.

Both overlays deliberately retain the F1a namespace and workload names.  Each
run uses a fresh, dedicated kind cluster, while retaining the exact gateway,
mock image, readiness, pre-stop, and placement-audit contracts already proven
by F1a.
