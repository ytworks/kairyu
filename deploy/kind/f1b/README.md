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

Run either profile from one clean source commit:

```bash
bash scripts/kind_rollout_gate.sh --formal
bash scripts/kind_rollout_gate.sh --smoke
```

Pull requests run the smoke profile by default.  Applying the
`f1b-formal` label runs the formal profile on the pull request commit, including
the initial pull request before this workflow exists on the default branch.
`workflow_dispatch` uses formal by default.

Both overlays deliberately retain the F1a namespace and workload names.  Each
run uses a fresh, dedicated kind cluster, while retaining the exact gateway,
mock image, readiness, pre-stop, and placement-audit contracts already proven
by F1a.
