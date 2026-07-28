# F1a kind fleet

This directory contains the resource-bounded topology for the G5 F1a churn
gate:

- one Kairyu gateway using Kubernetes EndpointSlice discovery;
- one headless Service backed by a 200-replica StatefulSet;
- a static, standard-library-only Go mock backend;
- namespaced EndpointSlice RBAC; and
- graceful replica removal through readiness drain plus a five-second
  propagation window.

The mock returns its Kubernetes pod name and UID from every endpoint and embeds
them in completion text. Its `drain` subcommand is also the image's pre-stop
client, so the final image can remain `scratch` without a shell or curl.

## Run the gates

```bash
bash scripts/kind_churn_gate.sh --formal
bash scripts/kind_churn_gate.sh --smoke
```

The kind v0.32.0 gate pins its Kubernetes v1.36.1 node image by digest and
configures the kubelet for 350 pods. The mock StatefulSet requests only
1 millicore and 4 MiB per replica; its 32 MiB memory limit protects the runner
from a faulty process without reserving 6.4 GiB in the scheduler.

The formal profile applies the base 200-replica manifest and is the
authoritative test of the five-second graceful withdrawal window. The pull
request smoke profile applies `overlays/smoke` directly with four replicas; it
never creates 200 and scales down. It retains the same readiness and five-second
pre-stop protocol as formal while using fewer replicas and shorter epochs.

Both profiles build the fixed `kairyu:dev` and `kairyu-f1a-mock:dev` tags, load
those exact tags into a fresh kind cluster, wait for an exact Ready replica
count, and require the gateway metrics to report configured and healthy counts
equal to the profile size with zero draining replicas before traffic begins.

## Inspect

```bash
kubectl -n kairyu-f1a port-forward service/f1a-gateway 18080:8000
curl -sf http://127.0.0.1:18080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"f1a","messages":[{"role":"user","content":"hello"}],"max_tokens":1}'
kubectl -n kairyu-f1a exec deployment/f1a-gateway -- \
  sh -c 'cat /evidence/placements.jsonl'
```

The formal churn driver should delete 20 distinct Ready replica pods per minute
with `--wait=false`. The StatefulSet replaces them with new pod UIDs. Pre-stop
first changes `/readyz` to 503, causing the EndpointSlice controller and gateway
reconciler to remove the old UID, while the old process remains able to finish
already-selected requests for five seconds.
