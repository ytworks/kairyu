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
The gateway's requests and limits both freeze 2 CPU and 256 MiB. Its sole
container therefore receives Guaranteed QoS and a two-core CFS weight on the
shared four-vCPU runner. Formal evidence measured less than 300m gateway CPU
and 56 MiB while the node reached 3.459 cores during replacement, so unused
requested CPU remains available to kubelet/containerd while contention gives
the latency-bearing process the intended weight.

The formal profile applies the base 200-replica manifest and is the
authoritative test of the five-second graceful withdrawal window. The pull
request smoke profile applies `overlays/smoke` directly with four replicas; it
never creates 200 and scales down. It retains the same readiness and five-second
pre-stop protocol as formal while using fewer replicas and shorter epochs.

Both profiles build the fixed `kairyu:dev` and `kairyu-f1a-mock:dev` tags, load
those exact tags into a fresh kind cluster, wait for an exact Ready replica
count, and require the gateway metrics to report configured and healthy counts
equal to the profile size with zero draining replicas before traffic begins.
Image provenance is joined through the OCI config digest: the Docker source
config, the config referenced by containerd's loaded manifest, and the CRI
status ID must be identical. The raw manifest is retained and rehashed to
containerd's target descriptor. This works with both classic Docker stores,
whose `.Id` is the config digest, and containerd-backed Docker stores, whose
`.Id` is the manifest target, without weakening runtime image pinning.
The gateway Service uses NodePort 30080. The kind node maps that port directly
to `127.0.0.1:18080` on the runner, avoiding a `kubectl port-forward` process in
the traffic path. `F1A_GATEWAY_PORT` changes the localhost port by rendering the
pinned kind config into the evidence directory before cluster creation.

## Inspect

```bash
curl -sf http://127.0.0.1:18080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"f1a","messages":[{"role":"user","content":"hello"}],"max_tokens":1}'
kubectl -n kairyu-f1a exec deployment/f1a-gateway -- \
  sh -c 'cat /evidence/placements.jsonl'
```

The formal churn driver deletes 20 distinct Ready replica pods per minute
through its lifecycle-owned Kubernetes proxy/client. It issues at most eight
REST DELETE requests concurrently, selects background propagation, and leaves
`gracePeriodSeconds` unset. The StatefulSet replaces them with new pod UIDs.
Pre-stop first changes `/readyz` to 503, causing the EndpointSlice controller
and gateway reconciler to remove the old UID, while the old process remains
able to finish already-selected requests for five seconds.

Each deletion arms an endpoint-only observer before waiting for the Kubernetes
delete command. It records raw EndpointSlice payloads on absolute 250 ms
deadlines until every old UID is absent; slower multi-pod readiness polling
starts afterward and cannot dilute this causal sampling cadence. Artifact
replay verifies the observer sequence, scheduled/fetch/observation timestamps,
the exact disjoint claim, and the unchanged one-second last-old-to-disjoint
bracket.

Recovery readiness is EndpointSlice-first: the driver requires exactly 200
unique Ready, non-terminating Pod targets and new UIDs for all twenty scheduled
names. Only then does it issue one label-selected collection LIST for those
twenty Pods to corroborate UID, Ready state, and runtime image. Pod evidence is
the initial 200 identities, those ten targeted captures, and one final
exact-200 capture; it does not poll and serialize all 200 Pod objects once per
second.

Periodic EndpointSlice and resource evidence remains absolute-deadline paced.
The five-second resource sample uses kubelet's
`/stats/summary?only_cpu_and_memory=true` representation, retaining node and
Pod CPU, working-set memory, Pod UID, timestamps, and the existing nullable
network field without collecting filesystem or volume statistics. If a fetch
overruns its interval, the row records the skipped interval count and the
sampler resumes at the first future deadline instead of hammering the control
plane with catch-up requests. All raw evidence JSON is admitted in event-loop
order to a bounded lifecycle-owned writer. The post-traffic gateway audit copy
drains and retries bounded backpressure on a worker thread, and shutdown drains
every accepted row before sidecars are hashed and replayed. All Kubernetes
sampling reads and Pod DELETE requests share one lifecycle-owned
`kubectl proxy` and persistent HTTP client. The gateway likewise polls
EndpointSlices every 500 ms, retaining at least 1.13 seconds of worst-phase
margin against the historical 3.618-second withdrawal maximum and the
unchanged five-second limit. Its exact-compatible EndpointSlice parser reduced
the recorded 200-endpoint payload from 1.150 ms to 0.308 ms (3.74x) with an
identical member map.
