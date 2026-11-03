# F1c three-gateway kind topology

This directory is the production-shaped, CPU-only topology for G5 F1c:

- three independently restartable Kairyu gateway Deployments (`a`, `b`, `c`);
- one auditable L7 load balancer that partitions `X-Session-ID` by HRW;
- twelve F1a static Go mock replicas exposed by one headless Service; and
- one pinned PostgreSQL instance backing every gateway's batch store.

The load balancer uses the same frozen rendezvous input as `ReplicaPool`:
`sha256(f"{session_id}:{gateway_id}")`, with the lexicographically greatest
digest selected. It retries the remaining HRW candidates when the TCP connect
fails (an unreachable gateway) or on HTTP 5xx; once connected, the response
wait is only a hang bound, so a slow gateway is never mistaken for a dead one.
It returns `X-Kairyu-Gateway-ID` plus
`X-Kairyu-LB-Request-ID`. Requests without `X-Session-ID` use the load-balancer
request ID as their key and therefore have no cross-request affinity.

Every proxy decision is appended to `/evidence/decisions.jsonl`. A decision
contains the session hash, request-body hash, complete candidate order, each
attempt, selected gateway, and that gateway's `X-Request-ID`. The latter joins
the LB row to the selected gateway's `/evidence/placements.jsonl`, whose
replica UID can then be checked independently against the mock response.
Raw session values and request bodies are never written.

## Run and inspect

The gate builds `kairyu:dev` and `kairyu-f1a-mock:dev`, pulls PostgreSQL by
its linux/amd64 manifest digest, saves all three with `docker image save`, and
loads only those archives with `kind load image-archive`. Each image is
identified by the config digest inside its archive, which must equal kind's CRI
image ID (and, for PostgreSQL, the pinned config digest); Docker's own `.Id`
is store-dependent and is not used. The PostgreSQL Pod runs the
archive-loaded tag with `imagePullPolicy: Never`. The gate then applies this
kustomization to a fresh cluster created from `kind-config.yaml`. The NodePort
is mapped directly to `127.0.0.1:18082`.

```bash
bash scripts/kind_gateway_gate.sh

curl -sf http://127.0.0.1:18082/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'X-Session-ID: example-session' \
  -d '{"model":"f1a","messages":[{"role":"user","content":"hello"}],"max_tokens":1}'

kubectl -n kairyu-f1c exec deployment/f1c-lb -- \
  sh -c 'cat /evidence/decisions.jsonl'
kubectl -n kairyu-f1c exec deployment/f1c-gateway-a -- \
  sh -c 'cat /evidence/placements.jsonl'
```

The evidence mounts are kind-node `hostPath` directories under
`/var/lib/kairyu-f1c/evidence/{a,b,c,lb}`. They deliberately survive scaling an
owner gateway to zero and recreating it during the failover proof. A fresh kind
node makes them empty at the start of every gate; they are not a production
storage recommendation.

The shared batch configuration polls PostgreSQL every 100 ms, leases one job
for three seconds, and identifies a claimant by immutable gateway Pod UID.
The gate proves that another gateway reclaims an expired lease and publishes
one fenced terminal result after the owner Pod disappears. It judges this by
fencing tokens and claim-audit order only, never by whether the kill or the
terminal commit fell inside a particular lease window. It does not claim
PostgreSQL high availability: the database uses an `emptyDir`, and the checked
guarantee is gateway restart/failover while PostgreSQL remains available. The
password committed here is restricted to this disposable, non-exposed kind
fixture.
