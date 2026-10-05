# Winnow-12B Q8_0 (GGUF, llama.cpp) x 8 replicas on 8 x RTX PRO 6000 Blackwell

```text
Open WebUI (:3000) -> Kairyu L3 (:8001) -> ReplicaPool -> winnow-server x 8 (llama.cpp, one per GPU)
Playground (:3001) -^       \-> /v1/systemone -> least busy of the same 8 servers
```

Eight identical `winnow-server` replicas serve one public model, `winnow-12b`.
Each replica has the settings of [`winnow-12b-q8-1gpu`](../winnow-12b-q8-1gpu/README.md):

- the same pinned GGUF files, shared on disk;
- llama.cpp b11036 with Winnow's patches, plus llama.cpp's Gemma 4
  `tool_choice: "required"` fix `f072b10` (`winnow-patches/`, image
  `local/winnow-inference:77d1458-gemma4req-sm120`);
- 8 chat slots of 65,536 tokens;
- Gemma 4's recommended sampling as server defaults.

Kairyu L2 adds no orchestration. The pool places each request on a replica:

- a warm prompt prefix wins only while its replica is idle;
- otherwise the replica with the fewest outstanding requests wins, so
  concurrent traffic spreads one per replica first.

Kairyu admits 64 chat requests, matching 8 replicas x 8 slots. Each replica
is attached with `backend: openai`, `upstream: llamacpp` and its own
`/health` URL. A replica that returns a 5xx is ejected; an HTTP 400 from a bad
request does not count against it.

Typed decisions (`/v1/systemone`, model `winnow-12b-systemone`) go to the
least busy of the same eight servers. A read moves once to another replica if
its replica is unreachable or overloaded. They never affect the chat pool's
replica health.

## Start

```sh
./run.sh
```

The command:

1. validates all eight GPUs and pins each replica to its GPU's NUMA CPUs;
2. builds the pinned `winnow-server` image if absent;
3. downloads and verifies the GGUF files once;
4. waits for all eight replicas, Kairyu, Open WebUI and the playground.

The Chat UI (`:3000`) and the System One playground (`:3001`, no
authentication) listen on all interfaces. `run.sh` prints their public URLs.
Set `CHAT_UI_BIND_ADDRESS=127.0.0.1` to keep both host-local. The playground
is that of the 1-GPU example.

Replica `winnow-N` is published host-locally on port `8091+N`, used only by
`verify.sh attest` and `contract`.

## Verification

```sh
./verify.sh list
./verify.sh all
```

The gates are those of the 1-GPU example, run on every replica where they
read L1 directly (`attest`, `contract`). In addition:

- `tool-calling` sends a 16-request burst and requires the pool's placement
  log to show every replica serving a tool call;
- `serving` runs at concurrency 8/32/64.

Results go to `verification/results/examples/winnow-12b-q8-dp8-8gpu/<run>/`.

The limitations are those of the 1-GPU example: Batch/AsyncRequest fail
closed, a deferred SLO request is shed, and llama.cpp relaxes regex `pattern`.
