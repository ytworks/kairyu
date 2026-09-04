# Measured performance

Not yet measured. This section is filled in from the locked
`./verify.sh serving`, `./verify.sh tool-calling`, and `./verify.sh vision`
runs on the target host (8 x RTX PRO 6000 Blackwell), together with the
served-config SHA-256, the model tree SHA-256, and the vLLM image ID those
runs pinned. Until then the example carries `PENDING-*` placeholders for the
tree hash and image ID in `example.json` / `kairyu.yaml`, and `run.sh up`
refuses to serve until they are replaced with the values it computes.
