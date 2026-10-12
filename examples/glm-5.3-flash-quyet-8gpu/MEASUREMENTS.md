# glm-5.3-flash-quyet-8gpu evidence

Status: **GPU verification pending.** The example was built on a machine
without the GPUs; no gate has run yet.

To run (8 × RTX PRO 6000 Blackwell, GPUs 0-7 idle):

```sh
./run.sh
./verify.sh l1
./verify.sh routing
./verify.sh think-route
./verify.sh effort
./verify.sh verified-tool-route
./verify.sh fallback
./verify.sh serving
./verify.sh serving-routed
./verify.sh browser
```

The pass criteria are those of `deepseek-v4.1-quyet-8gpu` (VCO-D22, with
its 2026-10-11 amendment); the gates are listed in `README.md`. Results land
in `/mnt/nvme/kairyu/model-volumes/glm-5.3-flash-quyet-8gpu/results/`.
