#!/usr/bin/env python3
"""One-command lifecycle for Quyet-1.0-Large on one GPU: vLLM chat + System One adapter."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import socket
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = json.loads((HERE / "example.json").read_text(encoding="utf-8"))
ROOT = HERE.parents[1]
PROJECT = "kairyu-" + SPEC["environment"].replace(".", "-")
VLLM_CONTAINER = f"{PROJECT}-vllm-1"
SYSTEMONE_CONTAINER = f"{PROJECT}-quyet-systemone-1"
SERVED = SPEC["model"]["served_name"]
SYSTEMONE = SPEC["systemone"]


def _check_spec() -> None:
    """Refuse an example.json whose parts disagree before anything runs."""

    adapter = SYSTEMONE["settings"]
    if (
        int(SPEC["hardware"]["gpu_count"]) != 1
        or int(SPEC["allocation"]["gpu_count"]) != 1
        or int(SPEC["allocation"]["replicas"]) != 1
        or SPEC["allocation"]["model"] != SERVED
        or int(SPEC["vllm"]["settings"]["VLLM_MAX_MODEL_LEN"])
        != int(SPEC["model"]["max_context_tokens"])
        # The adapter answers 529 once QUYET_MAX_INFLIGHT reads run and
        # QUYET_MAX_QUEUE wait; Kairyu never forwards more than its own
        # max_concurrency, so callers get Kairyu's 429 first.
        or int(SYSTEMONE["max_concurrency"])
        > int(adapter["QUYET_MAX_INFLIGHT"]) + int(adapter["QUYET_MAX_QUEUE"])
        # Chat and the adapter's reads together fit vLLM's running sequences.
        or int(SPEC["pool"]["max_concurrency"]) + int(adapter["QUYET_READ_WORKERS"])
        > int(SPEC["vllm"]["settings"]["VLLM_MAX_NUM_SEQS"])
    ):
        raise SystemExit(
            "example.json is inconsistent (one GPU and replica; context = max_model_len; "
            "Kairyu's System One forwarding within the adapter's in-flight + queue; chat "
            "plus adapter reads within vLLM's max_num_seqs)"
        )


_check_spec()


def _run(
    command: list[str],
    *,
    env: dict[str, str] | None = None,
    capture: bool = False,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    printable = " ".join(command[:4])
    print(f"+ {printable}{' ...' if len(command) > 4 else ''}", flush=True)
    return subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        check=check,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )


def _nvme_root() -> Path:
    configured = Path(os.environ.get("NVME_STORAGE_ROOT", SPEC["storage"]["root"]))
    if not configured.is_absolute():
        raise SystemExit("NVME_STORAGE_ROOT must be an absolute path below /mnt/nvme")
    root = configured.resolve()
    nvme = Path("/mnt/nvme")
    if root != nvme and nvme not in root.parents:
        raise SystemExit("NVME_STORAGE_ROOT must be /mnt/nvme or one of its descendants")
    return root


def environment_storage() -> Path:
    return _nvme_root() / "model-volumes" / SPEC["environment"]


def storage_paths() -> dict[str, Path]:
    environment = environment_storage()
    paths = {
        "models": environment / "models",
        "webui": environment / "webui-data",
        "placement_log": environment / "placement-log",
        "cache": environment / "vllm-cache",
    }
    for path in paths.values():
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise SystemExit(f"cannot prepare NVMe storage {path}: {error}") from error
    return paths


def systemone_image() -> str:
    return os.environ.get("QUYET_SYSTEMONE_IMAGE", SYSTEMONE["image"])


def compose_env() -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"COMPOSE_FILE", "COMPOSE_PROFILES", "COMPOSE_PROJECT_NAME"}
    }
    paths = storage_paths()
    env.update({key: str(value) for key, value in SPEC["vllm"]["settings"].items()})
    env.update({key: str(value) for key, value in SYSTEMONE["settings"].items()})
    env.update(
        {
            "COMPOSE_DISABLE_ENV_FILE": "1",
            "COMPOSE_PROJECT_NAME": PROJECT,
            "MODEL_STORAGE_PATH": str(paths["models"]),
            "VLLM_CACHE_PATH": str(paths["cache"]),
            "WEBUI_STORAGE_PATH": str(paths["webui"]),
            "PLACEMENT_LOG_PATH": str(paths["placement_log"]),
            "VLLM_IMAGE": SPEC["vllm"]["image"],
            "VLLM_HOST_PORT": str(SPEC["vllm"]["host_port"]),
            "QUYET_SYSTEMONE_IMAGE": systemone_image(),
            "SYSTEMONE_HOST_PORT": str(SYSTEMONE["host_port"]),
            "OPEN_WEBUI_IMAGE": os.environ.get("OPEN_WEBUI_IMAGE", SPEC["webui"]["image"]),
            "PLAYGROUND_IMAGE": os.environ.get("PLAYGROUND_IMAGE", SPEC["playground"]["image"]),
            "API_PORT": os.environ.get("API_PORT", str(SPEC["api_port"])),
            "CHAT_UI_PORT": os.environ.get("CHAT_UI_PORT", str(SPEC["webui"]["port"])),
            "PLAYGROUND_PORT": os.environ.get("PLAYGROUND_PORT", str(SPEC["playground"]["port"])),
            "CHAT_UI_BIND_ADDRESS": os.environ.get("CHAT_UI_BIND_ADDRESS", "0.0.0.0"),
            "GPU_ID": os.environ.get("GPU_ID", "0"),
            # Render-safe default for down/status/logs; `up` replaces it with
            # the selected GPU's NUMA-local CPUs.
            "GPU_CPUSET": os.environ.get("GPU_CPUSET", "0"),
        }
    )
    return env


def _compose(
    arguments: list[str], *, env: dict[str, str] | None = None, check: bool = True
) -> None:
    _run(
        ["docker", "compose", "--project-directory", str(HERE),
         "--file", str(HERE / "compose.yaml"), *arguments],
        env=env or compose_env(),
        check=check,
    )  # fmt: skip


def gpu_inventory(text: str) -> dict[int, dict[str, object]]:
    rows: dict[int, dict[str, object]] = {}
    for raw in text.splitlines():
        if not raw.strip():
            continue
        fields = [part.strip() for part in raw.split(",")]
        if len(fields) != 6:
            raise SystemExit(f"unexpected nvidia-smi row: {raw}")
        rows[int(fields[0])] = {
            "name": fields[1],
            "memory_mib": int(fields[2]),
            "memory_used_mib": int(fields[3]),
            "compute_capability": float(fields[4]),
            "pci_bus_id": fields[5].lower(),
        }
    return rows


def numa_node(pci_bus_id: str, sysfs: Path = Path("/sys")) -> int:
    canonical = pci_bus_id[4:] if pci_bus_id.startswith("00000000:") else pci_bus_id
    try:
        node = int((sysfs / "bus/pci/devices" / canonical / "numa_node").read_text())
    except (OSError, ValueError) as error:
        raise SystemExit(f"cannot determine NUMA node of GPU {pci_bus_id}: {error}") from error
    if node < 0:
        raise SystemExit(f"GPU {pci_bus_id} reports no NUMA node")
    return node


def _node_cpulist(node: int) -> str:
    try:
        cpus = Path(f"/sys/devices/system/node/node{node}/cpulist").read_text().strip()
    except OSError as error:
        raise SystemExit(f"cannot read CPUs of NUMA node {node}: {error}") from error
    if not cpus:
        raise SystemExit(f"NUMA node {node} has no CPUs")
    return cpus


def _vllm_holds(gpu_id: int) -> bool:
    """True when this example's running vLLM container is assigned exactly ``gpu_id``."""

    state = _run(
        ["docker", "inspect", "--format", "{{.State.Running}} {{json .HostConfig.DeviceRequests}}",
         VLLM_CONTAINER],
        capture=True,
        check=False,
    )  # fmt: skip
    if state.returncode != 0:
        return False
    running, _, requests = state.stdout.strip().partition(" ")
    devices = [
        device
        for request in json.loads(requests or "null") or ()
        for device in request.get("DeviceIDs") or ()
    ]
    return running == "true" and devices == [str(gpu_id)]


def selected_gpu(env: dict[str, str], *, allow_own: bool = True) -> dict[str, object]:
    """The selected GPU's row after the hardware checks; it must be idle."""

    for executable in ("docker", "nvidia-smi"):
        if shutil.which(executable) is None:
            raise SystemExit(f"required executable is missing: {executable}")
    try:
        gpu_id = int(env["GPU_ID"])
    except ValueError as error:
        raise SystemExit(f"GPU_ID must be one GPU index: {env['GPU_ID']!r}") from error
    query = _run(
        ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,compute_cap,pci.bus_id",
         "--format=csv,noheader,nounits"],
        capture=True,
    ).stdout  # fmt: skip
    rows = gpu_inventory(query)
    if gpu_id not in rows:
        raise SystemExit(f"GPU {gpu_id} is not present (GPUs: {sorted(rows)})")
    row, expected = rows[gpu_id], SPEC["hardware"]
    if row["name"] != expected["product"]:
        raise SystemExit(f"GPU {gpu_id} is {row['name']!r}; expected {expected['product']!r}")
    if int(row["memory_mib"]) < int(expected["minimum_vram_mib"]):
        raise SystemExit(f"GPU {gpu_id} has insufficient VRAM")
    if float(row["compute_capability"]) < float(expected["minimum_compute_capability"]):
        raise SystemExit(f"GPU {gpu_id} has insufficient compute capability")
    # Never evict another workload: the GPU must be idle unless this
    # example's own vLLM already holds that very GPU.
    held = allow_own and _vllm_holds(gpu_id)
    if not held and int(row["memory_used_mib"]) > int(expected["maximum_idle_used_mib"]):
        raise SystemExit(
            f"GPU {gpu_id} already has {row['memory_used_mib']} MiB in use; "
            "stop the workload holding it or choose another GPU_ID"
        )
    return row


def _preflight(env: dict[str, str]) -> None:
    row = selected_gpu(env)
    node = numa_node(str(row["pci_bus_id"]))
    env["GPU_CPUSET"] = _node_cpulist(node)
    print(
        f"hardware: GPU {env['GPU_ID']} = {SPEC['hardware']['product']} (NUMA {node})", flush=True
    )


def image_id(image: str) -> str | None:
    inspected = _run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", image], capture=True, check=False
    )
    return inspected.stdout.strip() if inspected.returncode == 0 else None


def ensure_images() -> None:
    """Pull the pinned vLLM image, build the System One adapter on it, attest both by ID."""

    vllm = SPEC["vllm"]
    if image_id(vllm["image"]) is None:
        _run(["docker", "pull", vllm["image"]])
    actual = image_id(vllm["image"])
    if actual != vllm["image_id"]:
        raise SystemExit(
            f"vLLM image {vllm['image']} has ID {actual}; example.json pins {vllm['image_id']}"
        )
    image = systemone_image()
    actual = image_id(image)
    if actual is None:
        if image != SYSTEMONE["image"]:
            raise SystemExit(f"QUYET_SYSTEMONE_IMAGE does not exist locally: {image}")
        print("System One adapter image is absent; building it", flush=True)
        _run(
            ["docker", "build", "--file", str(HERE / SYSTEMONE["dockerfile"]),
             "--build-arg", f"VLLM_IMAGE={vllm['image']}",
             "--build-arg", f"QUYET_VERSION={SYSTEMONE['quyet_version']}",
             "--tag", image, str(HERE)]
        )  # fmt: skip
        actual = image_id(image)
    if os.environ.get("QUYET_ALLOW_UNPINNED_IMAGE") == "1":
        print(f"WARNING: unpinned System One image {image} ({actual}); not evidence", flush=True)
        return
    if SYSTEMONE["image_id"] is None:
        raise SystemExit(
            f"System One image {image} ({actual}) is not pinned yet: record this ID as "
            "example.json systemone.image_id, or set QUYET_ALLOW_UNPINNED_IMAGE=1 for an "
            "unpinned (non-evidence) run"
        )
    if actual != SYSTEMONE["image_id"]:
        raise SystemExit(
            f"System One image {image} has ID {actual}; example.json pins {SYSTEMONE['image_id']}"
        )


# Runs inside the adapter image (huggingface_hub is vLLM's). Downloads the pinned
# revision when needed, then checks every file against the publisher's
# MANIFEST.sha256 and the whole tree against example.json.
_MODEL_PROGRAM = r"""
import hashlib, json, os, sys
from pathlib import Path

repo, revision, slug, expected_tree = sys.argv[1:]
target = Path('/models') / slug
attestation = target / '.kairyu-model-attestation.json'

def inventory():
    rows = []
    for path in sorted(target.rglob('*')):
        if not path.is_file() or path == attestation or '.cache' in path.parts:
            continue
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b''):
                digest.update(chunk)
        rows.append({'path': str(path.relative_to(target)), 'size': path.stat().st_size,
                     'sha256': digest.hexdigest()})
    tree = hashlib.sha256(json.dumps(rows, sort_keys=True,
                                     separators=(',', ':')).encode()).hexdigest()
    return rows, tree

def check_manifest(rows):
    listed = {}
    for line in (target / 'MANIFEST.sha256').read_text().splitlines():
        if line.strip():
            digest, name = line.split(None, 1)
            listed[name.strip()] = digest
    have = {row['path']: row['sha256'] for row in rows}
    wrong = sorted(name for name, digest in listed.items() if have.get(name) != digest)
    if wrong:
        raise SystemExit(f'files differ from the publisher MANIFEST.sha256: {wrong}')

if attestation.exists() and os.environ.get('VERIFY_MODEL') != '1':
    current = json.loads(attestation.read_text())
    pinned = (current.get('repo'), current.get('revision'), current.get('tree_sha256'))
    if pinned != (repo, revision, expected_tree):
        raise SystemExit('model attestation does not match the pinned checkpoint')
    raise SystemExit(0)

if not attestation.exists():
    from huggingface_hub import snapshot_download
    snapshot_download(repo, revision=revision, local_dir=target,
                      token=os.environ.get('HF_TOKEN') or None)
rows, tree = inventory()
check_manifest(rows)
if tree != expected_tree:
    raise SystemExit(f'checkpoint tree {tree} differs from model.tree_sha256 {expected_tree}')
attestation.write_text(json.dumps({'schema_version': 1, 'repo': repo, 'revision': revision,
                                   'tree_sha256': tree, 'files': rows}, sort_keys=True))
print(f'model attested: {len(rows)} files, tree {tree}')
"""


def ensure_model(env: dict[str, str]) -> None:
    model = SPEC["model"]
    target = Path(env["MODEL_STORAGE_PATH"]) / model["slug"]
    if not (target / ".kairyu-model-attestation.json").exists():
        free_gib = shutil.disk_usage(target.parent).free // (1024**3)
        minimum = int(SPEC["storage"]["minimum_free_gib"])
        if free_gib < minimum:
            raise SystemExit(f"NVMe storage has {free_gib} GiB free; {minimum} GiB is required")
    command = ["docker", "run", "--rm", "--entrypoint", "python3",
               "--env", "HF_HUB_DISABLE_TELEMETRY=1",
               "--volume", f"{env['MODEL_STORAGE_PATH']}:/models"]  # fmt: skip
    if "HF_TOKEN" in os.environ:
        command.extend(["--env", "HF_TOKEN"])
    if os.environ.get("VERIFY_MODEL") == "1":
        command.extend(["--env", "VERIFY_MODEL=1"])
    command.extend(
        [env["QUYET_SYSTEMONE_IMAGE"], "-c", _MODEL_PROGRAM, model["repo"], model["revision"],
         model["slug"], str(model["tree_sha256"])]
    )  # fmt: skip
    _run(command)


def json_url(url: str, timeout_s: float = 5) -> dict:
    with urllib.request.urlopen(url, timeout=timeout_s) as response:
        return json.loads(response.read())


def text_url(url: str) -> str:
    with urllib.request.urlopen(url, timeout=5) as response:
        return response.read().decode("utf-8")


def post_json(url: str, payload: dict, *, timeout_s: float) -> dict:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return json.loads(response.read())


def healthy_replicas(metrics: str, pool: str) -> int | None:
    pattern = re.compile(
        r'^kairyu_pool_replicas\{pool="' + re.escape(pool) + r'",state="healthy"\} (\S+)$',
        re.MULTILINE,
    )
    match = pattern.search(metrics)
    return int(float(match.group(1))) if match else None


def validate_ready(api_url: str) -> None:
    try:
        ready = json_url(f"{api_url}/readyz")
        listing = json_url(f"{api_url}/v1/models")
        healthy = healthy_replicas(text_url(f"{api_url}/metrics"), SERVED)
    except (KeyError, OSError, ValueError, urllib.error.URLError) as error:
        raise SystemExit(f"Kairyu readiness evidence is incomplete: {error}") from error
    chat_models = {row["id"] for row in listing.get("data", [])}
    jev_models = {row["name"] for row in listing.get("models", [])}
    systemone_names = {SYSTEMONE["model"], *SYSTEMONE["aliases"]}
    if (
        ready.get("status") != "ready"
        or chat_models != {SERVED}
        or not systemone_names <= jev_models
    ):
        raise SystemExit(
            f"Kairyu must serve exactly chat model {SERVED!r} and System One "
            f"{sorted(systemone_names)}; "
            f"got {sorted(chat_models)} / {sorted(jev_models)}"
        )
    if healthy != 1:
        raise SystemExit(f"Kairyu pool {SERVED!r} must report 1 healthy replica, got {healthy!r}")


def _bare(content: str) -> str:
    return content.strip().strip("*`").strip().rstrip(".")


def chat_answer_error(body: dict, *, expected: str | None = None) -> str | None:
    """Why a chat response is not a completed answer, or None."""

    try:
        choice = body["choices"][0]
        message = choice["message"]
    except (KeyError, IndexError, TypeError):
        return f"malformed response: {str(body)[:200]}"
    if choice.get("finish_reason") != "stop":
        return f"finish_reason is {choice.get('finish_reason')!r}"
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        return "empty answer content"
    if expected is not None and _bare(content) != expected:
        return f"answer is {content.strip()[:80]!r}, expected {expected!r}"
    return None


ARITHMETIC = "What is 17 * 19? Reply with only the integer."


def chat_request() -> dict:
    messages = [{"role": "user", "content": ARITHMETIC}]
    return {"model": SERVED, "max_tokens": 256, "messages": messages}


SYSTEMONE_STATE = "Everything is down and we have a demo with our biggest client at noon."
SYSTEMONE_QUESTIONS = {
    "urgent": {"type": "noul", "instructions": "Does the customer need a reply within the hour?"},
    "team": {
        "type": "choice",
        "instructions": "Which team should handle it?",
        "criteria": {
            "outage": "service down",
            "billing": "charges, refunds",
            "feature": "requests, how-to",
        },
    },
    "tone": {
        "type": "score",
        "instructions": "How upset is the customer?",
        "criteria": ["calm", "annoyed", "furious"],
    },
}


def systemone_request(state: str = SYSTEMONE_STATE, model: str | None = None) -> dict:
    return {"model": model or SYSTEMONE["model"], "state": state, "questions": SYSTEMONE_QUESTIONS}


def systemone_answer_error(body: dict) -> str | None:
    """Why a System One answer to SYSTEMONE_QUESTIONS is not a well-formed one."""

    try:
        answers, usage = body["answers"], body["usage"]
        if set(answers) != set(SYSTEMONE_QUESTIONS):
            return f"answers for {sorted(answers)}"
        team, tone = answers["team"], answers["tone"]
        if not 0 <= answers["urgent"]["noul"] <= 1:
            return f"noul out of range: {answers['urgent']}"
        for answer in (team, tone):
            if abs(sum(answer["probabilities"].values()) - 1) > 2e-3:
                return f"probabilities do not sum to 1: {answer}"
        if team["choice"] not in SYSTEMONE_QUESTIONS["team"]["criteria"]:
            return f"choice {team['choice']!r}"
        if not isinstance(usage.get("input_tokens"), int) or usage["input_tokens"] < 1:
            return f"usage {usage!r}"
        return None
    except (KeyError, TypeError, AttributeError) as error:
        return f"malformed answer ({error!r}): {str(body)[:200]}"


def validate_serving(api_url: str) -> None:
    """Readiness: pool state, a plain chat answer, and System One under every name."""

    validate_ready(api_url)
    body = post_json(f"{api_url}/v1/chat/completions", chat_request(), timeout_s=600)
    error = chat_answer_error(body, expected="323")
    if error:
        raise SystemExit(f"chat probe: {error}")
    for name in (SYSTEMONE["model"], *SYSTEMONE["aliases"]):
        body = post_json(f"{api_url}/v1/systemone", systemone_request(model=name), timeout_s=120)
        error = systemone_answer_error(body)
        if error:
            raise SystemExit(f"System One probe ({name}): {error}")


_NO_PUBLIC_HOST = "cannot discover an externally reachable UI host; set PUBLIC_HOST"


def public_ui_host() -> str:
    configured = os.environ.get("PUBLIC_HOST", "").strip()
    if configured:
        if "://" in configured or "/" in configured:
            raise SystemExit("PUBLIC_HOST must be a hostname or IPv4 address, not a URL")
        return configured
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as route:
            # UDP connect selects the outward-facing interface without sending data.
            route.connect(("192.0.2.1", 9))
            detected = str(route.getsockname()[0])
    except OSError as error:
        raise SystemExit(_NO_PUBLIC_HOST) from error
    if detected == "0.0.0.0" or detected.startswith("127."):
        raise SystemExit(_NO_PUBLIC_HOST)
    return detected


def up() -> None:
    env = compose_env()
    bind = env["CHAT_UI_BIND_ADDRESS"]
    ui_host = public_ui_host() if bind == "0.0.0.0" else bind
    env.setdefault("WEBUI_URL", f"http://{ui_host}:{env['CHAT_UI_PORT']}")
    _preflight(env)
    ensure_images()
    ensure_model(env)
    _compose(["up", "--build", "--detach", "--wait", "--wait-timeout", "3600"], env=env)
    api_url = f"http://127.0.0.1:{env['API_PORT']}"
    validate_serving(api_url)
    print("\nEnvironment is ready.")
    print(f"OpenAI API: {api_url}/v1  (model {SERVED})")
    print(
        f"System One: {api_url}/v1/systemone  (model {SYSTEMONE['model']}, "
        f"aliases {', '.join(SYSTEMONE['aliases'])})"
    )
    print(f"Chat UI:    http://{ui_host}:{env['CHAT_UI_PORT']} (no authentication)")
    print(f"Playground: http://{ui_host}:{env['PLAYGROUND_PORT']} (System One, no authentication)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=("up", "down", "status", "logs"), default="up")
    args = parser.parse_args()
    if args.action == "up":
        up()
    elif args.action == "down":
        _compose(["down"])
    elif args.action == "status":
        _compose(["ps"])
    else:
        _compose(["logs", "--follow", "--tail", "200"])


if __name__ == "__main__":
    main()
