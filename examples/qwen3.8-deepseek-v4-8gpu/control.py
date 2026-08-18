#!/usr/bin/env python3
"""One-command lifecycle for the eight-GPU Qwen/DeepSeek tiered stack."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import socket
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = json.loads((HERE / "example.json").read_text(encoding="utf-8"))
ROOT = HERE.parents[1]


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


def _storage_paths() -> dict[str, Path]:
    root = _nvme_root()
    environment = root / "model-volumes" / SPEC["environment"]
    qwen_models = root / "model-volumes/qwen3.8-27b-1gpu/models"
    paths = {
        "qwen_models": qwen_models,
        "deepseek_models": root / "model-volumes/deepseek-v4-flash-0731-8gpu",
        "webui": environment / "webui-data",
        "deepseek_cache": environment / "compile-cache/deepseek",
    }
    paths.update(
        {
            f"qwen_cache_{index}": environment / f"compile-cache/qwen-{index}"
            for index in range(4)
        }
    )
    for path in paths.values():
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise SystemExit(f"cannot prepare NVMe storage {path}: {error}") from error
    free_gib = shutil.disk_usage(root).free // (1024**3)
    minimum = int(SPEC["storage"]["minimum_free_gib"])
    if free_gib < minimum:
        raise SystemExit(f"NVMe storage has {free_gib} GiB free; {minimum} GiB is required")
    return paths


def _compose_env() -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"COMPOSE_FILE", "COMPOSE_PROFILES", "COMPOSE_PROJECT_NAME"}
    }
    paths = _storage_paths()
    env.update(
        {
            "COMPOSE_DISABLE_ENV_FILE": "1",
            "COMPOSE_PROJECT_NAME": SPEC["environment"].replace(".", "-"),
            "QWEN_MODEL_STORAGE_PATH": str(paths["qwen_models"]),
            "DEEPSEEK_MODEL_VOLUME": os.environ.get(
                "DEEPSEEK_MODEL_VOLUME", SPEC["storage"]["deepseek_model_volume"]
            ),
            "WEBUI_STORAGE_PATH": str(paths["webui"]),
            "DEEPSEEK_CACHE_PATH": str(paths["deepseek_cache"]),
            "QWEN_VLLM_IMAGE": os.environ.get(
                "QWEN_VLLM_IMAGE", SPEC["vllm"]["qwen"]["image"]
            ),
            "DEEPSEEK_VLLM_IMAGE": os.environ.get(
                "DEEPSEEK_VLLM_IMAGE", SPEC["vllm"]["deepseek"]["image"]
            ),
            "OPEN_WEBUI_IMAGE": os.environ.get("OPEN_WEBUI_IMAGE", SPEC["webui"]["image"]),
            "API_BIND_ADDRESS": os.environ.get("API_BIND_ADDRESS", "0.0.0.0"),
            "API_PORT": os.environ.get("API_PORT", str(SPEC["api_port"])),
            "DEEPSEEK_L1_PORT": os.environ.get(
                "DEEPSEEK_L1_PORT", str(SPEC["deepseek_l1_loopback_port"])
            ),
            "CHAT_UI_PORT": os.environ.get("CHAT_UI_PORT", str(SPEC["webui"]["port"])),
            "CHAT_UI_BIND_ADDRESS": os.environ.get("CHAT_UI_BIND_ADDRESS", "0.0.0.0"),
            # Render-safe defaults for down/status/logs. `up` replaces these
            # with NUMA-local CPU sets discovered from the exact GPU inventory.
            "DEEPSEEK_CPUSET": os.environ.get("DEEPSEEK_CPUSET", "0"),
        }
    )
    for index in range(4):
        env[f"QWEN_CACHE_{index}_PATH"] = str(paths[f"qwen_cache_{index}"])
        env[f"QWEN_{index}_CPUSET"] = os.environ.get(f"QWEN_{index}_CPUSET", "0")
    return env


def _compose(arguments: list[str], *, check: bool = True) -> None:
    _run(
        [
            "docker",
            "compose",
            "--project-directory",
            str(HERE),
            "--file",
            str(HERE / "compose.yaml"),
            *arguments,
        ],
        env=_compose_env(),
        check=check,
    )


def _gpu_inventory(text: str) -> dict[int, dict[str, object]]:
    rows: dict[int, dict[str, object]] = {}
    for raw in text.splitlines():
        if not raw.strip():
            continue
        fields = [part.strip() for part in raw.split(",")]
        if len(fields) != 5:
            raise SystemExit(f"unexpected nvidia-smi row: {raw}")
        index = int(fields[0])
        rows[index] = {
            "name": fields[1],
            "memory_mib": int(fields[2]),
            "compute_capability": float(fields[3]),
            "pci_bus_id": fields[4].lower(),
        }
    return rows


def _numa_cpuset(pci_bus_id: str) -> str:
    canonical = pci_bus_id
    if canonical.startswith("00000000:"):
        canonical = canonical[4:]
    try:
        node = int((Path("/sys/bus/pci/devices") / canonical / "numa_node").read_text())
        if node < 0:
            raise ValueError("negative NUMA node")
        cpuset = Path(f"/sys/devices/system/node/node{node}/cpulist").read_text().strip()
    except (OSError, ValueError) as error:
        raise SystemExit(
            f"cannot determine NUMA affinity for GPU {pci_bus_id}: {error}"
        ) from error
    if not cpuset:
        raise SystemExit(f"NUMA CPU set for GPU {pci_bus_id} is empty")
    return cpuset


def _preflight(env: dict[str, str]) -> None:
    for executable in ("docker", "nvidia-smi"):
        if shutil.which(executable) is None:
            raise SystemExit(f"required executable is missing: {executable}")
    query = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,compute_cap,pci.bus_id",
            "--format=csv,noheader,nounits",
        ],
        capture=True,
    ).stdout
    rows = _gpu_inventory(query)
    expected = SPEC["hardware"]
    if sorted(rows) != list(range(expected["gpu_count"])):
        raise SystemExit("exactly eight contiguous GPU indices 0..7 are required")
    for index, row in rows.items():
        if row["name"] != expected["product"]:
            raise SystemExit(
                f"GPU {index} is {row['name']!r}; expected {expected['product']!r}"
            )
        if row["memory_mib"] < expected["minimum_vram_mib"]:
            raise SystemExit(f"GPU {index} has insufficient VRAM")
        if row["compute_capability"] < expected["minimum_compute_capability"]:
            raise SystemExit(f"GPU {index} has insufficient compute capability")
    cpusets = {index: _numa_cpuset(str(row["pci_bus_id"])) for index, row in rows.items()}
    for index in range(4):
        env[f"QWEN_{index}_CPUSET"] = cpusets[index]
    env["DEEPSEEK_CPUSET"] = ",".join(dict.fromkeys(cpusets[index] for index in range(4, 8)))
    print(
        f"hardware: 8 x {expected['product']} ({rows[0]['memory_mib']} MiB each); "
        "Qwen GPUs 0-3, DeepSeek GPUs 4-7",
        flush=True,
    )


def _image_exists(image: str) -> bool:
    return _run(["docker", "image", "inspect", image], capture=True, check=False).returncode == 0


def _ensure_vllm_image(env: dict[str, str], env_key: str, source: dict) -> None:
    image = env[env_key]
    if _image_exists(image):
        return
    if image != source["image"]:
        raise SystemExit(f"{env_key} does not exist locally: {image}")
    if source.get("distribution") == "upstream":
        print("vLLM image is absent; pulling the pinned upstream release", flush=True)
        _run(["docker", "pull", image])
        return
    print("vLLM image is absent; building the pinned SM120 source revision", flush=True)
    _run(
        [
            "docker",
            "build",
            "--pull",
            "--file",
            "docker/Dockerfile",
            "--tag",
            image,
            "--label",
            f"org.opencontainers.image.revision={source['source_revision']}",
            f"{source['source_repository']}#{source.get('source_ref', source['source_revision'])}",
        ]
    )


_MODEL_PROGRAM = r"""
import hashlib, json, os, sys
from pathlib import Path
from huggingface_hub import snapshot_download

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

if attestation.exists():
    current = json.loads(attestation.read_text())
    fields = (current.get('repo'), current.get('revision'), current.get('tree_sha256'))
    if fields != (repo, revision, expected_tree):
        raise SystemExit('model attestation does not match the pinned checkpoint')
    if os.environ.get('VERIFY_MODEL') != '1':
        raise SystemExit(0)
    rows, tree = inventory()
    if tree != expected_tree or current.get('files') != rows:
        raise SystemExit('model files differ from the pinned attestation')
    raise SystemExit(0)

snapshot_download(repo, revision=revision, local_dir=target,
                  token=os.environ.get('HF_TOKEN') or None)
rows, tree = inventory()
if tree != expected_tree:
    raise SystemExit(f'downloaded checkpoint tree mismatch: {tree}')
attestation.write_text(json.dumps({'schema_version': 1, 'repo': repo,
                                    'revision': revision, 'tree_sha256': tree,
                                    'files': rows}, sort_keys=True))
"""


def _download_model(image: str, mount: str, model: dict) -> None:
    command = [
        "docker",
        "run",
        "--rm",
        "--entrypoint",
        "python3",
        "--volume",
        f"{mount}:/models",
    ]
    if "HF_TOKEN" in os.environ:
        command.extend(["--env", "HF_TOKEN"])
    if os.environ.get("VERIFY_MODEL") == "1":
        command.extend(["--env", "VERIFY_MODEL=1"])
    command.extend(
        [
            image,
            "-c",
            _MODEL_PROGRAM,
            model["repo"],
            model["revision"],
            model["slug"],
            model["tree_sha256"],
        ]
    )
    _run(command)


def _ensure_deepseek_volume(env: dict[str, str]) -> str:
    name = env["DEEPSEEK_MODEL_VOLUME"]
    desired = str(_storage_paths()["deepseek_models"])
    inspected = _run(["docker", "volume", "inspect", name], capture=True, check=False)
    if inspected.returncode:
        _run(
            [
                "docker",
                "volume",
                "create",
                "--driver",
                "local",
                "--opt",
                "type=none",
                "--opt",
                "o=bind",
                "--opt",
                f"device={desired}",
                name,
            ],
            capture=True,
        )
        inspected = _run(["docker", "volume", "inspect", name], capture=True)
    payload = json.loads(inspected.stdout)
    options = payload[0].get("Options") if len(payload) == 1 else None
    expected = {"device": desired, "o": "bind", "type": "none"}
    if options != expected:
        raise SystemExit(
            f"existing volume {name} is not the expected NVMe bind: "
            f"expected {expected}, found {options}"
        )
    return name


def _ensure_models(env: dict[str, str]) -> None:
    _download_model(
        env["QWEN_VLLM_IMAGE"],
        env["QWEN_MODEL_STORAGE_PATH"],
        SPEC["models"]["tier1"],
    )
    volume = _ensure_deepseek_volume(env)
    _download_model(env["DEEPSEEK_VLLM_IMAGE"], volume, SPEC["models"]["tier2"])


def _json_url(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=5) as response:
        return json.loads(response.read())


def _post_json_url(url: str, payload: dict) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.loads(response.read())


def _validate_embedding_smoke(payload: dict) -> None:
    embedding = SPEC["embedding"]
    model = embedding["served_name"]
    if payload.get("model") != model:
        raise SystemExit(
            "Kairyu embedding response has the wrong model identity: "
            f"expected {model!r}, got {payload.get('model')!r}"
        )
    data = payload.get("data")
    if not isinstance(data, list) or [
        row.get("index") if isinstance(row, dict) else None for row in data
    ] != [0, 1]:
        raise SystemExit("Kairyu embedding response must preserve indices [0, 1]")
    dimensions = embedding["dimensions"]
    for row in data:
        vector = row.get("embedding")
        if not isinstance(vector, list) or len(vector) != dimensions:
            raise SystemExit(
                f"Kairyu embedding response vectors must have {dimensions} dimensions"
            )
        if not all(
            type(value) in {int, float} and math.isfinite(value) for value in vector
        ):
            raise SystemExit("Kairyu embedding response vectors must contain finite numbers")
    usage = payload.get("usage")
    prompt_tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
    total_tokens = usage.get("total_tokens") if isinstance(usage, dict) else None
    if (
        type(prompt_tokens) is not int
        or prompt_tokens < 1
        or type(total_tokens) is not int
        or total_tokens != prompt_tokens
    ):
        raise SystemExit("Kairyu embedding response must report positive exact usage")


def _validate_ready(api_url: str, tokenizer_url: str) -> None:
    try:
        ready = _json_url(f"{api_url}/readyz")
        models = {row["id"] for row in _json_url(f"{api_url}/v1/models")["data"]}
        routing = _json_url(f"{api_url}/routing")["models"]
        embedding_model = SPEC["embedding"]["served_name"]
        embedding_response = _post_json_url(
            f"{api_url}/v1/embeddings",
            {
                "model": embedding_model,
                "input": ["kairyu readiness probe", "two-input contract"],
                "encoding_format": "float",
            },
        )
        token_count = _post_json_url(
            tokenizer_url,
            {"model": "deepseek-v4-flash-0731", "prompt": "kairyu"},
        )["count"]
    except (KeyError, OSError, ValueError, urllib.error.URLError) as error:
        raise SystemExit(f"Kairyu readiness evidence is incomplete: {error}") from error
    product_model = SPEC["orchestration"]["auto_max_model"]
    expected_models = {product_model, embedding_model}
    if ready.get("status") != "ready" or models != expected_models:
        raise SystemExit(
            "Kairyu public model inventory must be exactly "
            f"{sorted(expected_models)!r}, got {sorted(models)!r}"
        )
    _validate_embedding_smoke(embedding_response)
    if type(token_count) is not int or token_count < 1:
        raise SystemExit("DeepSeek public-output tokenizer oracle is not ready")
    if set(routing) != {product_model}:
        raise SystemExit(f"Kairyu product routing inventory is not isolated: {sorted(routing)}")
    policy = routing[product_model]
    orchestration = SPEC["orchestration"]
    expected_roles = list(orchestration["roles"])
    if [role.get("name") for role in policy.get("roles", ())] != expected_roles:
        raise SystemExit(
            "Kairyu L2 does not report the required "
            f"{len(expected_roles)}-role dual-track product DAG"
        )
    # The dual-track DAG is the single profile: every turn runs it, so a
    # served general profile or profile judge means the wrong policy is live.
    if policy.get("general_roles"):
        raise SystemExit("Kairyu product policy must serve exactly one role profile")
    if policy.get("profile_judge"):
        raise SystemExit("Kairyu product policy must not configure a profile judge")
    if policy.get("stream_head") != orchestration["stream_head"]:
        raise SystemExit("Kairyu product policy must stream the head role publicly")
    if policy.get("moa_samples") != 0:
        raise SystemExit("Kairyu product policy must use the explicit DAG, not MoA")
    if policy.get("budget", {}).get("max_steps") != orchestration["max_steps"]:
        raise SystemExit(
            f"Kairyu product policy max_steps must be {orchestration['max_steps']}"
        )
    expected_refinements = orchestration["product_max_refinements"]
    if policy.get("budget", {}).get("max_refine_depth") != expected_refinements:
        raise SystemExit(
            f"Kairyu product policy max_refine_depth must be {expected_refinements}"
        )
    if policy.get("expose_intermediate_outputs") is not True:
        raise SystemExit("Kairyu product policy must expose separate intermediate output")
    configured = policy.get("configured_engines", {})
    if configured.get("tier1", {}).get("model") != "qwen3.8-27b":
        raise SystemExit("Kairyu Tier1 L2 worker is not bound to the Qwen L1 pool")
    if (
        configured.get("tier2", {}).get("model")
        != "deepseek-v4-flash-0731-thinking"
    ):
        raise SystemExit("Kairyu Tier2 L2 worker is not bound to the DeepSeek L1 pool")


def _public_ui_host() -> str:
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
        raise SystemExit(
            "cannot discover an externally reachable Chat UI host; set PUBLIC_HOST"
        ) from error
    if detected == "0.0.0.0" or detected.startswith("127."):
        raise SystemExit(
            "cannot discover an externally reachable Chat UI host; set PUBLIC_HOST"
        )
    return detected


def up() -> None:
    env = _compose_env()
    ui_host = _public_ui_host()
    env["WEBUI_URL"] = os.environ.get(
        "WEBUI_URL", f"http://{ui_host}:{env['CHAT_UI_PORT']}"
    )
    _preflight(env)
    _ensure_vllm_image(env, "QWEN_VLLM_IMAGE", SPEC["vllm"]["qwen"])
    _ensure_vllm_image(env, "DEEPSEEK_VLLM_IMAGE", SPEC["vllm"]["deepseek"])
    _ensure_models(env)
    _run(
        [
            "docker",
            "compose",
            "--project-directory",
            str(HERE),
            "--file",
            str(HERE / "compose.yaml"),
            "up",
            "--build",
            "--detach",
            "--wait",
            "--wait-timeout",
            "7200",
        ],
        env=env,
    )
    validation_api_host = (
        "127.0.0.1" if env["API_BIND_ADDRESS"] == "0.0.0.0" else env["API_BIND_ADDRESS"]
    )
    api_url = f"http://{validation_api_host}:{env['API_PORT']}"
    advertised_api_host = (
        ui_host if env["API_BIND_ADDRESS"] == "0.0.0.0" else env["API_BIND_ADDRESS"]
    )
    tokenizer_url = f"http://127.0.0.1:{env['DEEPSEEK_L1_PORT']}/tokenize"
    _validate_ready(api_url, tokenizer_url)
    print("\nEnvironment is ready.")
    print(f"OpenAI API: http://{advertised_api_host}:{env['API_PORT']}/v1")
    print(f"Chat UI:    http://{ui_host}:{env['CHAT_UI_PORT']} (no authentication)")
    print("Chat model:      kairyu-auto-max (the only Chat UI model)")
    print("Embedding model: embed-small")


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
