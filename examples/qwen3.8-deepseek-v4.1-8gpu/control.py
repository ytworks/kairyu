#!/usr/bin/env python3
"""One-command lifecycle for the 6 + 2 GPU DeepSeek V4.1 / Qwen3.8 tiered stack."""

from __future__ import annotations

import argparse
import hashlib
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
PARENT_EXAMPLE = HERE.parent / SPEC["vllm"]["deepseek"]["parent_example"]
QWEN_GPU_IDS: list[int] = list(SPEC["allocation"]["tier1"]["gpu_ids"])
DEEPSEEK_GPU_IDS: list[int] = list(SPEC["allocation"]["tier2"]["gpu_ids"])


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
    """Persistent NVMe storage. The checkpoints are the sibling examples'
    attested downloads (read-only reuse); UI, placement log and compile
    caches are private to this environment."""

    root = _nvme_root()
    environment = root / "model-volumes" / SPEC["environment"]
    storage = SPEC["storage"]
    paths = {
        "qwen_models": (root / "model-volumes" / storage["qwen_model_environment"] / "models"),
        "deepseek_models": (
            root / "model-volumes" / storage["deepseek_model_environment"] / "models"
        ),
        "webui": environment / "webui-data",
        "placement_log": environment / "placement-log",
        "deepseek_cache": environment / "compile-cache/deepseek",
    }
    paths.update(
        {
            f"qwen_cache_{index}": environment / f"compile-cache/qwen-{index}"
            for index in range(len(QWEN_GPU_IDS))
        }
    )
    for path in paths.values():
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise SystemExit(f"cannot prepare NVMe storage {path}: {error}") from error
    free_gib = shutil.disk_usage(root).free // (1024**3)
    minimum = int(storage["minimum_free_gib"])
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
            "DEEPSEEK_MODEL_STORAGE_PATH": str(paths["deepseek_models"]),
            "WEBUI_STORAGE_PATH": str(paths["webui"]),
            "PLACEMENT_LOG_PATH": str(paths["placement_log"]),
            "DEEPSEEK_CACHE_PATH": str(paths["deepseek_cache"]),
            "QWEN_VLLM_IMAGE": os.environ.get("QWEN_VLLM_IMAGE", SPEC["vllm"]["qwen"]["image"]),
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
    for index in range(len(QWEN_GPU_IDS)):
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
        raise SystemExit(f"cannot determine NUMA affinity for GPU {pci_bus_id}: {error}") from error
    if not cpuset:
        raise SystemExit(f"NUMA CPU set for GPU {pci_bus_id} is empty")
    return cpuset


def _assign_cpusets(env: dict[str, str], cpusets: dict[int, str]) -> None:
    for index, gpu in enumerate(QWEN_GPU_IDS):
        env[f"QWEN_{index}_CPUSET"] = cpusets[gpu]
    env["DEEPSEEK_CPUSET"] = ",".join(dict.fromkeys(cpusets[gpu] for gpu in DEEPSEEK_GPU_IDS))


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
            raise SystemExit(f"GPU {index} is {row['name']!r}; expected {expected['product']!r}")
        if row["memory_mib"] < expected["minimum_vram_mib"]:
            raise SystemExit(f"GPU {index} has insufficient VRAM")
        if row["compute_capability"] < expected["minimum_compute_capability"]:
            raise SystemExit(f"GPU {index} has insufficient compute capability")
    _assign_cpusets(
        env, {index: _numa_cpuset(str(row["pci_bus_id"])) for index, row in rows.items()}
    )
    print(
        f"hardware: 8 x {expected['product']} ({rows[0]['memory_mib']} MiB each); "
        f"DeepSeek GPUs {DEEPSEEK_GPU_IDS[0]}-{DEEPSEEK_GPU_IDS[-1]}, "
        f"Qwen GPUs {', '.join(str(gpu) for gpu in QWEN_GPU_IDS)}",
        flush=True,
    )


def _image_id(image: str) -> str | None:
    inspected = _run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", image],
        capture=True,
        check=False,
    )
    return inspected.stdout.strip() if inspected.returncode == 0 else None


def _ensure_qwen_image(env: dict[str, str]) -> None:
    image = env["QWEN_VLLM_IMAGE"]
    source = SPEC["vllm"]["qwen"]
    if _image_id(image) is not None:
        return
    if image != source["image"]:
        raise SystemExit(f"QWEN_VLLM_IMAGE does not exist locally: {image}")
    print("Qwen vLLM image is absent; pulling the pinned upstream release", flush=True)
    _run(["docker", "pull", image])


def _image_file_sha256(image: str, paths: list[str]) -> dict[str, str]:
    """SHA-256 of files inside an image (no GPU, no model), keyed by basename."""

    output = _run(
        ["docker", "run", "--rm", "--entrypoint", "sha256sum", image, *paths],
        capture=True,
    ).stdout
    return {
        line.split()[1].rsplit("/", 1)[-1]: line.split()[0]
        for line in output.splitlines()
        if line.strip()
    }


def _attest_overlay(image: str, expected: dict[str, str], *, prefix: str = "/opt/kairyu/") -> None:
    """The image must carry exactly the patch scripts this checkout ships."""

    actual = _image_file_sha256(image, [prefix + name for name in expected])
    for name, digest in expected.items():
        if actual.get(name) != digest:
            raise SystemExit(
                f"image {image} carries {name} with SHA-256 {actual.get(name)}; this checkout "
                f"ships {digest} — rebuild the overlay from these files before serving"
            )


def _ensure_parent_image(env: dict[str, str]) -> str:
    """The sibling example's SM120 overlay, built exactly as that example builds
    it (its Dockerfile, its build arguments, its directory as context) when the
    tag is absent. The sibling directory is never written to."""

    source = SPEC["vllm"]["deepseek"]
    parent = json.loads((PARENT_EXAMPLE / "example.json").read_text(encoding="utf-8"))["vllm"]
    if parent["image"] != source["base_image"]:
        raise SystemExit(
            "the sibling example no longer builds the image this overlay is based on; "
            "re-validate before updating example.json"
        )
    image = parent["image"]
    if _image_id(image) is None:
        print("parent SM120 overlay is absent; building it from the sibling example", flush=True)
        _run(
            [
                "docker",
                "build",
                "--pull",
                "--file",
                str(PARENT_EXAMPLE / parent["dockerfile"]),
                "--build-arg",
                f"VLLM_BASE_IMAGE={parent['base_image']}",
                "--build-arg",
                f"FLASHINFER_REVISION={parent['flashinfer_revision']}",
                "--tag",
                image,
                "--label",
                f"org.opencontainers.image.revision={parent['source_revision']}",
                str(PARENT_EXAMPLE),
            ]
        )
    _attest_overlay(
        image,
        {
            "patch_runtime.py": hashlib.sha256(
                (PARENT_EXAMPLE / "patch_runtime.py").read_bytes()
            ).hexdigest()
        },
    )
    return image


def _ensure_deepseek_image(env: dict[str, str]) -> None:
    """Build this example's DeepSeek runtime on any host: the sibling's SM120
    overlay plus this directory's masked-KV / top-p overlay (vllm-sm120.Dockerfile).

    Attestation is by content — the patch scripts inside the image must match
    the files in this checkout — because a rebuild on another host yields a
    different image ID. The ID recorded in example.json is the one the
    measurements were taken on; a different local ID is reported, not refused.
    """

    image = env["DEEPSEEK_VLLM_IMAGE"]
    source = SPEC["vllm"]["deepseek"]
    expected = {
        name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in source["patches"]
    }
    if expected != source["patches"]:
        raise SystemExit(
            "example.json pins different patch-script hashes than the files in this "
            "directory; update the pins together with the scripts"
        )
    if _image_id(image) is None:
        if image != source["image"]:
            raise SystemExit(f"DEEPSEEK_VLLM_IMAGE does not exist locally: {image}")
        parent = _ensure_parent_image(env)
        print("DeepSeek overlay is absent; building it from this example's Dockerfile", flush=True)
        _run(
            [
                "docker",
                "build",
                "--file",
                str(HERE / source["dockerfile"]),
                "--build-arg",
                f"VLLM_BASE_IMAGE={parent}",
                "--tag",
                image,
                "--label",
                f"org.opencontainers.image.revision={source['source_revision']}",
                str(HERE),
            ]
        )
    _attest_overlay(image, expected)
    actual = _image_id(image)
    if actual != source["image_id"]:
        print(
            f"note: {image} has ID {actual}; the measurements in MEASUREMENTS.md were taken on "
            f"{source['image_id']} (same recipe, different build). Evidence produced here "
            "records this host's ID.",
            flush=True,
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


def _ensure_model(image: str, mount: str, model: dict) -> None:
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


def _ensure_models(env: dict[str, str]) -> None:
    _ensure_model(env["QWEN_VLLM_IMAGE"], env["QWEN_MODEL_STORAGE_PATH"], SPEC["models"]["tier1"])
    _ensure_model(
        env["DEEPSEEK_VLLM_IMAGE"], env["DEEPSEEK_MODEL_STORAGE_PATH"], SPEC["models"]["tier2"]
    )


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


_CHAT_UI_EFFORT_FILTER_ID = "reasoning_effort"
_CHAT_UI_EFFORT_LEVELS: list[str] = list(SPEC["webui"]["reasoning_effort_levels"])


def _webui_api(
    ui_url: str,
    path: str,
    *,
    token: str | None = None,
    payload: dict | None = None,
    method: str | None = None,
):
    headers = {"Content-Type": "application/json"}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(
        f"{ui_url}{path}",
        data=json.dumps(payload).encode() if payload is not None else None,
        headers=headers,
        method=method or ("POST" if payload is not None else "GET"),
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read())


def _provision_chat_ui_effort_selector(ui_url: str) -> None:
    """Install the Reasoning Effort dropdown into the pinned Open WebUI.

    The stock v0.11.0 Advanced Params control is a free-text field; the
    product knob must be selectable. A global filter whose enum-typed user
    valve renders as a dropdown in Chat Controls forwards the selection as
    the OpenAI-compatible ``reasoning_effort`` body field.
    """

    filter_source = (HERE / "webui-reasoning-effort-filter.py").read_text(encoding="utf-8")
    try:
        signin = _webui_api(ui_url, "/api/v1/auths/signin", payload={"email": "", "password": ""})
        if not isinstance(signin, dict) or not signin.get("token"):
            raise SystemExit("Chat UI signin did not return an auth-disabled session token")
        if signin.get("role") != "admin":
            raise SystemExit("Chat UI auth-disabled session must be the admin user")
        token = signin["token"]
        listed = _webui_api(ui_url, "/api/v1/functions/", token=token)
        existing = next((row for row in listed if row.get("id") == _CHAT_UI_EFFORT_FILTER_ID), None)
        body = {
            "id": _CHAT_UI_EFFORT_FILTER_ID,
            "name": "Reasoning Effort",
            "content": filter_source,
            "meta": {"description": "Select the Kairyu reasoning effort from a dropdown."},
        }
        if existing is None:
            state = _webui_api(ui_url, "/api/v1/functions/create", token=token, payload=body)
        else:
            state = _webui_api(
                ui_url,
                f"/api/v1/functions/id/{_CHAT_UI_EFFORT_FILTER_ID}/update",
                token=token,
                payload=body,
            )
            # /update keeps the stored activation flags; carry them over.
            state = {**existing, **(state or {})}
        # The /toggle endpoints FLIP state, so only call them while the flag
        # is off — calling unconditionally would deactivate on every re-up.
        if not state.get("is_active"):
            state = _webui_api(
                ui_url,
                f"/api/v1/functions/id/{_CHAT_UI_EFFORT_FILTER_ID}/toggle",
                token=token,
                payload={},
            )
        if not state.get("is_global"):
            state = _webui_api(
                ui_url,
                f"/api/v1/functions/id/{_CHAT_UI_EFFORT_FILTER_ID}/toggle/global",
                token=token,
                payload={},
            )
        if not (state.get("is_active") and state.get("is_global")):
            raise SystemExit("Chat UI effort selector could not be activated globally")
        spec = _webui_api(
            ui_url,
            f"/api/v1/functions/id/{_CHAT_UI_EFFORT_FILTER_ID}/valves/user/spec",
            token=token,
        )
        enum = spec.get("properties", {}).get("reasoning_effort", {}).get("enum")
    except (KeyError, OSError, TypeError, ValueError, urllib.error.URLError) as error:
        raise SystemExit(f"Chat UI effort selector provisioning failed: {error}") from error
    if enum != _CHAT_UI_EFFORT_LEVELS:
        raise SystemExit(
            f"Chat UI effort selector must expose exactly {_CHAT_UI_EFFORT_LEVELS!r}, got {enum!r}"
        )


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
            raise SystemExit(f"Kairyu embedding response vectors must have {dimensions} dimensions")
        if not all(type(value) in {int, float} and math.isfinite(value) for value in vector):
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


def _validate_policy(policy: dict, *, judged: bool) -> None:
    """The served L2 policy must be the DAG example.json describes."""

    orchestration = SPEC["orchestration"]
    expected_roles = list(orchestration["roles"])
    if [role.get("name") for role in policy.get("roles", ())] != expected_roles:
        raise SystemExit(
            f"Kairyu L2 does not report the required {len(expected_roles)}-role "
            "DeepSeek-led ensemble DAG"
        )
    served_profiles = {
        name: [role.get("name") for role in roles]
        for name, roles in (policy.get("profiles") or {}).items()
    }
    judge = policy.get("profile_judge") or {}
    if judged:
        expected_profiles = {name: list(roles) for name, roles in orchestration["profiles"].items()}
        if served_profiles != expected_profiles:
            raise SystemExit(
                "Kairyu L2 does not report the required direct-route profiles "
                f"{sorted(expected_profiles)!r}, got {sorted(served_profiles)!r}"
            )
        expected_sampling = orchestration["direct_route_sampling"]
        served_sampling = {
            name: {
                key: value
                for key, value in (roles[0].get("sampling") or {}).items()
                if value not in (None, [], {})
            }
            for name, roles in (policy.get("profiles") or {}).items()
        }
        if served_sampling != expected_sampling:
            raise SystemExit("Kairyu L2 does not report the required direct-route sampling policy")
        expected_judge = orchestration["profile_judge"]
        served_choices = [
            {"label": choice.get("label"), "profile": choice.get("profile")}
            for choice in judge.get("choices", ())
        ]
        if (
            judge.get("worker") != expected_judge["worker"]
            or judge.get("fallback") != expected_judge["fallback"]
            or served_choices != expected_judge["choices"]
        ):
            raise SystemExit(
                "Kairyu product policy must judge routes on the Qwen worker with the "
                f"{len(expected_judge['choices'])} configured choices"
            )
    elif served_profiles or judge:
        raise SystemExit("Kairyu ensemble policy must serve the primary DAG without a judge")
    if policy.get("stream_head") != orchestration["stream_head"]:
        raise SystemExit("Kairyu policy must stream the head role publicly")
    if policy.get("moa_samples") != 0:
        raise SystemExit("Kairyu policy must use the explicit DAG, not MoA")
    if policy.get("budget", {}).get("max_steps") != orchestration["max_steps"]:
        raise SystemExit(f"Kairyu policy max_steps must be {orchestration['max_steps']}")
    expected_refinements = orchestration["product_max_refinements"]
    if policy.get("budget", {}).get("max_refine_depth") != expected_refinements:
        raise SystemExit(f"Kairyu policy max_refine_depth must be {expected_refinements}")
    if policy.get("expose_intermediate_outputs") is not True:
        raise SystemExit("Kairyu policy must expose separate intermediate output")
    configured = policy.get("configured_engines", {})
    if configured.get("tier1", {}).get("model") != SPEC["allocation"]["tier1"]["model"]:
        raise SystemExit("Kairyu tier1 L2 worker is not bound to the Qwen L1 pool")
    if configured.get("tier2", {}).get("model") != SPEC["allocation"]["tier2"]["model"]:
        raise SystemExit("Kairyu tier2 L2 worker is not bound to the DeepSeek V4.1 L1 pool")


def _validate_ready(api_url: str, tokenizer_url: str) -> None:
    orchestration = SPEC["orchestration"]
    product_model = orchestration["auto_max_model"]
    ensemble_model = orchestration["ensemble_model"]
    embedding_model = SPEC["embedding"]["served_name"]
    try:
        ready = _json_url(f"{api_url}/readyz")
        models = {row["id"] for row in _json_url(f"{api_url}/v1/models")["data"]}
        routing = _json_url(f"{api_url}/routing")["models"]
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
            {"model": SPEC["models"]["tier2"]["served_name"], "prompt": "kairyu"},
        )["count"]
    except (KeyError, OSError, ValueError, urllib.error.URLError) as error:
        raise SystemExit(f"Kairyu readiness evidence is incomplete: {error}") from error
    expected_models = {product_model, ensemble_model, embedding_model}
    if ready.get("status") != "ready" or models != expected_models:
        raise SystemExit(
            "Kairyu public model inventory must be exactly "
            f"{sorted(expected_models)!r}, got {sorted(models)!r}"
        )
    _validate_embedding_smoke(embedding_response)
    if type(token_count) is not int or token_count < 1:
        raise SystemExit("DeepSeek public-output tokenizer oracle is not ready")
    if set(routing) != {product_model, ensemble_model}:
        raise SystemExit(
            f"Kairyu routing inventory is not the two orchestrators: {sorted(routing)}"
        )
    _validate_policy(routing[product_model], judged=True)
    _validate_policy(routing[ensemble_model], judged=False)


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
        raise SystemExit("cannot discover an externally reachable Chat UI host; set PUBLIC_HOST")
    return detected


def up() -> None:
    env = _compose_env()
    ui_host = _public_ui_host()
    env["WEBUI_URL"] = os.environ.get("WEBUI_URL", f"http://{ui_host}:{env['CHAT_UI_PORT']}")
    _preflight(env)
    _ensure_qwen_image(env)
    _ensure_deepseek_image(env)
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
    validation_ui_host = (
        "127.0.0.1" if env["CHAT_UI_BIND_ADDRESS"] == "0.0.0.0" else env["CHAT_UI_BIND_ADDRESS"]
    )
    _provision_chat_ui_effort_selector(f"http://{validation_ui_host}:{env['CHAT_UI_PORT']}")
    print("\nEnvironment is ready.")
    print(f"OpenAI API: http://{advertised_api_host}:{env['API_PORT']}/v1")
    print(f"Chat UI:    http://{ui_host}:{env['CHAT_UI_PORT']} (no authentication)")
    print("Chat model:      kairyu-auto-max (the only Chat UI model)")
    print("API-only model:  kairyu-ensemble-max (always the five-candidate ensemble)")
    print("Embedding model: embed-small")
    print("Reasoning effort: Chat Controls -> Valves -> Reasoning Effort (default/low/high/max)")


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
