"""Install and drive real Codex CLI binaries for the matrix (M20 WP-05, D1).

- ``resolve_codex``: a version (``0.160.0``), an npm dist-tag (``latest``,
  ``alpha``) or a path to an existing ``codex`` executable; npm installs go to a
  cache directory, never to the repository.
- ``provider_toml``: the Codex ``config.toml`` for the two provider shapes the
  gate covers -- the documented custom provider (``wire_api="responses"``) and
  the Harbor/Terminal-Bench shape (built-in ``openai`` provider repointed with
  ``openai_base_url``). Retries keep Codex's defaults.
- ``run_turn``: one ``codex exec --json`` turn with an isolated
  ``HOME``/``CODEX_HOME`` and a dummy or deployment API key; the user's own
  Codex configuration and credentials are never read.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

NPM_PACKAGE = "@openai/codex"
CUSTOM = "custom-responses"
OPENAI_BASE_URL = "openai-base-url"
PROVIDER_ID = "kairyu"
CUSTOM_KEY_ENV = "KAIRYU_API_KEY"
OPENAI_KEY_ENV = "OPENAI_API_KEY"
INSTALL_TIMEOUT_S = 600
# The trace line Codex writes after each sampling response names the
# auto-compaction limit it derived from the model metadata (catalog).
TRACE_FILTER = "warn,codex_core::session::turn=trace"
_AUTO_COMPACT_LIMIT = re.compile(r"auto_compact_scope_limit=Some\((\d+)\)")
_VERSION = re.compile(r"(\d+\.\d+\.\d+(?:-[0-9A-Za-z.]+)?)")
_FALLBACK_METADATA = re.compile(r"Model metadata for `[^`]*` not found")


@dataclass(frozen=True)
class CodexBinary:
    version: str
    path: Path


def _run(command: Sequence[str], timeout: float) -> str:
    completed = subprocess.run(
        list(command), check=True, capture_output=True, text=True, timeout=timeout
    )
    return completed.stdout.strip()


def binary_version(path: Path) -> str:
    match = _VERSION.search(_run([str(path), "--version"], timeout=60))
    if match is None:
        raise RuntimeError(f"{path} --version printed no version")
    return match.group(1)


def resolve_codex(spec: str, cache_dir: Path) -> CodexBinary:
    """A runnable Codex for ``spec``, installing it from npm when needed."""

    candidate = Path(spec).expanduser()
    if candidate.is_file():
        return CodexBinary(binary_version(candidate), candidate.resolve())
    npm = shutil.which("npm")
    if npm is None:
        raise RuntimeError("npm is required to install Codex versions")
    version = spec
    if not _VERSION.fullmatch(spec):
        version = _run([npm, "view", f"{NPM_PACKAGE}@{spec}", "version"], timeout=120)
    prefix = cache_dir / version
    path = prefix / "node_modules" / ".bin" / "codex"
    if not path.exists():
        prefix.mkdir(parents=True, exist_ok=True)
        _run(
            [
                npm,
                "install",
                "--prefix",
                str(prefix),
                f"{NPM_PACKAGE}@{version}",
                "--no-audit",
                "--no-fund",
                "--loglevel=error",
            ],
            timeout=INSTALL_TIMEOUT_S,
        )
    installed = binary_version(path)
    if installed != version:
        raise RuntimeError(f"{path} reports {installed}, expected {version}")
    return CodexBinary(version, path)


@dataclass(frozen=True)
class Provider:
    shape: str
    base_url: str  # Kairyu (or proxy) OpenAI base URL ending in /v1
    model: str
    catalog: Path | None
    idle_timeout_ms: int | None = None


def _toml_string(value: str) -> str:
    return json.dumps(value)  # a JSON string is a valid TOML basic string


def provider_toml(provider: Provider) -> str:
    """The isolated ``config.toml``: the shapes docs/deployment.md documents."""

    lines = [f"model = {_toml_string(provider.model)}"]
    if provider.catalog is not None:
        lines.append(f"model_catalog_json = {_toml_string(str(provider.catalog))}")
    if provider.shape == OPENAI_BASE_URL:
        lines.append(f"openai_base_url = {_toml_string(provider.base_url)}")
        return "\n".join(lines) + "\n"
    if provider.shape != CUSTOM:
        raise ValueError(f"unknown provider shape {provider.shape!r}")
    lines += [
        f"model_provider = {_toml_string(PROVIDER_ID)}",
        "",
        f"[model_providers.{PROVIDER_ID}]",
        'name = "Kairyu"',
        f"base_url = {_toml_string(provider.base_url)}",
        f"env_key = {_toml_string(CUSTOM_KEY_ENV)}",
        'wire_api = "responses"',
    ]
    if provider.idle_timeout_ms is not None:
        lines.append(f"stream_idle_timeout_ms = {provider.idle_timeout_ms}")
    return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class CommandRun:
    command: str
    exit_code: int | None
    status: str | None


@dataclass(frozen=True)
class TurnResult:
    exit_code: int | None  # None: timed out
    completed: bool
    failure: str | None  # turn.failed / error message
    agent_messages: tuple[str, ...]
    commands: tuple[CommandRun, ...]
    fallback_metadata: bool  # Codex did not find the model in the catalog
    auto_compact_limits: tuple[int, ...]  # from the trace log, one per sampling

    @property
    def final_message(self) -> str:
        return self.agent_messages[-1] if self.agent_messages else ""


def parse_turn(stdout: str, stderr: str, exit_code: int | None) -> TurnResult:
    """Summarize ``codex exec --json`` events and the trace log of one turn."""

    failure = None
    completed = fallback = False
    messages: list[str] = []
    commands: list[CommandRun] = []
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = event.get("type") if isinstance(event, Mapping) else None
        item = (event.get("item") or {}) if isinstance(event, Mapping) else {}
        if kind == "turn.completed":
            completed = True
        elif kind == "turn.failed":
            failure = (event.get("error") or {}).get("message", "turn failed")
        elif kind == "error" and failure is None:
            failure = event.get("message")
        elif kind == "item.completed" and item.get("type") == "agent_message":
            messages.append(item.get("text") or "")
        elif kind == "item.completed" and item.get("type") == "command_execution":
            commands.append(
                CommandRun(item.get("command") or "", item.get("exit_code"), item.get("status"))
            )
        elif kind == "item.completed" and item.get("type") == "error":
            fallback = fallback or bool(_FALLBACK_METADATA.search(item.get("message") or ""))
    return TurnResult(
        exit_code=exit_code,
        completed=completed,
        failure=failure,
        agent_messages=tuple(messages),
        commands=tuple(commands),
        fallback_metadata=fallback,
        auto_compact_limits=tuple(int(m) for m in _AUTO_COMPACT_LIMIT.findall(stderr)),
    )


def isolated_env(run_dir: Path, api_key: str) -> dict[str, str]:
    """A minimal environment: isolated HOME/CODEX_HOME, node on PATH, one key."""

    node = shutil.which("node")
    system_path = os.defpath.split(os.pathsep) + ["/usr/local/bin", "/usr/sbin", "/sbin"]
    path = ([str(Path(node).parent)] if node else []) + system_path
    return {
        "PATH": os.pathsep.join(dict.fromkeys(path)),
        "HOME": str(run_dir / "home"),
        "CODEX_HOME": str(run_dir / "codex-home"),
        "LANG": "C.UTF-8",
        "RUST_LOG": TRACE_FILTER,
        CUSTOM_KEY_ENV: api_key,
        OPENAI_KEY_ENV: api_key,
    }


def prepare_home(run_dir: Path, provider: Provider) -> Path:
    """Create the isolated homes and the working directory; return the latter."""

    for name in ("home", "codex-home", "work"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    (run_dir / "codex-home" / "config.toml").write_text(provider_toml(provider), "utf-8")
    return run_dir / "work"


def run_turn(
    binary: CodexBinary,
    run_dir: Path,
    *,
    prompt: str,
    api_key: str,
    exec_args: Sequence[str] = (),
    config: Sequence[tuple[str, str]] = (),
    timeout_s: float,
) -> TurnResult:
    """Run one turn; stdout/stderr are kept as ``codex.jsonl``/``codex.log``."""

    command = [str(binary.path), "exec", "--json", "--skip-git-repo-check", *exec_args]
    for key, value in config:
        command += ["-c", f"{key}={value}"]
    command.append(prompt)
    stdout_path = run_dir / "codex.jsonl"
    stderr_path = run_dir / "codex.log"
    with stdout_path.open("w", encoding="utf-8") as out, stderr_path.open("w") as err:
        try:
            exit_code: int | None = subprocess.run(
                command,
                cwd=run_dir / "work",
                env=isolated_env(run_dir, api_key),
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=err,
                timeout=timeout_s,
                check=False,
            ).returncode
        except subprocess.TimeoutExpired:
            exit_code = None
    return parse_turn(
        stdout_path.read_text(encoding="utf-8"),
        stderr_path.read_text(encoding="utf-8", errors="replace"),
        exit_code,
    )
