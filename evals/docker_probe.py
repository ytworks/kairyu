"""Docker availability probe for the agentic benchmarks (probed once per run)."""

from __future__ import annotations

import shutil
import subprocess


def docker_available(timeout_s: float = 10.0) -> tuple[bool, str]:
    if shutil.which("docker") is None:
        return False, "docker unavailable (binary not found)"
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return False, f"docker unavailable ({error})"
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).decode(errors="replace").strip()
        return False, f"docker unavailable (daemon not reachable: {detail[:120]})"
    return True, "docker available"
