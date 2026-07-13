#!/usr/bin/env python3
"""Reject dangling relative Compose binds before container creation."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml

from kairyu.deploy.spec import load_deployment_spec

_DEPLOYMENT_CONFIG_TARGET = "/etc/kairyu/config.yaml"


@dataclass(frozen=True)
class RelativeBind:
    service: str
    source: str
    target: str | None


def _literal_relative_binds(compose_file: Path) -> tuple[RelativeBind, ...]:
    data = yaml.safe_load(compose_file.read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError("Compose YAML must contain a mapping")
    services = data.get("services")
    if not isinstance(services, Mapping):
        raise ValueError("Compose YAML must declare a services mapping")

    binds: list[RelativeBind] = []
    for service_name, service in services.items():
        if not isinstance(service_name, str) or not isinstance(service, Mapping):
            raise ValueError("Compose services must map names to service mappings")
        volumes = service.get("volumes")
        if volumes is None:
            continue
        if not isinstance(volumes, list):
            raise ValueError(f"service {service_name!r} volumes must be a list")
        for volume in volumes:
            if isinstance(volume, str):
                source, separator, remainder = volume.partition(":")
                if (
                    not separator
                    or not source.startswith(".")
                    or "$" in source
                ):
                    continue
                target = remainder.partition(":")[0] or None
            elif isinstance(volume, Mapping) and volume.get("type") == "bind":
                source = volume.get("source")
                target = volume.get("target")
                if (
                    not isinstance(source, str)
                    or "$" in source
                    or Path(source).is_absolute()
                ):
                    continue
                if not isinstance(target, str):
                    target = None
            else:
                continue
            binds.append(RelativeBind(service_name, source, target))
    return tuple(binds)


def _lexical_absolute(path: Path) -> Path:
    """Normalize dot segments without following symlinks."""
    return Path(os.path.abspath(path))


def _relative_git_path(path: Path, root: Path) -> str | None:
    try:
        return _lexical_absolute(path).relative_to(root).as_posix()
    except ValueError:
        return None


def _is_tracked(path: Path, root: Path, tracked: frozenset[str]) -> bool:
    relative = _relative_git_path(path, root)
    if relative is None:
        return False
    prefix = relative.rstrip("/") + "/"
    return relative in tracked or any(item.startswith(prefix) for item in tracked)


def _tracked_checkout(compose_file: Path) -> tuple[Path, frozenset[str]] | None:
    try:
        root_result = subprocess.run(
            ["git", "-C", str(compose_file.parent), "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return None
    if root_result.returncode != 0:
        return None
    root = Path(root_result.stdout.strip()).resolve()
    files_result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        capture_output=True,
        text=True,
        check=False,
    )
    if files_result.returncode != 0:
        return None
    tracked = frozenset(item for item in files_result.stdout.split("\0") if item)
    if not _is_tracked(compose_file, root, tracked):
        return None
    return root, tracked


def validate_compose_file(compose_file: str | Path) -> tuple[str, ...]:
    path = _lexical_absolute(Path(compose_file))
    try:
        binds = _literal_relative_binds(path)
    except (OSError, ValueError, yaml.YAMLError) as error:
        return (f"{path}: invalid Compose file: {error}",)

    tracked_checkout = _tracked_checkout(path)
    errors: list[str] = []
    for bind in binds:
        source_path = _lexical_absolute(path.parent / bind.source)
        prefix = f"{path}: service {bind.service!r}: source {bind.source!r}"
        if not source_path.exists():
            errors.append(f"{prefix}: does not exist ({source_path})")
            continue
        if tracked_checkout is not None:
            root, tracked = tracked_checkout
            if not _is_tracked(source_path, root, tracked):
                errors.append(f"{prefix}: not tracked; unavailable in a clean checkout")
        if bind.target == _DEPLOYMENT_CONFIG_TARGET:
            try:
                load_deployment_spec(source_path)
            except (OSError, ValueError, yaml.YAMLError) as error:
                errors.append(f"{prefix}: invalid DeploymentSpec: {error}")
    return tuple(errors)


def validate_compose_files(compose_files: Iterable[str | Path]) -> tuple[str, ...]:
    return tuple(
        error
        for compose_file in compose_files
        for error in validate_compose_file(compose_file)
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("compose_files", nargs="+", help="Compose YAML paths")
    args = parser.parse_args(argv)
    errors = validate_compose_files(args.compose_files)
    for error in errors:
        print(error, file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
