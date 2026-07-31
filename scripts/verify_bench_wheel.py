#!/usr/bin/env python3
"""Build and verify the benchmark package boundary in an isolated directory."""

from __future__ import annotations

import argparse
import configparser
import json
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

FIXTURE_FILES = (
    "charxiv-reasoning.jsonl",
    "gpqa-diamond.jsonl",
    "hle.jsonl",
    "livecodebench-pro.jsonl",
    "livecodebench.jsonl",
    "long-context-reasoning.jsonl",
    "mrcr-v2.jsonl",
    "scicode.jsonl",
)
FIXTURE_PREFIX = "kairyu/bench/fixtures/"
ENTRYPOINT_MANIFEST = "kairyu/bench/entrypoints.toml"
CONSOLE_TARGET = "kairyu.entrypoints.cli:main"


class VerificationError(RuntimeError):
    """The built wheel does not preserve the declared package boundary."""


def _build_wheel(repo: Path, output_dir: Path, uv: str) -> Path:
    subprocess.run(
        [
            uv,
            "build",
            "--wheel",
            "--out-dir",
            str(output_dir),
            "--no-create-gitignore",
            str(repo),
        ],
        check=True,
    )
    wheels = sorted(output_dir.glob("*.whl"))
    if len(wheels) != 1:
        raise VerificationError(
            f"expected one wheel in {output_dir}, found {[path.name for path in wheels]}"
        )
    return wheels[0]


def _inspect_wheel(wheel: Path) -> tuple[str, ...]:
    with zipfile.ZipFile(wheel) as archive:
        names = tuple(sorted(archive.namelist()))
        for name in names:
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts:
                raise VerificationError(f"unsafe wheel member: {name}")

        expected_fixtures = {
            f"{FIXTURE_PREFIX}{fixture}" for fixture in FIXTURE_FILES
        }
        actual_fixtures = {
            name
            for name in names
            if name.startswith(FIXTURE_PREFIX) and name.endswith(".jsonl")
        }
        if actual_fixtures != expected_fixtures:
            raise VerificationError(
                "packaged fixtures differ: "
                f"expected={sorted(expected_fixtures)}, actual={sorted(actual_fixtures)}"
            )
        if ENTRYPOINT_MANIFEST not in names:
            raise VerificationError(f"wheel omits {ENTRYPOINT_MANIFEST}")

        forbidden = tuple(
            name
            for name in names
            if name.startswith(("bench/", "bench/results/", "tests/"))
        )
        if forbidden:
            raise VerificationError(
                f"wheel contains checkout-only files: {list(forbidden)}"
            )

        metadata_files = [
            name for name in names if name.endswith(".dist-info/entry_points.txt")
        ]
        if len(metadata_files) != 1:
            raise VerificationError(
                f"expected one entry_points.txt, found {metadata_files}"
            )
        parser = configparser.ConfigParser()
        parser.read_string(archive.read(metadata_files[0]).decode("utf-8"))
        try:
            target = parser["console_scripts"]["kairyu"]
        except KeyError as error:
            raise VerificationError("wheel omits the kairyu console script") from error
        if target != CONSOLE_TARGET:
            raise VerificationError(
                f"kairyu console target is {target!r}, expected {CONSOLE_TARGET!r}"
            )
    return names


def _isolated_run(
    extracted: Path,
    dependency_site: Path,
    code: str,
    *,
    cache_dir: Path,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    for name in ("PYTHONHOME", "PYTHONPATH", "PYTHONSAFEPATH"):
        env.pop(name, None)
    env["KAIRYU_BENCH_CACHE"] = str(cache_dir)
    bootstrap = f"import sys; sys.path.append({str(dependency_site)!r}); {code}"
    return subprocess.run(
        [sys.executable, "-S", "-c", bootstrap],
        cwd=extracted,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


def _verify_isolated_runtime(wheel: Path, scratch: Path) -> None:
    extracted = scratch / "site"
    extracted.mkdir()
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(extracted)

    dependency_site = Path(sysconfig.get_paths()["purelib"]).resolve()
    expected = json.dumps(list(FIXTURE_FILES))
    fixture_code = (
        "import json; "
        "from importlib import resources; "
        "from pathlib import Path; "
        "import kairyu; "
        "root=Path.cwd().resolve(); "
        "module=Path(kairyu.__file__).resolve(); "
        "assert module.is_relative_to(root), (module, root); "
        "fixtures=resources.files('kairyu.bench.fixtures'); "
        "names=sorted(item.name for item in fixtures.iterdir() "
        "if item.name.endswith('.jsonl')); "
        f"assert names == {expected}, names; "
        "[json.loads(line) for name in names "
        "for line in fixtures.joinpath(name).read_text(encoding='utf-8').splitlines() "
        "if line.strip()]; "
        "manifest=resources.files('kairyu.bench').joinpath('entrypoints.toml'); "
        "assert manifest.read_text(encoding='utf-8').strip(); "
        "print(module)"
    )
    fixture_result = _isolated_run(
        extracted,
        dependency_site,
        fixture_code,
        cache_dir=scratch / "cache",
    )
    module_path = Path(fixture_result.stdout.strip()).resolve()
    if not module_path.is_relative_to(extracted.resolve()):
        raise VerificationError(
            f"isolated fixture check imported outside the wheel: {module_path}"
        )

    entrypoint_code = (
        "from kairyu.entrypoints.cli import main; "
        "main(['bench', 'entrypoints', '--json'])"
    )
    entrypoint_result = _isolated_run(
        extracted,
        dependency_site,
        entrypoint_code,
        cache_dir=scratch / "cache",
    )
    try:
        registry = json.loads(entrypoint_result.stdout)
    except json.JSONDecodeError as error:
        raise VerificationError(
            f"`kairyu bench entrypoints --json` was not JSON: {entrypoint_result.stdout!r}"
        ) from error
    if len(registry.get("entrypoints", ())) != 51:
        raise VerificationError(
            "packaged benchmark entrypoint registry does not contain all 51 wrappers"
        )

    for command in (
        ("bench", "--help"),
        ("bench", "run", "--help"),
        ("bench", "download", "--help"),
        ("bench", "report", "--help"),
        ("bench", "list", "--help"),
        ("bench", "entrypoints", "--help"),
    ):
        help_code = (
            "from kairyu.entrypoints.cli import main; "
            f"main({list(command)!r})"
        )
        help_result = _isolated_run(
            extracted,
            dependency_site,
            help_code,
            cache_dir=scratch / "cache",
        )
        if "usage:" not in help_result.stdout.lower():
            raise VerificationError(
                f"`kairyu {' '.join(command)}` did not render CLI help: "
                f"{help_result.stdout!r}"
            )

    list_code = (
        "from kairyu.entrypoints.cli import main; main(['bench', 'list'])"
    )
    list_result = _isolated_run(
        extracted,
        dependency_site,
        list_code,
        cache_dir=scratch / "cache",
    )
    if "suite fugu (11 slots)" not in list_result.stdout:
        raise VerificationError(
            "`kairyu bench list` did not dispatch the packaged Fugu suite: "
            f"{list_result.stdout!r}"
        )


def verify(repo: Path, *, uv: str, wheel: Path | None = None) -> Path:
    repo = repo.resolve()
    with tempfile.TemporaryDirectory(prefix="kairyu-bench-wheel-") as temp:
        scratch = Path(temp)
        built_wheel = wheel.resolve() if wheel is not None else _build_wheel(
            repo, scratch / "dist", uv
        )
        _inspect_wheel(built_wheel)
        _verify_isolated_runtime(built_wheel, scratch)
        print(
            f"verified {built_wheel.name}: packaged CLI, entrypoint manifest, "
            f"{len(FIXTURE_FILES)} fixtures; checkout-only bench/results/tests excluded"
        )
        return built_wheel


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root (default: parent of scripts/)",
    )
    parser.add_argument(
        "--wheel",
        type=Path,
        default=None,
        help="verify an existing wheel instead of building one",
    )
    parser.add_argument(
        "--uv",
        default=shutil.which("uv") or "uv",
        help="uv executable used to build the wheel",
    )
    args = parser.parse_args(argv)
    verify(args.repo, uv=args.uv, wheel=args.wheel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
