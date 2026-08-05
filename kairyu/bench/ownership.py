"""Benchmark ownership manifest and source-checkout validation.

The installed :mod:`kairyu.bench` package owns reusable benchmark contracts,
the public ``kairyu bench`` CLI, synthetic fixtures, and this manifest.
Top-level ``bench/*.py`` files are stable, repository-only executable
entrypoints.  They may compose gate-specific code, but production/package code
must never depend on that repository-only namespace.
"""

from __future__ import annotations

import ast
import json
import tomllib
from dataclasses import asdict, dataclass
from importlib import resources
from pathlib import Path
from typing import Any

MANIFEST_NAME = "entrypoints.toml"
MANIFEST_SCHEMA_VERSION = 1
REPOSITORY_ENTRYPOINT_ROOT = "bench"
REPOSITORY_RESULTS_ROOT = "bench/results"
INSTALLED_FIXTURE_PACKAGE = "kairyu.bench.fixtures"

_KINDS = frozenset(
    {
        "external-runtime",
        "formal-gate",
        "microbench",
        "operator",
    }
)


@dataclass(frozen=True, slots=True)
class BenchmarkEntrypoint:
    """One supported repository-only benchmark executable."""

    path: str
    module: str
    kind: str
    requires: tuple[str, ...]
    documentation: tuple[str, ...]
    installed: bool
    invocations: tuple[str, ...]

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> BenchmarkEntrypoint:
        expected = {
            "path",
            "module",
            "kind",
            "requires",
            "documentation",
            "installed",
            "invocations",
        }
        unknown = sorted(set(value) - expected)
        missing = sorted(expected - set(value))
        if missing or unknown:
            detail = []
            if missing:
                detail.append(f"missing {', '.join(missing)}")
            if unknown:
                detail.append(f"unknown {', '.join(unknown)}")
            raise ValueError(f"invalid benchmark entrypoint: {'; '.join(detail)}")

        path = _required_string(value["path"], "path")
        module = _required_string(value["module"], "module")
        kind = _required_string(value["kind"], "kind")
        requires = _string_tuple(value["requires"], "requires")
        documentation = _string_tuple(value["documentation"], "documentation")
        invocations = _string_tuple(value["invocations"], "invocations")
        installed = value["installed"]
        if not isinstance(installed, bool):
            raise ValueError(f"{path}: installed must be a boolean")
        if kind not in _KINDS:
            raise ValueError(f"{path}: unsupported kind {kind!r}")
        if not requires:
            raise ValueError(f"{path}: requires must not be empty")
        if not documentation:
            raise ValueError(f"{path}: documentation must not be empty")
        if invocations not in (("path",), ("path", "module")):
            raise ValueError(
                f"{path}: invocations must be path, optionally followed by module"
            )
        if installed:
            raise ValueError(f"{path}: top-level benchmark entrypoints are not installed")

        expected_module = path.removesuffix(".py").replace("/", ".")
        if module != expected_module:
            raise ValueError(
                f"{path}: module must be derived from path ({expected_module!r})"
            )
        if not path.startswith(f"{REPOSITORY_ENTRYPOINT_ROOT}/"):
            raise ValueError(f"{path}: entrypoint must live under bench/")
        return cls(
            path=path,
            module=module,
            kind=kind,
            requires=requires,
            documentation=documentation,
            installed=installed,
            invocations=invocations,
        )


def load_entrypoints() -> tuple[BenchmarkEntrypoint, ...]:
    """Load the package-owned entrypoint manifest from installed resources."""

    payload = _load_manifest_payload()
    if payload.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            "unsupported benchmark entrypoint manifest schema: "
            f"{payload.get('schema_version')!r}"
        )
    raw_entries = payload.get("entrypoints")
    if not isinstance(raw_entries, list):
        raise ValueError("benchmark entrypoint manifest requires [[entrypoints]]")
    entries = tuple(BenchmarkEntrypoint.from_mapping(value) for value in raw_entries)
    paths = [entry.path for entry in entries]
    if paths != sorted(paths):
        raise ValueError("benchmark entrypoint manifest paths must be sorted")
    duplicates = sorted(path for path in set(paths) if paths.count(path) > 1)
    if duplicates:
        raise ValueError(f"duplicate benchmark entrypoints: {', '.join(duplicates)}")
    return entries


def load_compatibility_imports() -> dict[str, tuple[str, ...]]:
    """Load the exact retained wrapper-to-wrapper composition allowlist."""

    payload = _load_manifest_payload()
    raw_imports = payload.get("compatibility_imports")
    if not isinstance(raw_imports, dict):
        raise ValueError(
            "benchmark entrypoint manifest requires [compatibility_imports]"
        )
    entries = load_entrypoints()
    modules = {entry.module for entry in entries}
    imports: dict[str, tuple[str, ...]] = {}
    for source, raw_targets in raw_imports.items():
        if not isinstance(source, str) or source not in modules:
            raise ValueError(f"unknown compatibility import source: {source!r}")
        targets = _string_tuple(raw_targets, f"compatibility_imports.{source}")
        if not targets:
            raise ValueError(f"{source}: compatibility imports must not be empty")
        if targets != tuple(sorted(targets)):
            raise ValueError(f"{source}: compatibility imports must be sorted")
        unknown = sorted(set(targets) - modules)
        if unknown:
            raise ValueError(
                f"{source}: unknown compatibility import targets: "
                + ", ".join(unknown)
            )
        imports[source] = targets
    if list(imports) != sorted(imports):
        raise ValueError("compatibility import sources must be sorted")
    return imports


def entrypoints_payload() -> dict[str, Any]:
    """Return a stable JSON-serializable ownership description."""

    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "ownership": {
            "reusable_code": "kairyu.bench",
            "installed_cli": "kairyu bench",
            "installed_fixtures": INSTALLED_FIXTURE_PACKAGE,
            "repository_entrypoints": REPOSITORY_ENTRYPOINT_ROOT,
            "repository_results": REPOSITORY_RESULTS_ROOT,
        },
        "compatibility_imports": load_compatibility_imports(),
        "entrypoints": [asdict(entry) for entry in load_entrypoints()],
    }


def render_entrypoints_json() -> str:
    """Render the installed ownership manifest without source-checkout access."""

    return json.dumps(entrypoints_payload(), indent=2, sort_keys=True) + "\n"


def validate_repository(root: str | Path) -> tuple[str, ...]:
    """Return every ownership violation in a Kairyu source checkout.

    Validation is deliberately read-only.  It proves that every supported
    top-level executable is classified and documented, that every declared
    invocation form remains possible, and that installed package code does not
    depend on repository-only ``bench`` modules.
    """

    root = Path(root)
    errors: list[str] = []
    bench_root = root / REPOSITORY_ENTRYPOINT_ROOT
    package_root = root / "kairyu"
    entries = load_entrypoints()
    by_path = {entry.path: entry for entry in entries}
    wrapper_stems = frozenset(Path(entry.path).stem for entry in entries)

    actual_paths = {
        path.relative_to(root).as_posix()
        for path in bench_root.glob("*.py")
        if path.name != "__init__.py"
    }
    declared_paths = set(by_path)
    for path in sorted(actual_paths - declared_paths):
        errors.append(f"unregistered benchmark entrypoint: {path}")
    for path in sorted(declared_paths - actual_paths):
        errors.append(f"manifest entrypoint does not exist: {path}")

    package_bench_imports: list[str] = []
    for source in sorted(package_root.rglob("*.py")):
        package_bench_imports.extend(
            _top_level_bench_imports(
                source,
                wrapper_stems=wrapper_stems,
            )
        )
    if package_bench_imports:
        errors.append(
            "installed kairyu package imports repository-only bench modules: "
            + ", ".join(package_bench_imports)
        )

    actual_compatibility_imports: dict[str, tuple[str, ...]] = {}
    for relative, entry in sorted(by_path.items()):
        path = root / relative
        if not path.is_file():
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, UnicodeDecodeError, SyntaxError) as error:
            errors.append(f"{relative}: cannot parse: {error}")
            continue
        if ast.get_docstring(tree) is None:
            errors.append(f"{relative}: module docstring is required")
        if not _has_main_guard(tree):
            errors.append(f"{relative}: __main__ guard is required")
        relative_imports = sorted(
            {
                f"{'.' * node.level}{node.module or ''}"
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom) and node.level
            }
        )
        if relative_imports:
            errors.append(
                f"{relative}: relative wrapper imports are unsupported: "
                + ", ".join(relative_imports)
            )
        if entry.module != relative.removesuffix(".py").replace("/", "."):
            errors.append(f"{relative}: module name does not match its path")
        imported_wrappers = tuple(
            _top_level_bench_import_names(path, wrapper_stems=wrapper_stems)
        )
        if imported_wrappers:
            actual_compatibility_imports[entry.module] = imported_wrappers
        for doc in entry.documentation:
            doc_path = root / doc
            if not doc_path.is_file():
                errors.append(f"{relative}: documentation does not exist: {doc}")
                continue
            try:
                documented = doc_path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as error:
                errors.append(f"{relative}: cannot read documentation {doc}: {error}")
                continue
            if relative not in documented:
                errors.append(f"{relative}: path is not mentioned in {doc}")

    declared_compatibility_imports = load_compatibility_imports()
    for source in sorted(
        set(actual_compatibility_imports) | set(declared_compatibility_imports)
    ):
        actual = actual_compatibility_imports.get(source, ())
        declared = declared_compatibility_imports.get(source, ())
        if actual != declared:
            errors.append(
                f"{source}: wrapper imports {list(actual)!r}, "
                f"manifest allows {list(declared)!r}"
            )

    return tuple(errors)


def _required_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"benchmark entrypoint {field} must be a non-empty string")
    return value


def _string_tuple(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise ValueError(f"benchmark entrypoint {field} must be a string array")
    if len(value) != len(set(value)):
        raise ValueError(f"benchmark entrypoint {field} must not contain duplicates")
    return tuple(value)


def _load_manifest_payload() -> dict[str, Any]:
    manifest = resources.files("kairyu.bench").joinpath(MANIFEST_NAME)
    return tomllib.loads(manifest.read_text(encoding="utf-8"))


def _top_level_bench_imports(
    path: Path,
    *,
    wrapper_stems: frozenset[str] = frozenset(),
) -> list[str]:
    return [
        f"{path.as_posix()} -> {name}"
        for name in _top_level_bench_import_names(
            path,
            wrapper_stems=wrapper_stems,
        )
    ]


def _top_level_bench_import_names(
    path: Path,
    *,
    wrapper_stems: frozenset[str] = frozenset(),
) -> list[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, UnicodeDecodeError, SyntaxError):
        return []
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "bench" or alias.name.startswith("bench."):
                    imports.append(alias.name)
                elif alias.name.split(".", 1)[0] in wrapper_stems:
                    imports.append(f"bench.{alias.name}")
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            if node.module == "bench":
                imports.extend(f"bench.{alias.name}" for alias in node.names)
            elif node.module.startswith("bench."):
                imports.append(node.module)
            elif node.module.split(".", 1)[0] in wrapper_stems:
                imports.append(f"bench.{node.module}")
    return sorted(set(imports))


def _has_main_guard(tree: ast.Module) -> bool:
    for node in tree.body:
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if not (
            isinstance(test, ast.Compare)
            and isinstance(test.left, ast.Name)
            and test.left.id == "__name__"
            and len(test.ops) == 1
            and isinstance(test.ops[0], ast.Eq)
            and len(test.comparators) == 1
            and isinstance(test.comparators[0], ast.Constant)
            and test.comparators[0].value == "__main__"
        ):
            continue
        return True
    return False
