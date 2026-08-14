"""Strict registry and checkout-boundary validation for verification gates."""

from __future__ import annotations

import ast
import json
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REGISTRY_NAME = "registry.toml"
REGISTRY_SCHEMA_VERSION = 1
VERIFICATION_ROOT = "verification"
LEGACY_RESULTS_ROOT = "bench/results"

_SCOPES = frozenset({"l1", "orchestration", "fleet", "product"})
_KINDS = frozenset({"correctness", "performance", "resilience", "diagnostic"})
_ENTRYPOINT_CLASSES = frozenset(
    {"external-runtime", "formal-gate", "microbench", "operator"}
)


@dataclass(frozen=True, slots=True)
class VerificationGate:
    """One checkout-only executable and its evidence classification."""

    id: str
    scope: str
    kind: str
    path: str
    module: str
    output_schema: str
    entrypoint_class: str
    formal: bool
    requires: tuple[str, ...]
    documentation: tuple[str, ...]
    installed: bool
    invocations: tuple[str, ...]

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> VerificationGate:
        expected = {
            "id", "scope", "kind", "path", "module", "output_schema",
            "entrypoint_class", "formal", "requires", "documentation",
            "installed", "invocations",
        }
        missing = sorted(expected - set(value))
        unknown = sorted(set(value) - expected)
        if missing or unknown:
            details = []
            if missing:
                details.append(f"missing {', '.join(missing)}")
            if unknown:
                details.append(f"unknown {', '.join(unknown)}")
            raise ValueError(f"invalid verification gate: {'; '.join(details)}")

        fields = {
            name: _required_string(value[name], name)
            for name in (
                "id", "scope", "kind", "path", "module", "output_schema",
                "entrypoint_class",
            )
        }
        formal = value["formal"]
        installed = value["installed"]
        if not isinstance(formal, bool):
            raise ValueError(f"{fields['id']}: formal must be a boolean")
        if not isinstance(installed, bool) or installed:
            raise ValueError(f"{fields['id']}: verification gates are not installed")
        if fields["scope"] not in _SCOPES:
            raise ValueError(f"{fields['id']}: unsupported scope {fields['scope']!r}")
        if fields["kind"] not in _KINDS:
            raise ValueError(f"{fields['id']}: unsupported kind {fields['kind']!r}")
        if fields["entrypoint_class"] not in _ENTRYPOINT_CLASSES:
            raise ValueError(
                f"{fields['id']}: unsupported entrypoint_class "
                f"{fields['entrypoint_class']!r}"
            )
        if formal != (fields["entrypoint_class"] == "formal-gate"):
            raise ValueError(f"{fields['id']}: formal must match entrypoint_class")

        requires = _string_tuple(value["requires"], "requires")
        documentation = _string_tuple(value["documentation"], "documentation")
        invocations = _string_tuple(value["invocations"], "invocations")
        if not requires or not documentation:
            raise ValueError(f"{fields['id']}: requirements and documentation are required")
        if invocations not in (("path",), ("path", "module")):
            raise ValueError(
                f"{fields['id']}: invocations must be path, optionally followed by module"
            )

        expected_path = fields["module"].replace(".", "/") + ".py"
        if fields["path"] != expected_path:
            raise ValueError(
                f"{fields['id']}: path must be derived from module ({expected_path!r})"
            )
        module_parts = fields["module"].split(".")
        if len(module_parts) != 4 or module_parts[0] != VERIFICATION_ROOT:
            raise ValueError(
                f"{fields['id']}: module must be verification.<scope>.<kind>.<name>"
            )
        _, path_scope, path_kind, stem = module_parts
        expected_id = f"{path_scope}.{path_kind}.{stem}"
        if (
            fields["scope"] != path_scope
            or fields["kind"] != path_kind
            or fields["id"] != expected_id
        ):
            raise ValueError(f"{fields['id']}: id/scope/kind must match module path")

        return cls(
            **fields,
            formal=formal,
            requires=requires,
            documentation=documentation,
            installed=installed,
            invocations=invocations,
        )


def load_registry(path: str | Path | None = None) -> tuple[VerificationGate, ...]:
    """Load and strictly validate registry v1."""

    source = Path(path) if path is not None else Path(__file__).with_name(REGISTRY_NAME)
    payload = tomllib.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema_version") != REGISTRY_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported verification registry schema: {payload.get('schema_version')!r}"
        )
    raw_entries = payload.get("entrypoints")
    if not isinstance(raw_entries, list):
        raise ValueError("verification registry requires [[entrypoints]]")
    entries = tuple(VerificationGate.from_mapping(value) for value in raw_entries)
    ids = [entry.id for entry in entries]
    paths = [entry.path for entry in entries]
    if len(ids) != len(set(ids)):
        raise ValueError("verification registry gate ids must be unique")
    if len(paths) != len(set(paths)):
        raise ValueError("verification registry paths must be unique")
    return entries


def load_compatibility_imports(
    path: str | Path | None = None,
) -> dict[str, tuple[str, ...]]:
    source = Path(path) if path is not None else Path(__file__).with_name(REGISTRY_NAME)
    payload = tomllib.loads(source.read_text(encoding="utf-8"))
    raw = payload.get("compatibility_imports")
    if not isinstance(raw, dict):
        raise ValueError("verification registry requires [compatibility_imports]")
    modules = {entry.module for entry in load_registry(source)}
    result: dict[str, tuple[str, ...]] = {}
    for importer, raw_targets in raw.items():
        if not isinstance(importer, str) or importer not in modules:
            raise ValueError(f"unknown compatibility import source: {importer!r}")
        targets = _string_tuple(raw_targets, f"compatibility_imports.{importer}")
        unknown = sorted(set(targets) - modules)
        if unknown:
            raise ValueError(
                f"{importer}: unknown compatibility targets: {', '.join(unknown)}"
            )
        if targets != tuple(sorted(targets)):
            raise ValueError(f"{importer}: compatibility targets must be sorted")
        result[importer] = targets
    return result


def registry_payload() -> dict[str, Any]:
    return {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "ownership": {
            "verification": VERIFICATION_ROOT,
            "legacy_results": LEGACY_RESULTS_ROOT,
            "installed": False,
        },
        "compatibility_imports": load_compatibility_imports(),
        "gates": [asdict(entry) for entry in load_registry()],
    }


def render_registry_json() -> str:
    return json.dumps(registry_payload(), indent=2, sort_keys=True) + "\n"


def validate_repository(root: str | Path) -> tuple[str, ...]:
    """Return all registry, inventory, documentation, and import violations."""

    root = Path(root)
    errors: list[str] = []
    entries = load_registry()
    by_path = {entry.path: entry for entry in entries}
    modules = frozenset(entry.module for entry in entries)

    actual_paths: set[str] = set()
    for path in sorted((root / VERIFICATION_ROOT).rglob("*.py")):
        if path.name == "__main__.py":
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, UnicodeDecodeError, SyntaxError) as error:
            errors.append(f"{path.relative_to(root).as_posix()}: cannot parse: {error}")
            continue
        if _has_main_guard(tree):
            actual_paths.add(path.relative_to(root).as_posix())

    for path in sorted(actual_paths - set(by_path)):
        errors.append(f"unregistered verification entrypoint: {path}")
    for path in sorted(set(by_path) - actual_paths):
        errors.append(f"registry entrypoint does not exist or lacks __main__: {path}")

    for package in ("kairyu", "evals", "evidence"):
        package_root = root / package
        if not package_root.is_dir():
            continue
        offenders = _namespace_imports(package_root, VERIFICATION_ROOT)
        if offenders:
            errors.append(
                f"{package} imports checkout-only verification modules: "
                + ", ".join(offenders)
            )
    eval_imports = _namespace_imports(root / VERIFICATION_ROOT, "evals")
    if eval_imports:
        errors.append("verification imports evals modules: " + ", ".join(eval_imports))

    actual_edges: dict[str, tuple[str, ...]] = {}
    for relative, entry in by_path.items():
        path = root / relative
        if not path.is_file():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if ast.get_docstring(tree) is None:
            errors.append(f"{relative}: module docstring is required")
        targets = tuple(sorted(_registered_imports(tree, modules)))
        if targets:
            actual_edges[entry.module] = targets
        for document in entry.documentation:
            document_path = root / document
            if not document_path.is_file():
                errors.append(f"{relative}: documentation does not exist: {document}")
                continue
            if relative not in document_path.read_text(encoding="utf-8"):
                errors.append(f"{relative}: path is not mentioned in {document}")

    allowed_edges = load_compatibility_imports()
    for source in sorted(set(actual_edges) | set(allowed_edges)):
        if actual_edges.get(source, ()) != allowed_edges.get(source, ()):
            errors.append(
                f"{source}: imports {list(actual_edges.get(source, ()))!r}, "
                f"registry allows {list(allowed_edges.get(source, ()))!r}"
            )
    return tuple(errors)


def _required_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"verification gate {field} must be a non-empty string")
    return value


def _string_tuple(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item for item in value
    ):
        raise ValueError(f"verification gate {field} must be a string array")
    if len(value) != len(set(value)):
        raise ValueError(f"verification gate {field} must not contain duplicates")
    return tuple(value)


def _has_main_guard(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if not isinstance(test, ast.Compare) or len(test.ops) != 1:
            continue
        if not isinstance(test.left, ast.Name) or test.left.id != "__name__":
            continue
        if not isinstance(test.ops[0], ast.Eq) or len(test.comparators) != 1:
            continue
        comparator = test.comparators[0]
        if isinstance(comparator, ast.Constant) and comparator.value == "__main__":
            return True
    return False


def _namespace_imports(root: Path, namespace: str) -> list[str]:
    results: list[str] = []
    for path in sorted(root.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, UnicodeDecodeError, SyntaxError):
            continue
        for name in sorted(_all_imports(tree)):
            if name == namespace or name.startswith(f"{namespace}."):
                results.append(f"{path.as_posix()} -> {name}")
    return results


def _all_imports(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module)
    return names


def _registered_imports(tree: ast.AST, modules: frozenset[str]) -> set[str]:
    results: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            results.update(alias.name for alias in node.names if alias.name in modules)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module in modules:
                results.add(node.module)
            for alias in node.names:
                candidate = f"{node.module}.{alias.name}"
                if candidate in modules:
                    results.add(candidate)
    return results


__all__ = [
    "LEGACY_RESULTS_ROOT", "REGISTRY_NAME", "REGISTRY_SCHEMA_VERSION",
    "VERIFICATION_ROOT", "VerificationGate", "load_compatibility_imports",
    "load_registry", "registry_payload", "render_registry_json",
    "validate_repository",
]
