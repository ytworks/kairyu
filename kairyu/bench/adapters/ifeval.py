"""IFEval: deterministic programmatic instruction-following evaluation."""

from __future__ import annotations

import hashlib
import io
import zipfile
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as distribution_version
from pathlib import Path

import httpx

from kairyu.bench.adapters.base import (
    AdapterInfo,
    DownloadContext,
    GenerativeAdapter,
    RunContext,
    cache_pins,
    excerpt,
)
from kairyu.bench.cache import BenchCache
from kairyu.bench.types import (
    BenchItem,
    BenchTarget,
    ChatRequestSpec,
    DatasetUnavailable,
    DownloadReport,
    ItemResult,
    PairResult,
    SkipItem,
)

_EXPECTED_ROWS = 541
_EXPECTED_INSTRUCTIONS = 834
_EXPECTED_INSTRUCTION_IDS = frozenset(
    {
        "change_case:capital_word_frequency",
        "change_case:english_capital",
        "change_case:english_lowercase",
        "combination:repeat_prompt",
        "combination:two_responses",
        "detectable_content:number_placeholders",
        "detectable_content:postscript",
        "detectable_format:constrained_response",
        "detectable_format:json_format",
        "detectable_format:multiple_sections",
        "detectable_format:number_bullet_lists",
        "detectable_format:number_highlighted_sections",
        "detectable_format:title",
        "keywords:existence",
        "keywords:forbidden_words",
        "keywords:frequency",
        "keywords:letter_frequency",
        "language:response_language",
        "length_constraints:nth_paragraph_first_word",
        "length_constraints:number_paragraphs",
        "length_constraints:number_sentences",
        "length_constraints:number_words",
        "punctuation:no_comma",
        "startend:end_checker",
        "startend:quotation",
    }
)
_REQUIRED_KWARGS = {
    "change_case:capital_word_frequency": {"capital_frequency", "capital_relation"},
    "change_case:english_capital": set(),
    "change_case:english_lowercase": set(),
    "combination:repeat_prompt": {"prompt_to_repeat"},
    "combination:two_responses": set(),
    "detectable_content:number_placeholders": {"num_placeholders"},
    "detectable_content:postscript": {"postscript_marker"},
    "detectable_format:constrained_response": set(),
    "detectable_format:json_format": set(),
    "detectable_format:multiple_sections": {"num_sections", "section_spliter"},
    "detectable_format:number_bullet_lists": {"num_bullets"},
    "detectable_format:number_highlighted_sections": {"num_highlights"},
    "detectable_format:title": set(),
    "keywords:existence": {"keywords"},
    "keywords:forbidden_words": {"forbidden_words"},
    "keywords:frequency": {"frequency", "keyword", "relation"},
    "keywords:letter_frequency": {"let_frequency", "let_relation", "letter"},
    "language:response_language": {"language"},
    "length_constraints:nth_paragraph_first_word": {
        "first_word",
        "nth_paragraph",
        "num_paragraphs",
    },
    "length_constraints:number_paragraphs": {"num_paragraphs"},
    "length_constraints:number_sentences": {"num_sentences", "relation"},
    "length_constraints:number_words": {"num_words", "relation"},
    "punctuation:no_comma": set(),
    "startend:end_checker": {"end_phrase"},
    "startend:quotation": set(),
}

_CHECKER_REVISION = "066e1eda43f4785922e3994e95429e496080231f"
_PUNKT_REVISION = "550b6625bcef1f2abff2ff770a5a0d272c9c6b2a"
_PUNKT_SHA256 = "e57f64187974277726a3417ca6f181ec5403676c717672eef6a748a7b20e0106"
_PUNKT_URL = (
    "https://raw.githubusercontent.com/nltk/nltk_data/"
    f"{_PUNKT_REVISION}/packages/tokenizers/punkt_tab.zip"
)
_PUNKT_ARCHIVE = f"punkt_tab-{_PUNKT_SHA256}.zip"
_PUNKT_FILES = {
    "collocations.tab": "8e2da1225e4dd2cc9dba261ee231ccb134859e21b46006e7f472c5ee269af0cf",
    "sent_starters.txt": "f3f8535483e1dba487241b764945168123bca3209a9645e59acd1225dc76edac",
    "abbrev_types.txt": "92a3e070f43d9b4c5534758ca40ad7343b04e7e29bfe0c2eb658a39445a4f779",
    "ortho_context.tab": "4bbcca25ed3d3f06c02402abf8419b9f033b8adc06e7b482eca4e45f81a5dc4c",
}
_CHECKER_DEPENDENCY_VERSIONS = {
    "langdetect": "1.0.9",
    "nltk": "3.9.2",
}
_VALID_RELATIONS = frozenset({"less than", "at least"})
_VALID_LANGUAGES = frozenset(
    {
        "ar", "bg", "bn", "de", "en", "es", "fa", "fi", "fr", "gu",
        "he", "hi", "it", "ja", "kn", "ko", "ml", "mr", "ne", "pa",
        "pl", "pt", "ru", "sw", "ta", "te", "th", "uk", "ur", "vi",
    }
)
_POSITIVE_INTEGER_ARGS = frozenset(
    {
        "capital_frequency",
        "frequency",
        "let_frequency",
        "nth_paragraph",
        "num_bullets",
        "num_highlights",
        "num_paragraphs",
        "num_placeholders",
        "num_sections",
        "num_sentences",
        "num_words",
    }
)
_NONEMPTY_STRING_ARGS = frozenset(
    {
        "capital_relation",
        "end_phrase",
        "first_word",
        "keyword",
        "language",
        "let_relation",
        "letter",
        "postscript_marker",
        "prompt_to_repeat",
        "relation",
        "section_spliter",
    }
)
_NONEMPTY_STRING_LIST_ARGS = frozenset({"forbidden_words", "keywords"})


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _punkt_root(cache: BenchCache) -> Path:
    return cache.assets_dir("ifeval") / "nltk_data"


def _punkt_archive(cache: BenchCache) -> Path:
    return cache.assets_dir("ifeval") / _PUNKT_ARCHIVE


def _punkt_file(cache: BenchCache, filename: str) -> Path:
    return _punkt_root(cache) / "tokenizers" / "punkt_tab" / "english" / filename


def _punkt_ready(cache: BenchCache) -> bool:
    try:
        archive = _punkt_archive(cache)
        if not archive.is_file() or _sha256(archive.read_bytes()) != _PUNKT_SHA256:
            return False
        return all(
            path.is_file() and _sha256(path.read_bytes()) == expected
            for filename, expected in _PUNKT_FILES.items()
            for path in (_punkt_file(cache, filename),)
        )
    except OSError:
        return False


def _download_punkt(cache: BenchCache) -> None:
    """Fetch one pinned archive and materialize only its four English data files."""

    if _punkt_ready(cache):
        return
    try:
        response = httpx.get(_PUNKT_URL, follow_redirects=True, timeout=60.0)
        response.raise_for_status()
    except httpx.HTTPError as error:
        raise DatasetUnavailable(f"pinned NLTK punkt_tab could not be fetched: {error}") from error
    archive_bytes = response.content
    actual_digest = _sha256(archive_bytes)
    if actual_digest != _PUNKT_SHA256:
        raise DatasetUnavailable(
            f"pinned NLTK punkt_tab SHA-256 is {actual_digest}, expected {_PUNKT_SHA256}"
        )

    try:
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            extracted: dict[str, bytes] = {}
            for filename, expected_digest in _PUNKT_FILES.items():
                member = f"punkt_tab/english/{filename}"
                if archive.namelist().count(member) != 1:
                    raise DatasetUnavailable(
                        f"pinned NLTK punkt_tab must contain exactly one {member!r}"
                    )
                data = archive.read(member)
                if _sha256(data) != expected_digest:
                    raise DatasetUnavailable(
                        f"pinned NLTK punkt_tab member {member!r} failed SHA-256 validation"
                    )
                extracted[filename] = data
    except zipfile.BadZipFile as error:
        raise DatasetUnavailable("pinned NLTK punkt_tab is not a valid ZIP archive") from error

    assets = cache.assets_dir("ifeval")
    assets.mkdir(parents=True, exist_ok=True)
    _punkt_archive(cache).write_bytes(archive_bytes)
    for filename, data in extracted.items():
        path = _punkt_file(cache, filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def _configure_punkt(cache: BenchCache) -> None:
    if not _punkt_ready(cache):
        raise RuntimeError("pinned IFEval punkt_tab resource is unavailable or corrupt")
    import nltk

    path = str(_punkt_root(cache))
    # Always put the verified tree first. Both NLTK and the vendored evaluator
    # memoize tokenizers, so clear objects that may have been created earlier
    # in this process from an ambient/global nltk_data directory.
    nltk.data.path[:] = [entry for entry in nltk.data.path if entry != path]
    nltk.data.path.insert(0, path)
    nltk.data.clear_cache()
    nltk.tokenize._get_punkt_tokenizer.cache_clear()

    from kairyu.bench._vendor.ifeval import instructions_util

    instructions_util._get_sentence_tokenizer.cache_clear()
    # Parse both tokenizer paths now so malformed-but-hash-matching resources
    # cannot turn into a mid-run scoring failure.
    instructions_util._get_sentence_tokenizer()
    nltk.tokenize._get_punkt_tokenizer("english")


def _installed_checker_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in _CHECKER_DEPENDENCY_VERSIONS:
        try:
            versions[package] = distribution_version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions


def _checker_dependency_error() -> str | None:
    try:
        import immutabledict  # noqa: F401
        import langdetect  # noqa: F401
        import nltk  # noqa: F401
    except ImportError as error:
        return f"IFEval checker dependencies unavailable ({error}); install kairyu[bench]"
    installed = _installed_checker_versions()
    for package, expected in _CHECKER_DEPENDENCY_VERSIONS.items():
        actual = installed[package]
        if actual != expected:
            return f"IFEval checker requires {package}=={expected}, found {actual or 'missing'}"
    return None


def _valid_instruction_args(instruction_id: str, entry: dict[str, object]) -> bool:
    for name, value in entry.items():
        if name in _POSITIVE_INTEGER_ARGS:
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                return False
        elif name in _NONEMPTY_STRING_ARGS:
            if not isinstance(value, str) or not value.strip():
                return False
        elif name in _NONEMPTY_STRING_LIST_ARGS:
            if (
                not isinstance(value, list)
                or not value
                or not all(
                    isinstance(item, str) and bool(item.strip()) for item in value
                )
            ):
                return False
        else:
            return False
    for relation_key in ("relation", "let_relation", "capital_relation"):
        if relation_key in entry and entry[relation_key] not in _VALID_RELATIONS:
            return False
    if instruction_id == "language:response_language":
        return entry["language"] in _VALID_LANGUAGES
    if instruction_id == "keywords:letter_frequency":
        letter = entry["letter"]
        return (
            isinstance(letter, str)
            and len(letter.strip()) == 1
            and len(letter.strip().lower()) == 1
        )
    if instruction_id == "length_constraints:nth_paragraph_first_word":
        return entry["nth_paragraph"] <= entry["num_paragraphs"]
    return True


class IfevalAdapter(GenerativeAdapter):
    info = AdapterInfo(
        name="ifeval",
        display_name="IFEval",
        metric="prompt-level strict accuracy",
        binary_outcomes=True,
        hf_dataset="google/IFEval",
        extra_sources=(
            (
                "google-research/google-research:instruction_following_eval",
                _CHECKER_REVISION,
            ),
            (
                "nltk/nltk_data:packages/tokenizers/punkt_tab.zip",
                _PUNKT_REVISION,
            ),
        ),
        annotations=(
            "headline score is prompt-level strict accuracy; four strict/loose "
            "metrics use the pinned official checker with a deterministic two-row "
            "dataset-consistency amendment",
        ),
        evaluation_resources=(
            ("kairyu.bench.adapters", "ifeval.py"),
            ("kairyu.bench._vendor.ifeval", "__init__.py"),
            ("kairyu.bench._vendor.ifeval", "checker.py"),
            ("kairyu.bench._vendor.ifeval", "instructions.py"),
            ("kairyu.bench._vendor.ifeval", "instructions_registry.py"),
            ("kairyu.bench._vendor.ifeval", "instructions_util.py"),
        ),
        evaluation_distributions=("immutabledict", "langdetect", "nltk"),
    )

    def additional_cache_ready(self, cache: BenchCache) -> bool:
        return _punkt_ready(cache)

    def download(self, ctx: DownloadContext) -> DownloadReport:
        dataset_ready = ctx.cache.is_ready(
            self.info.name, **cache_pins(self.info)
        )
        if not ctx.force and dataset_ready:
            punkt_was_ready = _punkt_ready(ctx.cache)
            try:
                _download_punkt(ctx.cache)
            except Exception as error:
                detail = str(error)
                if not isinstance(error, DatasetUnavailable):
                    detail = f"{type(error).__name__}: {error}"
                return DownloadReport(
                    adapter=self.info.name,
                    status="unavailable",
                    detail=f"pinned punkt_tab repair failed: {detail}",
                )
            action = "verified" if punkt_was_ready else "repaired"
            return DownloadReport(
                adapter=self.info.name,
                status="cached",
                detail=f"dataset cached; pinned punkt_tab {action}",
            )
        return super().download(ctx)

    def normalize(self, ctx: DownloadContext) -> list[dict]:
        from kairyu.bench.hub import load_hf_rows

        rows = load_hf_rows(
            self.info.hf_dataset,
            name="default",
            split="train",
            revision=self.info.hf_revision,
        )
        if len(rows) != _EXPECTED_ROWS:
            raise DatasetUnavailable(
                f"{self.info.hf_dataset}@{self.info.hf_revision} default/train yielded "
                f"{len(rows)} rows, expected exactly {_EXPECTED_ROWS}"
            )

        normalized: list[dict] = []
        keys: set[int] = set()
        prompts: set[str] = set()
        instruction_ids: set[str] = set()
        instruction_total = 0
        for index, row in enumerate(rows):
            key = row.get("key")
            prompt = row.get("prompt")
            ids = row.get("instruction_id_list")
            kwargs = row.get("kwargs")
            if (
                isinstance(key, bool)
                or not isinstance(key, int)
                or not isinstance(prompt, str)
                or not prompt
                or not isinstance(ids, list)
                or not ids
                or not all(isinstance(instruction_id, str) for instruction_id in ids)
                or not isinstance(kwargs, list)
                or not all(isinstance(entry, dict) for entry in kwargs)
                or len(ids) != len(kwargs)
            ):
                raise DatasetUnavailable(
                    f"IFEval row {index} has an invalid key/prompt/instruction/kwargs schema"
                )
            clean_kwargs = [
                {name: value for name, value in entry.items() if value is not None}
                for entry in kwargs
            ]
            for instruction_index, (instruction_id, entry) in enumerate(
                zip(ids, clean_kwargs, strict=True)
            ):
                required = _REQUIRED_KWARGS.get(instruction_id)
                if required is None:
                    raise DatasetUnavailable(
                        f"IFEval row {index} has unknown instruction ID {instruction_id!r}"
                    )
                if set(entry) != required or not _valid_instruction_args(
                    instruction_id, entry
                ):
                    raise DatasetUnavailable(
                        f"IFEval row {index} instruction {instruction_index} {instruction_id!r} "
                        "has missing, extra, or non-concrete checker arguments"
                    )
            if key in keys or prompt in prompts:
                raise DatasetUnavailable(f"IFEval row {index} duplicates a key or prompt")
            keys.add(key)
            prompts.add(prompt)
            instruction_ids.update(ids)
            instruction_total += len(ids)
            normalized.append(
                {
                    "id": f"ifeval-{key}",
                    "key": key,
                    "prompt": prompt,
                    "instruction_id_list": list(ids),
                    "kwargs": clean_kwargs,
                }
            )

        if instruction_total != _EXPECTED_INSTRUCTIONS:
            raise DatasetUnavailable(
                f"IFEval yielded {instruction_total} instructions, "
                f"expected exactly {_EXPECTED_INSTRUCTIONS}"
            )
        if instruction_ids != _EXPECTED_INSTRUCTION_IDS:
            missing = sorted(_EXPECTED_INSTRUCTION_IDS - instruction_ids)
            extra = sorted(instruction_ids - _EXPECTED_INSTRUCTION_IDS)
            raise DatasetUnavailable(
                f"IFEval instruction registry drifted (missing={missing}, extra={extra})"
            )

        _download_punkt(ctx.cache)
        return normalized

    def check_preconditions(self, target: BenchTarget, ctx: RunContext) -> str | None:
        # Check the normalized rows separately from the Punkt hook so a valid
        # dataset with a missing auxiliary asset gets an accurate repair hint.
        if not ctx.offline_fixtures and not ctx.cache.is_ready(
            self.info.name, **cache_pins(self.info)
        ):
            detail = ctx.download_failures.get(
                self.info.name, "run `kairyu bench download`"
            )
            return f"dataset not in cache ({detail})"
        dependency_error = _checker_dependency_error()
        if dependency_error is not None:
            return dependency_error
        if not ctx.offline_fixtures:
            if not _punkt_ready(ctx.cache):
                return "pinned IFEval punkt_tab resource unavailable; re-run download"
            try:
                _configure_punkt(ctx.cache)
            except Exception as error:
                return (
                    "pinned IFEval punkt_tab resource could not be initialized "
                    f"({type(error).__name__}: {error})"
                )
        return None

    def build_request(
        self, item: BenchItem, target: BenchTarget, ctx: RunContext
    ) -> ChatRequestSpec | SkipItem:
        # Byte-preserve the dataset prompt as the sole user message. Prefixes,
        # suffixes, and system messages invalidate repeat/start/end instructions.
        return ChatRequestSpec(
            messages=({"role": "user", "content": item.payload["prompt"]},),
            max_tokens=target.max_output_tokens,
        )

    async def score(
        self, item: BenchItem, response_text: str, ctx: RunContext
    ) -> ItemResult:
        from kairyu.bench._vendor.ifeval import IFEvalInput, evaluate_response

        inp = IFEvalInput(
            prompt=item.payload["prompt"],
            instruction_id_list=tuple(item.payload["instruction_id_list"]),
            kwargs=tuple(dict(entry) for entry in item.payload["kwargs"]),
        )
        evaluation = evaluate_response(inp, response_text)
        instruction_count = len(evaluation.strict)
        details = {
            "strict_instruction_following": list(evaluation.strict),
            "loose_instruction_following": list(evaluation.loose),
            "prompt_level_strict_accuracy": float(evaluation.strict_prompt),
            "instruction_level_strict_accuracy": (
                sum(evaluation.strict) / instruction_count
            ),
            "prompt_level_loose_accuracy": float(evaluation.loose_prompt),
            "instruction_level_loose_accuracy": sum(evaluation.loose) / instruction_count,
        }
        return ItemResult(
            item_id=item.id,
            status="completed",
            score=details["prompt_level_strict_accuracy"],
            response_excerpt=excerpt(response_text),
            details=details,
        )

    async def run(self, target: BenchTarget, ctx: RunContext) -> PairResult:
        result = await super().run(target, ctx)
        scored_details: list[dict] = []
        for item in result.items:
            if item.status != "completed":
                continue
            if item.sampling_attempts is None:
                if isinstance(item.details, dict):
                    scored_details.append(item.details)
                continue
            scored_details.extend(
                attempt.result.details
                for attempt in item.sampling_attempts
                if attempt.result.status == "completed"
                and isinstance(attempt.result.details, dict)
            )
        if not scored_details:
            return result

        strict = [
            followed
            for details in scored_details
            for followed in details["strict_instruction_following"]
        ]
        loose = [
            followed
            for details in scored_details
            for followed in details["loose_instruction_following"]
        ]
        prompt_strict = sum(
            details["prompt_level_strict_accuracy"] for details in scored_details
        ) / len(scored_details)
        prompt_loose = sum(
            details["prompt_level_loose_accuracy"] for details in scored_details
        ) / len(scored_details)
        metrics = {
            **result.metrics,
            "score": prompt_strict,
            "prompt_level_strict_accuracy": prompt_strict,
            "instruction_level_strict_accuracy": sum(strict) / len(strict),
            "prompt_level_loose_accuracy": prompt_loose,
            "instruction_level_loose_accuracy": sum(loose) / len(loose),
            "n_instructions_scored": len(strict),
        }
        return result.model_copy(update={"metrics": metrics})

    def methodology(self, ctx: RunContext) -> dict:
        methodology = super().methodology(ctx)
        methodology.update(
            {
                "config": "default",
                "split": "train",
                "expected_rows": _EXPECTED_ROWS,
                "expected_instructions": _EXPECTED_INSTRUCTIONS,
                "expected_instruction_ids": len(_EXPECTED_INSTRUCTION_IDS),
                "prompt": "dataset prompt byte-preserved as the sole user message",
                "max_tokens": "target.max_output_tokens (dataset reaches 1200 words/100 sentences)",
                "headline": "prompt_level_strict_accuracy",
                "metrics": [
                    "prompt_level_strict_accuracy",
                    "instruction_level_strict_accuracy",
                    "prompt_level_loose_accuracy",
                    "instruction_level_loose_accuracy",
                ],
                "checker": {
                    "source": "google-research/google-research:instruction_following_eval",
                    "revision": _CHECKER_REVISION,
                    "license": "Apache-2.0",
                    "langdetect_seed": 0,
                    "dataset_consistency_amendment": {
                        "keys": [1122, 1129],
                        "field": "keywords:letter_frequency.letter",
                        "values": ["#", "!"],
                        "reason": (
                            "upstream replaces punctuation with independently random "
                            "ASCII letters; Kairyu scores the pinned characters"
                        ),
                    },
                    "dependency_versions": {
                        package: {
                            "required": required,
                            "installed": _installed_checker_versions()[package],
                        }
                        for package, required in _CHECKER_DEPENDENCY_VERSIONS.items()
                    },
                },
                "punkt_tab": {
                    "source": "nltk/nltk_data:packages/tokenizers/punkt_tab.zip",
                    "revision": _PUNKT_REVISION,
                    "sha256": _PUNKT_SHA256,
                    "runtime_network": False,
                },
            }
        )
        if not ctx.offline_fixtures and ctx.cache.is_ready(
            self.info.name, **cache_pins(self.info)
        ):
            methodology["punkt_tab"]["verified"] = _punkt_ready(ctx.cache)
        return methodology
