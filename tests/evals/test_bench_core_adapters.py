"""Cheap core adapters: pinned normalization, exact scorers, and IFEval replay."""

from __future__ import annotations

import hashlib
import io
import json
import warnings
import zipfile

import httpx
import pytest
from conftest import make_config, make_target

from evals._vendor.ifeval import IFEvalInput, evaluate_response
from evals.adapters import all_adapters
from evals.adapters import ifeval as ifeval_module
from evals.adapters.base import (
    DownloadContext,
    RunContext,
    adapter_cache_ready,
    cache_pins,
)
from evals.adapters.gsm8k import Gsm8kAdapter, extract_gsm8k_answer
from evals.adapters.ifeval import IfevalAdapter
from evals.adapters.mmlu import (
    _CANONICAL_SUBJECTS,
    MmluAdapter,
)
from evals.cache import BenchCache
from evals.runner import SuiteRunner, _adapter_identity
from evals.types import (
    BenchItem,
    ContinuationLogLikelihood,
    DatasetUnavailable,
    LogLikelihoodResponse,
)


def _ctx(tmp_path, *, http_factory=None, offline_fixtures=True) -> RunContext:
    return RunContext(
        cache=BenchCache(tmp_path / "cache"),
        http_factory=http_factory or (lambda: httpx.AsyncClient()),
        offline_fixtures=offline_fixtures,
    )


def _install_fake_punkt_response(
    monkeypatch, *, duplicate: str | None = None, parseable: bool = False
):
    files = {
        filename: b"" if parseable else f"content:{filename}".encode()
        for filename in ifeval_module._PUNKT_FILES
    }
    archive_buffer = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(archive_buffer, "w") as archive:
            for filename, data in files.items():
                archive.writestr(f"punkt_tab/english/{filename}", data)
            if duplicate is not None:
                archive.writestr(f"punkt_tab/english/{duplicate}", files[duplicate])
            archive.writestr("../../outside", b"must not be extracted")
    archive_bytes = archive_buffer.getvalue()
    monkeypatch.setattr(
        ifeval_module, "_PUNKT_SHA256", hashlib.sha256(archive_bytes).hexdigest()
    )
    monkeypatch.setattr(
        ifeval_module,
        "_PUNKT_FILES",
        {
            filename: hashlib.sha256(data).hexdigest()
            for filename, data in files.items()
        },
    )
    response = httpx.Response(
        200,
        content=archive_bytes,
        request=httpx.Request("GET", "https://example.test/punkt.zip"),
    )
    monkeypatch.setattr(ifeval_module.httpx, "get", lambda *args, **kwargs: response)
    return files


def test_gsm8k_official_answer_extraction_boundaries() -> None:
    assert extract_gsm8k_answer("reasoning\n#### 1,234") == "1234"
    assert extract_gsm8k_answer("#### -7") == "-7"
    assert extract_gsm8k_answer("#### 18\nrestated #### 19") == "18"
    assert extract_gsm8k_answer("answer: 18") is None
    assert extract_gsm8k_answer("#### $18") is None


def test_gsm8k_normalize_pins_schema_count_and_nonempty_question(
    tmp_path, monkeypatch
) -> None:
    seen = {}
    rows = [{"question": f"Question {index}?", "answer": "work\n#### 1,000"}
            for index in range(1_319)]

    def fake_rows(dataset, *, name=None, split, revision=None, gated=False):
        seen.update(dataset=dataset, name=name, split=split, revision=revision)
        return rows

    monkeypatch.setattr("evals.hub.load_hf_rows", fake_rows)
    adapter = all_adapters()["gsm8k"]
    normalized = adapter.normalize(DownloadContext(BenchCache(tmp_path)))
    assert len(normalized) == 1_319
    assert normalized[0]["answer"] == "1000"
    assert seen == {
        "dataset": "openai/gsm8k",
        "name": "main",
        "split": "test",
        "revision": "740312add88f781978c0658806c59bc2815b9866",
    }

    rows[0] = {"question": " ", "answer": "#### 1"}
    with pytest.raises(DatasetUnavailable, match="string question/answer"):
        adapter.normalize(DownloadContext(BenchCache(tmp_path)))


async def test_gsm8k_request_and_exact_score(tmp_path) -> None:
    adapter = Gsm8kAdapter()
    item = BenchItem(id="g", payload={"question": "One plus one?", "answer": "2"})
    request = adapter.build_request(item, make_target(max_output_tokens=4096), _ctx(tmp_path))
    assert request.max_tokens == 1024
    assert request.messages[0]["content"].endswith("One plus one?")
    assert (await adapter.score(item, "work\n#### 2", _ctx(tmp_path))).score == 1.0
    assert (await adapter.score(item, "work\n#### 2.0", _ctx(tmp_path))).score == 0.0


def _mmlu_rows() -> list[dict]:
    subjects = sorted(_CANONICAL_SUBJECTS)
    return [
        {
            "question": f"Question {index}?",
            "subject": subjects[index % len(subjects)],
            "choices": ["first", "second", "third", "fourth"],
            "answer": index % 4,
        }
        for index in range(14_042)
    ]


def test_mmlu_normalize_requires_canonical_subjects_and_preserves_choices(
    tmp_path, monkeypatch
) -> None:
    rows = _mmlu_rows()
    seen = {}

    def fake_rows(dataset, *, name=None, split, revision=None, gated=False):
        seen.update(dataset=dataset, name=name, split=split, revision=revision)
        return rows

    monkeypatch.setattr("evals.hub.load_hf_rows", fake_rows)
    adapter = all_adapters()["mmlu"]
    normalized = adapter.normalize(DownloadContext(BenchCache(tmp_path)))
    assert len(normalized) == 14_042
    assert normalized[0]["choices"] == ["first", "second", "third", "fourth"]
    assert normalized[0]["answer"] == "A"
    assert seen == {
        "dataset": "cais/mmlu",
        "name": "all",
        "split": "test",
        "revision": "c30699e8356da336a370243923dbaf21066bb9fe",
    }

    rows[0] = {**rows[0], "subject": "not_a_canonical_subject"}
    with pytest.raises(DatasetUnavailable, match="subject set drifted"):
        adapter.normalize(DownloadContext(BenchCache(tmp_path)))


def test_mmlu_rejects_empty_question_and_choice(tmp_path, monkeypatch) -> None:
    rows = _mmlu_rows()
    rows[0] = {**rows[0], "question": "", "choices": ["first", "", "third", "fourth"]}
    monkeypatch.setattr("evals.hub.load_hf_rows", lambda *a, **k: rows)
    with pytest.raises(DatasetUnavailable, match="four string choices"):
        all_adapters()["mmlu"].normalize(DownloadContext(BenchCache(tmp_path)))


async def test_mmlu_request_is_zero_shot_fixed_order_and_annotated(tmp_path) -> None:
    adapter = MmluAdapter()
    item = BenchItem(
        id="m",
        payload={
            "question": "Q?",
            "subject": "abstract_algebra",
            "choices": ["one", "two", "three", "four"],
            "answer": "C",
        },
    )
    request = adapter.build_request(item, make_target(max_output_tokens=512), _ctx(tmp_path))
    prompt = request.context
    assert prompt.index("A. one") < prompt.index("B. two") < prompt.index("C. three")
    assert request.continuations == (" A", " B", " C", " D")
    assert request.reduction == "sum"
    assert not adapter.info.comparable_to_published
    assert "five demonstrations" in adapter.info.incomparable_reason
    response = LogLikelihoodResponse(
        reduction="sum",
        prompt_token_ids=(101, 102),
        candidates=tuple(
            ContinuationLogLikelihood(
                continuation=f" {letter}",
                token_ids=(index,),
                tokens=(f" {letter}",),
                token_logprobs=(value,),
                text_offsets=(0,),
                sum_logprob=value,
                score=value,
            )
            for index, (letter, value) in enumerate(
                zip("ABCD", (-3.0, -2.0, -0.1, -4.0), strict=True)
            )
        ),
    )
    result = await adapter.score(item, response, _ctx(tmp_path))
    assert result.score == 1.0
    assert result.details["winner"] == "C"
    assert result.details["candidates"][2]["text"] == "three"


_KWARG_VALUES = {
    "capital_frequency": 2,
    "capital_relation": "at least",
    "prompt_to_repeat": "Repeat this request.",
    "num_placeholders": 2,
    "postscript_marker": "P.S.",
    "num_sections": 2,
    "section_spliter": "SECTION",
    "num_bullets": 2,
    "num_highlights": 2,
    "keywords": ["orchid"],
    "forbidden_words": ["never"],
    "frequency": 2,
    "keyword": "orchid",
    "relation": "at least",
    "let_frequency": 2,
    "let_relation": "at least",
    "letter": "a",
    "language": "en",
    "first_word": "First",
    "nth_paragraph": 1,
    "num_paragraphs": 2,
    "num_sentences": 2,
    "num_words": 20,
    "end_phrase": "THE END",
}


def _ifeval_rows() -> list[dict]:
    instruction_ids = sorted(ifeval_module._EXPECTED_INSTRUCTION_IDS)
    cursor = 0
    rows = []
    # 293*2 + 248*1 = the pinned 834 instructions across 541 prompts.
    for index in range(541):
        count = 2 if index < 293 else 1
        ids = [instruction_ids[(cursor + offset) % len(instruction_ids)] for offset in range(count)]
        cursor += count
        kwargs = []
        for instruction_id in ids:
            entry = {
                key: _KWARG_VALUES[key]
                for key in ifeval_module._REQUIRED_KWARGS[instruction_id]
            }
            entry["unused_dense_field"] = None
            kwargs.append(entry)
        rows.append(
            {
                "key": index,
                "prompt": f"Pinned prompt {index}",
                "instruction_id_list": ids,
                "kwargs": kwargs,
            }
        )
    return rows


def test_ifeval_request_byte_preserves_prompt_as_sole_user_message(tmp_path) -> None:
    prompt = "  Keep leading whitespace.\r\nUnicode: 雨\n\nKeep trailing whitespace.  "
    item = BenchItem(
        id="i",
        payload={
            "prompt": prompt,
            "instruction_id_list": ["punctuation:no_comma"],
            "kwargs": [{}],
        },
    )
    request = IfevalAdapter().build_request(
        item, make_target(max_output_tokens=12_345), _ctx(tmp_path)
    )
    assert request.messages == ({"role": "user", "content": prompt},)
    assert request.max_tokens == 12_345


def test_ifeval_normalize_validates_full_shape_and_strips_dense_nulls(
    tmp_path, monkeypatch
) -> None:
    rows = _ifeval_rows()
    seen = {}

    def fake_rows(dataset, *, name=None, split, revision=None, gated=False):
        seen.update(dataset=dataset, name=name, split=split, revision=revision)
        return rows

    monkeypatch.setattr("evals.hub.load_hf_rows", fake_rows)
    monkeypatch.setattr(ifeval_module, "_download_punkt", lambda cache: None)
    normalized = all_adapters()["ifeval"].normalize(DownloadContext(BenchCache(tmp_path)))
    assert len(normalized) == 541
    assert sum(len(row["instruction_id_list"]) for row in normalized) == 834
    assert all(
        "unused_dense_field" not in entry
        for row in normalized
        for entry in row["kwargs"]
    )
    assert seen == {
        "dataset": "google/IFEval",
        "name": "default",
        "split": "train",
        "revision": "966cd89545d6b6acfd7638bc708b98261ca58e84",
    }


def test_ifeval_normalize_rejects_random_fallback_kwargs(tmp_path, monkeypatch) -> None:
    rows = _ifeval_rows()
    instruction_id = rows[0]["instruction_id_list"][0]
    required_key = next(iter(ifeval_module._REQUIRED_KWARGS[instruction_id]), None)
    if required_key is None:
        # Find an instruction with a score-bearing argument.
        row = next(
            row
            for row in rows
            if ifeval_module._REQUIRED_KWARGS[row["instruction_id_list"][0]]
        )
        instruction_id = row["instruction_id_list"][0]
        required_key = next(iter(ifeval_module._REQUIRED_KWARGS[instruction_id]))
        row["kwargs"][0][required_key] = None
    else:
        rows[0]["kwargs"][0][required_key] = None
    monkeypatch.setattr("evals.hub.load_hf_rows", lambda *a, **k: rows)
    monkeypatch.setattr(ifeval_module, "_download_punkt", lambda cache: None)
    with pytest.raises(DatasetUnavailable, match="checker arguments"):
        all_adapters()["ifeval"].normalize(DownloadContext(BenchCache(tmp_path)))


@pytest.mark.parametrize(
    ("instruction_id", "argument", "bad_value"),
    [
        ("length_constraints:number_words", "num_words", 0),
        ("length_constraints:number_words", "num_words", "20"),
        ("keywords:letter_frequency", "letter", "two"),
        ("keywords:existence", "keywords", "orchid"),
        ("keywords:frequency", "relation", "approximately"),
        ("language:response_language", "language", "not-a-language"),
    ],
)
def test_ifeval_normalize_rejects_values_that_trigger_or_bypass_checker_contract(
    tmp_path, monkeypatch, instruction_id, argument, bad_value
) -> None:
    rows = _ifeval_rows()
    row = next(
        candidate for candidate in rows if instruction_id in candidate["instruction_id_list"]
    )
    position = row["instruction_id_list"].index(instruction_id)
    row["kwargs"][position][argument] = bad_value
    monkeypatch.setattr("evals.hub.load_hf_rows", lambda *args, **kwargs: rows)
    monkeypatch.setattr(ifeval_module, "_download_punkt", lambda cache: None)
    with pytest.raises(DatasetUnavailable, match="checker arguments"):
        all_adapters()["ifeval"].normalize(DownloadContext(BenchCache(tmp_path)))


def test_ifeval_normalize_rejects_nth_paragraph_beyond_paragraph_count(
    tmp_path, monkeypatch
) -> None:
    rows = _ifeval_rows()
    instruction_id = "length_constraints:nth_paragraph_first_word"
    row = next(
        candidate for candidate in rows if instruction_id in candidate["instruction_id_list"]
    )
    position = row["instruction_id_list"].index(instruction_id)
    row["kwargs"][position]["num_paragraphs"] = 1
    row["kwargs"][position]["nth_paragraph"] = 2
    monkeypatch.setattr("evals.hub.load_hf_rows", lambda *args, **kwargs: rows)
    monkeypatch.setattr(ifeval_module, "_download_punkt", lambda cache: None)
    with pytest.raises(DatasetUnavailable, match="checker arguments"):
        all_adapters()["ifeval"].normalize(DownloadContext(BenchCache(tmp_path)))


def test_ifeval_official_loose_transform_can_remove_first_line() -> None:
    inp = IFEvalInput(
        prompt="Wrap the answer in quotation marks.",
        instruction_id_list=("startend:quotation",),
        kwargs=({},),
    )
    result = evaluate_response(inp, 'preamble\n"answer"')
    assert result.strict == (False,)
    assert result.loose == (True,)


@pytest.mark.parametrize("letter", ["#", "!"])
def test_ifeval_letter_frequency_scores_pinned_punctuation_without_random_fallback(
    monkeypatch, letter
) -> None:
    from evals._vendor.ifeval import instructions, instructions_util

    def unexpected_random(*args, **kwargs):
        raise AssertionError("valid pinned checker arguments must never use random fallback")

    monkeypatch.setattr(instructions.random, "choice", unexpected_random)
    monkeypatch.setattr(instructions.random, "randint", unexpected_random)
    monkeypatch.setattr(instructions_util.random, "sample", unexpected_random)
    inp = IFEvalInput(
        prompt="Use the required character twice.",
        instruction_id_list=("keywords:letter_frequency",),
        kwargs=(
            {
                "letter": letter,
                "let_frequency": 2,
                "let_relation": "at least",
            },
        ),
    )
    followed = evaluate_response(inp, f"answer {letter}{letter}")
    missed = evaluate_response(inp, f"answer {letter}")
    assert followed.strict == followed.loose == (True,)
    assert missed.strict == missed.loose == (False,)


def test_ifeval_all_normalized_checker_args_construct_without_random_fallback(
    monkeypatch,
) -> None:
    from evals._vendor.ifeval import checker, instructions, instructions_util

    def unexpected_random(*args, **kwargs):
        raise AssertionError("normalized checker arguments must never use random fallback")

    monkeypatch.setattr(instructions.random, "choice", unexpected_random)
    monkeypatch.setattr(instructions.random, "randint", unexpected_random)
    monkeypatch.setattr(instructions_util.random, "sample", unexpected_random)
    for row in _ifeval_rows():
        inp = IFEvalInput(
            prompt=row["prompt"],
            instruction_id_list=tuple(row["instruction_id_list"]),
            kwargs=tuple(
                {
                    name: value
                    for name, value in entry.items()
                    if value is not None
                }
                for entry in row["kwargs"]
            ),
        )
        for index in range(len(inp.instruction_id_list)):
            checker._instruction(inp, index)


async def test_ifeval_direct_score_is_completed_with_per_item_details(tmp_path) -> None:
    item = BenchItem(
        id="i",
        payload={
            "prompt": "Do not use a comma.",
            "instruction_id_list": ["punctuation:no_comma"],
            "kwargs": [{}],
        },
    )
    result = await IfevalAdapter().score(item, "A short answer.", _ctx(tmp_path))
    assert result.status == "completed"
    assert result.score == 1.0
    assert result.details == {
        "strict_instruction_following": [True],
        "loose_instruction_following": [True],
        "prompt_level_strict_accuracy": 1.0,
        "instruction_level_strict_accuracy": 1.0,
        "prompt_level_loose_accuracy": 1.0,
        "instruction_level_loose_accuracy": 1.0,
    }


async def test_ifeval_offline_pair_retains_all_four_metrics(tmp_path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        prompt = json.loads(request.read())["messages"][0]["content"]
        content = "ORCHID COMPASS" if "ORCHID" in prompt else "Blue skies are calm."
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": content}}]},
        )

    factory = lambda: httpx.AsyncClient(  # noqa: E731 - test factory is intentionally tiny
        transport=httpx.MockTransport(handler), base_url="http://gw"
    )
    result = await IfevalAdapter().run(make_target(), _ctx(tmp_path, http_factory=factory))
    assert result.status == "completed"
    assert result.metrics["score"] == 1.0
    assert result.metrics["prompt_level_strict_accuracy"] == 1.0
    assert result.metrics["instruction_level_strict_accuracy"] == 1.0
    assert result.metrics["prompt_level_loose_accuracy"] == 1.0
    assert result.metrics["instruction_level_loose_accuracy"] == 1.0
    assert result.metrics["n_instructions_scored"] == 2


def test_punkt_download_verifies_members_and_never_extracts_unlisted_paths(
    tmp_path, monkeypatch
) -> None:
    _install_fake_punkt_response(monkeypatch)
    cache = BenchCache(tmp_path / "cache")
    ifeval_module._download_punkt(cache)
    assert ifeval_module._punkt_ready(cache)
    assert not (tmp_path / "outside").exists()


def test_punkt_download_rejects_duplicate_required_member(
    tmp_path, monkeypatch
) -> None:
    duplicate = next(iter(ifeval_module._PUNKT_FILES))
    _install_fake_punkt_response(monkeypatch, duplicate=duplicate)
    with pytest.raises(DatasetUnavailable, match="exactly one"):
        ifeval_module._download_punkt(BenchCache(tmp_path / "cache"))


def test_ifeval_normal_download_and_runner_repair_corrupt_punkt_without_data_rewrite(
    tmp_path, monkeypatch
) -> None:
    _install_fake_punkt_response(monkeypatch)
    adapter = all_adapters()["ifeval"]
    cache = BenchCache(tmp_path / "cache")
    cache.write_rows(
        adapter.info.name,
        [{"id": "cached-row"}],
        cache_pins(adapter.info),
    )
    ifeval_module._download_punkt(cache)
    data_before = cache.data_path(adapter.info.name).read_bytes()
    manifest_before = cache.manifest_path(adapter.info.name).read_bytes()

    member = next(iter(ifeval_module._PUNKT_FILES))
    ifeval_module._punkt_file(cache, member).write_bytes(b"corrupt")
    assert not adapter_cache_ready(adapter, cache)
    assert _adapter_identity(adapter, cache, offline_fixtures=False)["unavailable"]

    report = adapter.download(DownloadContext(cache=cache))
    assert report.status == "cached"
    assert "repaired" in report.detail
    assert adapter_cache_ready(adapter, cache)
    assert cache.data_path(adapter.info.name).read_bytes() == data_before
    assert cache.manifest_path(adapter.info.name).read_bytes() == manifest_before

    ifeval_module._punkt_file(cache, member).write_bytes(b"corrupt again")
    config = make_config(
        tmp_path,
        models=("m",),
        suite="core",
        offline_fixtures=False,
        download=True,
    )
    runner = SuiteRunner(config, http_factory=lambda: None, probe_docker=lambda: (False, ""))
    ctx = RunContext(cache=cache, http_factory=lambda: None)
    runner._download_missing([adapter], cache, ctx)
    assert adapter_cache_ready(adapter, cache)
    assert cache.data_path(adapter.info.name).read_bytes() == data_before
    assert cache.manifest_path(adapter.info.name).read_bytes() == manifest_before


def test_ifeval_cached_download_fails_closed_when_punkt_repair_is_unavailable(
    tmp_path, monkeypatch
) -> None:
    adapter = all_adapters()["ifeval"]
    cache = BenchCache(tmp_path / "cache")
    cache.write_rows(adapter.info.name, [{"id": "cached-row"}], cache_pins(adapter.info))
    manifest_before = cache.manifest_path(adapter.info.name).read_bytes()

    def unavailable(*args, **kwargs):
        raise DatasetUnavailable("network unavailable")

    monkeypatch.setattr(ifeval_module, "_download_punkt", unavailable)
    report = adapter.download(DownloadContext(cache=cache))
    assert report.status == "unavailable"
    assert "repair failed" in report.detail
    assert cache.manifest_path(adapter.info.name).read_bytes() == manifest_before


def test_ifeval_precondition_distinguishes_missing_punkt_from_missing_dataset(
    tmp_path
) -> None:
    adapter = all_adapters()["ifeval"]
    cache = BenchCache(tmp_path / "cache")
    cache.write_rows(adapter.info.name, [{"id": "cached-row"}], cache_pins(adapter.info))
    reason = adapter.check_preconditions(
        make_target(),
        RunContext(cache=cache, http_factory=lambda: None, offline_fixtures=False),
    )
    assert reason is not None
    assert "punkt_tab resource unavailable" in reason
    assert "dataset not in cache" not in reason


def test_configure_punkt_clears_ambient_vendor_and_nltk_tokenizer_caches(
    tmp_path, monkeypatch
) -> None:
    import nltk

    from evals._vendor.ifeval import instructions_util

    ambient_root = tmp_path / "ambient"
    ambient_dir = ambient_root / "tokenizers" / "punkt_tab" / "english"
    ambient_dir.mkdir(parents=True)
    for filename in ifeval_module._PUNKT_FILES:
        (ambient_dir / filename).write_bytes(b"")
    (ambient_dir / "abbrev_types.txt").write_text("ambient\n", encoding="utf-8")
    monkeypatch.setattr(nltk.data, "path", [str(ambient_root)])
    instructions_util._get_sentence_tokenizer.cache_clear()
    nltk.tokenize._get_punkt_tokenizer.cache_clear()
    ambient_vendor = instructions_util._get_sentence_tokenizer()
    ambient_nltk = nltk.tokenize._get_punkt_tokenizer("english")

    _install_fake_punkt_response(monkeypatch, parseable=True)
    cache = BenchCache(tmp_path / "cache")
    ifeval_module._download_punkt(cache)
    ifeval_module._configure_punkt(cache)
    pinned_vendor = instructions_util._get_sentence_tokenizer()
    pinned_nltk = nltk.tokenize._get_punkt_tokenizer("english")

    assert pinned_vendor is not ambient_vendor
    assert pinned_nltk is not ambient_nltk
    assert "ambient" in ambient_vendor._params.abbrev_types
    assert "ambient" in ambient_nltk._params.abbrev_types
    assert "ambient" not in pinned_vendor._params.abbrev_types
    assert "ambient" not in pinned_nltk._params.abbrev_types
    assert nltk.data.path[0] == str(ifeval_module._punkt_root(cache))


def test_ifeval_dependency_versions_fail_closed(monkeypatch) -> None:
    expected = dict(ifeval_module._CHECKER_DEPENDENCY_VERSIONS)

    def wrong_nltk(package):
        return "9.9.9" if package == "nltk" else expected[package]

    monkeypatch.setattr(ifeval_module, "distribution_version", wrong_nltk)
    assert "requires nltk==3.9.2" in ifeval_module._checker_dependency_error()

    def missing_langdetect(package):
        if package == "langdetect":
            raise ifeval_module.PackageNotFoundError(package)
        return expected[package]

    monkeypatch.setattr(ifeval_module, "distribution_version", missing_langdetect)
    assert "requires langdetect==1.0.9" in ifeval_module._checker_dependency_error()
