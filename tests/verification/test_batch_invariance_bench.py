"""Portable operator tests for the A12 native evidence wrapper."""

from __future__ import annotations

import asyncio
import copy
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from verification.l1.correctness import batch_invariance as gate
from verification.l1.correctness import batch_invariance_bench as operator

SCRIPT = Path(operator.__file__).resolve()
REPO_ROOT = SCRIPT.parents[1]


class _VirtualVocab(list[str]):
    def __len__(self) -> int:
        return gate.TOKENIZER_VOCAB_SIZE

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [f"piece-{item}" for item in range(*index.indices(len(self)))]
        return f"piece-{index}"


class _LayerContainer:
    """ModuleList-like iterable that deliberately is not a list/tuple."""

    def __init__(self, *layers: object) -> None:
        self._layers = layers

    def __iter__(self):
        return iter(self._layers)


class _Tokenizer:
    def __init__(self) -> None:
        prompts = gate.fixed_prompts()
        descriptors = [
            prompts["target"],
            *prompts["distractors"],
            *prompts["pressure"],
        ]
        self._encoded = {
            descriptor["text"]: tuple(descriptor["token_ids"])
            for descriptor in descriptors
        }
        self._vocab = _VirtualVocab()

    def encode(self, text: str) -> tuple[int, ...]:
        return self._encoded[text]

    def vocab(self) -> list[str]:
        return self._vocab


class _Scheduler:
    def __init__(self) -> None:
        self._next = None

    def schedule(self):
        assert self._next is not None
        plan = self._next
        self._next = None
        return plan

    def emit(self, step: dict[str, object]) -> None:
        request_ids = step["request_ids"]
        num_tokens = step["num_tokens"]
        assert isinstance(request_ids, list) and isinstance(num_tokens, list)
        self._next = SimpleNamespace(
            scheduled=tuple(
                SimpleNamespace(
                    request_id=request_id,
                    num_tokens=count,
                    is_prefill=step["kind"] == "prefill",
                )
                for request_id, count in zip(request_ids, num_tokens, strict=True)
            )
        )
        self.schedule()


class _Runner:
    def __init__(self) -> None:
        self.phase = "startup"
        self.prefill_resets = 0
        self.decode_resets = 0

    def prefill_execution_stats(self, *, reset: bool = False):
        if reset:
            self.prefill_resets += 1
        return ({"phase": self.phase, "reset": self.prefill_resets},)

    def verification_execution_stats(self, *, reset: bool = False):
        if reset:
            self.decode_resets += 1
        return ({"phase": self.phase, "reset": self.decode_resets},)


class _Cache:
    def __init__(self) -> None:
        self.target_cached_tokens = 0

    def peek_cached_tokens(self, token_ids: tuple[int, ...]) -> int:
        if token_ids == gate.TARGET_PROMPT_TOKEN_IDS:
            return self.target_cached_tokens
        return 0


class _Backend:
    def __init__(self, *, chunk: bool = False) -> None:
        self.chunk = chunk
        self._scheduler = _Scheduler()
        self._runner = _Runner()
        self._loop = SimpleNamespace(
            tp_launcher=SimpleNamespace(runner=self._runner)
        )
        self._cache = _Cache()
        self._batch_pending: list[tuple[object, asyncio.Future, int]] = []
        self._pressure_count = 0

    def _emit(self, scenario: str, request_ids: list[str]) -> None:
        self._runner.phase = scenario
        for step in gate._expected_scheduler(scenario, request_ids):
            self._scheduler.emit(step)

    @staticmethod
    def _result(request: object, cached_tokens: int) -> object:
        token_ids = tuple(range(100, 100 + gate.OUTPUT_TOKENS))
        return SimpleNamespace(
            finished=True,
            prompt_token_ids=request.token_ids,
            completions=(
                SimpleNamespace(
                    token_ids=token_ids,
                    text="native fake answer",
                    finish_reason="length",
                ),
            ),
            usage=SimpleNamespace(
                prompt_tokens=len(request.token_ids),
                completion_tokens=gate.OUTPUT_TOKENS,
                cached_tokens=cached_tokens,
            ),
        )

    async def generate(self, request: object) -> object:
        request_id = request.request_id
        cached = self._cache.peek_cached_tokens(request.token_ids)
        if request_id.startswith("batch32-"):
            future = asyncio.get_running_loop().create_future()
            self._batch_pending.append((request, future, cached))
            if len(self._batch_pending) == gate.BATCH_SIZE:
                request_ids = [item.request_id for item, _future, _cached in self._batch_pending]
                self._emit("batch32-cold", request_ids)
                self._cache.target_cached_tokens = gate.WARM_CACHED_TOKENS
                for item, pending, item_cached in self._batch_pending:
                    pending.set_result(self._result(item, item_cached))
            return await future

        if request_id.startswith("pressure-"):
            self._emit("pressure", [request_id])
            self._pressure_count += 1
            if self._pressure_count == gate.MAX_PRESSURE_REQUESTS:
                self._cache.target_cached_tokens = 0
        elif self.chunk:
            self._emit("chunked-alone-cold", [request_id])
            self._cache.target_cached_tokens = gate.WARM_CACHED_TOKENS
        elif request_id == "alone-cold-target":
            self._emit("alone-cold", [request_id])
            self._cache.target_cached_tokens = gate.WARM_CACHED_TOKENS
        else:
            self._emit("alone-warm", [request_id])
        return self._result(request, cached)


def _request_factory(
    request_id: str,
    token_ids: tuple[int, ...],
    sampling: object,
) -> object:
    assert sampling == gate.SAMPLING_CONFIG
    return SimpleNamespace(request_id=request_id, token_ids=token_ids)


def test_fake_native_runtime_executes_all_fixed_phases_and_cache_transition() -> None:
    async def capture():
        tokenizer = _Tokenizer()
        prompts = operator._fixed_prompt_descriptors(tokenizer)
        full_backend = _Backend()
        full = await operator._capture_full_backend(
            backend=full_backend,
            tokenizer=tokenizer,
            request_factory=_request_factory,
            runtime_id="full",
            runtime_nonce="a" * 32,
            prompts=prompts,
            sampling=gate.SAMPLING_CONFIG,
        )
        chunk_backend = _Backend(chunk=True)
        chunk = await operator._capture_chunk_backend(
            backend=chunk_backend,
            tokenizer=tokenizer,
            request_factory=_request_factory,
            runtime_id="chunk",
            runtime_nonce="b" * 32,
            target=prompts["target"],
            sampling=gate.SAMPLING_CONFIG,
        )
        return full_backend, full, chunk_backend, chunk

    full_backend, full, chunk_backend, chunk = asyncio.run(capture())

    batch_ids = ["batch32-target"] + [
        f"batch32-distractor-{index:02d}"
        for index in range(gate.DISTRACTOR_COUNT)
    ]
    assert full["batch_structure"]["scheduler_steps"] == gate._expected_scheduler(
        "batch32-cold", batch_ids
    )
    assert full["pressure"]["target_cache_before"] == gate.WARM_CACHED_TOKENS
    assert full["pressure"]["target_cache_after"] == 0
    assert len(full["pressure"]["requests"]) == gate.MAX_PRESSURE_REQUESTS == 6
    pressure_ids = [
        f"pressure-{index:02d}" for index in range(gate.MAX_PRESSURE_REQUESTS)
    ]
    assert full["pressure"]["structure"]["scheduler_steps"] == gate._expected_scheduler(
        "pressure", pressure_ids
    )
    assert full["alone_cold"]["cache"] == {
        "probe_before": 0,
        "native_cached_tokens": 0,
    }
    assert full["alone_warm"]["cache"] == {
        "probe_before": gate.WARM_CACHED_TOKENS,
        "native_cached_tokens": gate.WARM_CACHED_TOKENS,
    }
    assert chunk["request"]["cache"] == {
        "probe_before": 0,
        "native_cached_tokens": 0,
    }
    assert [
        step["num_tokens"][0]
        for step in chunk["structure"]["scheduler_steps"]
        if step["kind"] == "prefill"
    ] == [32, 32, 32, 32, 1]
    answer_fields = ("token_ids", "token_pieces", "text")
    answers = [
        {
            field: observation["response"][field]
            for field in answer_fields
        }
        for observation in (
            full["batch"][0],
            full["alone_cold"],
            full["alone_warm"],
            chunk["request"],
        )
    ]
    assert answers == [answers[0]] * 4
    assert full_backend._runner.prefill_resets == 4
    assert full_backend._runner.decode_resets == 4
    assert chunk_backend._runner.prefill_resets == 1
    assert chunk_backend._runner.decode_resets == 1


def _fake_hardware() -> dict[str, object]:
    return {
        "gpus": [
            {"uuid": f"GPU-{rank + 1:08x}-0000-0000-0000-{rank + 1:012x}"}
            for rank in range(gate.TENSOR_PARALLEL_SIZE)
        ]
    }


def _runtime_backend(config: dict[str, object]) -> object:
    ownership = tuple(
        {
            "rank": rank,
            "control_world_size": gate.TENSOR_PARALLEL_SIZE,
            "control_backend": "gloo",
            "model_world_size": gate.TENSOR_PARALLEL_SIZE,
            "model_backend": "nccl",
            "sampling_owner": rank == 0,
            "sampler_present": rank == 0,
            "device": f"cuda:{rank}",
        }
        for rank in range(gate.TENSOR_PARALLEL_SIZE)
    )
    graph = SimpleNamespace(
        configured_buckets=gate.CUDA_GRAPH_BUCKETS,
        max_pages=16,
        _backend=SimpleNamespace(_warmup_iters=3),
    )
    components = {
        "prefill": "flashinfer",
        "decode": "flashinfer",
        "kv_mode": "paged-direct",
    }
    architecture = {
        "arch": "cuda",
        "device_name": gate.GPU_NAME,
        "sm": 120,
        "kernel_tier": "fa2",
    }
    decision = SimpleNamespace(
        requested="flashinfer",
        resolved="flashinfer",
        source="env",
        components=components,
        architecture=architecture,
    )
    selected_identity = gate.canonical_json(
        {
            "requested": "flashinfer",
            "resolved": "flashinfer",
            "source": "env",
            "components": components,
            "architecture": architecture,
        }
    )
    attention_backend = SimpleNamespace(
        _flashinfer=SimpleNamespace(__version__="0.6.14")
    )
    layer = SimpleNamespace(self_attn=SimpleNamespace(backend=attention_backend))
    local = SimpleNamespace(
        _graph=graph,
        _model=SimpleNamespace(
            model=SimpleNamespace(layers=_LayerContainer(layer))
        ),
    )
    runner = SimpleNamespace(
        _local=local,
        sampling_ownership_metadata=lambda: ownership,
    )
    launcher = SimpleNamespace(
        runner=runner,
        attention_backend_identity=selected_identity,
    )
    loop = SimpleNamespace(
        tp_launcher=launcher,
        _pipeline_depth=config["pipeline_depth"],
        _max_model_len=config["max_model_len"],
    )
    return SimpleNamespace(
        tensor_parallel_size=config["tensor_parallel_size"],
        expert_parallel_size=config["expert_parallel_size"],
        _loop=loop,
        _cache=SimpleNamespace(
            page_size=config["page_size"], num_pages=config["num_pages"]
        ),
        _scheduler=SimpleNamespace(
            _max_seqs=config["max_num_seqs"],
            _budget=config["max_num_batched_tokens"],
        ),
        kv_cache_dtype_requested="bfloat16",
        kv_cache_dtype_resolved="bfloat16",
        generation_defaults=SimpleNamespace(mode="none"),
        attention_backend_decision=decision,
    )


@pytest.mark.parametrize("runtime_id", ["full", "chunk"])
def test_runtime_record_is_derived_from_native_objects(runtime_id: str) -> None:
    config = copy.deepcopy(
        gate.FULL_RUNTIME_CONFIG
        if runtime_id == "full"
        else gate.CHUNK_RUNTIME_CONFIG
    )
    record = operator._runtime_record(
        backend=_runtime_backend(config),
        runtime_id=runtime_id,
        runtime_nonce="a" * 32,
        hardware=_fake_hardware(),
    )

    assert record["engine"] == config
    assert record["topology"]["control_backend"] == "gloo"
    assert record["topology"]["model_backend"] == "nccl"
    assert record["topology"]["sampling_ownership"][0]["sampling_owner"] is True
    assert all(
        row["sampling_owner"] is False
        for row in record["topology"]["sampling_ownership"][1:]
    )
    assert record["attention"]["requested_backend"] == "flashinfer"
    identity = json.loads(record["attention"]["execution_identity"])
    assert set(identity) == {
        "requested",
        "resolved",
        "source",
        "components",
        "component_versions",
        "component_builds",
        "architecture",
    }
    assert identity["component_versions"] == {
        "decode": "0.6.14",
        "prefill": "0.6.14",
    }


def test_runtime_record_rejects_echoed_config_when_native_geometry_differs() -> None:
    config = copy.deepcopy(gate.FULL_RUNTIME_CONFIG)
    backend = _runtime_backend(config)
    backend._scheduler._budget = 32

    with pytest.raises(ValueError, match="max_num_batched_tokens"):
        operator._runtime_record(
            backend=backend,
            runtime_id="full",
            runtime_nonce="a" * 32,
            hardware=_fake_hardware(),
        )


def test_hardware_normalizes_torch_uuid_and_records_exact_software(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    properties = [
        SimpleNamespace(
            uuid=f"{rank + 1:08x}-0000-0000-0000-{rank + 1:012x}",
            name=gate.GPU_NAME,
            major=12,
            minor=0,
            total_memory=100_000_000_000 + rank,
        )
        for rank in range(gate.TENSOR_PARALLEL_SIZE)
    ]
    fake_torch = SimpleNamespace(
        __version__="2.12.1+cu130",
        version=SimpleNamespace(cuda="13.0"),
        cuda=SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: gate.TENSOR_PARALLEL_SIZE,
            get_device_properties=lambda rank: properties[rank],
            nccl=SimpleNamespace(version=lambda: (2, 29, 7)),
        ),
    )

    def command_csv(command, *, description):
        del description
        if any("query-compute-apps" in item for item in command):
            return []
        return [
            [
                gate.GPU_NAME,
                f"GPU-{rank + 1:08x}-0000-0000-0000-{rank + 1:012x}",
                f"00000000:{rank + 1:02X}:00.0",
                "580.0",
            ]
            for rank in range(gate.TENSOR_PARALLEL_SIZE)
        ]

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(operator, "_command_csv", command_csv)
    monkeypatch.setattr(operator.importlib.metadata, "version", lambda _name: "0.6.14")

    hardware = operator._hardware()

    assert hardware["gpu_count"] == 8
    assert hardware["gpu_exclusive"] is True
    assert hardware["gpus"][0]["uuid"].startswith("GPU-")
    assert hardware["gpus"][7]["pci_bus_id"] == "00000000:08:00.0"
    assert hardware["software"] == {
        "python_version": operator.platform.python_version(),
        "torch_version": "2.12.1+cu130",
        "cuda_version": "13.0",
        "driver_version": "580.0",
        "nccl_version": "2.29.7",
        "flashinfer_version": "0.6.14",
    }


def test_output_directory_creation_is_exclusive(tmp_path: Path) -> None:
    target = operator._planned_output_directory(tmp_path / "artifact")
    operator._create_output_directory(target)

    assert target.is_dir()
    with pytest.raises(ValueError, match="overwrite"):
        operator._planned_output_directory(target)
    with pytest.raises(ValueError, match="overwrite"):
        operator._create_output_directory(target)


def test_run_native_awaits_async_shutdown_when_runtime_attestation_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()
    shutdown_finished = False

    class Backend:
        async def shutdown(self) -> None:
            nonlocal shutdown_finished
            await asyncio.sleep(0)
            shutdown_finished = True

    monkeypatch.setattr(operator, "_source_snapshot", lambda: {"source": True})
    monkeypatch.setattr(
        operator, "_live_checkpoint", lambda _path: {"checkpoint": True}
    )
    monkeypatch.setattr(operator, "_hardware", lambda: {"hardware": True})
    monkeypatch.setattr(
        operator, "_fixed_prompt_descriptors", lambda _tokenizer: gate.fixed_prompts()
    )
    monkeypatch.setattr(operator, "_request_factory", lambda: (object(), object()))
    monkeypatch.setattr(operator, "_native_backend", lambda _path, _config: Backend())
    monkeypatch.setattr(
        operator,
        "_runtime_record",
        lambda **_kwargs: (_ for _ in ()).throw(ValueError("runtime attestation")),
    )
    import kairyu.engine.tokenizer as tokenizer_module

    monkeypatch.setattr(tokenizer_module, "resolve_tokenizer", lambda _path: object())
    args = SimpleNamespace(model_path=model_path)

    with pytest.raises(ValueError, match="runtime attestation"):
        asyncio.run(operator._run_native(args))
    assert shutdown_finished is True


def test_main_retains_failed_artifact_bytes_after_assertion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "failed-artifact"

    async def failed_rows(_args):
        return [{"failed": True}]

    def write(raw: Path, manifest: Path, rows, *, assert_gate: bool):
        assert rows == [{"failed": True}]
        assert assert_gate is True
        raw.write_bytes(b"retained failed raw\n")
        manifest.write_bytes(b"retained failed manifest\n")
        raise gate.EvidenceError("synthetic gate failure")

    monkeypatch.setattr(operator, "_run_native", failed_rows)
    monkeypatch.setattr(operator.gate, "write_artifact", write)

    with pytest.raises(gate.EvidenceError, match="synthetic gate failure"):
        operator.main(
            [
                "run-native",
                "--model-path",
                str(tmp_path / "model"),
                "--output-dir",
                str(output_dir),
                "--assert-gate",
            ]
        )
    assert (output_dir / gate.RAW_NAME).read_bytes() == b"retained failed raw\n"
    assert (output_dir / gate.MANIFEST_NAME).read_bytes() == (
        b"retained failed manifest\n"
    )


def test_source_preflight_uses_real_git_bytes_and_rejects_tracked_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "source"
    repo.mkdir()
    source = repo / "one.py"
    source.write_text("one\n", encoding="utf-8")
    subprocess.run(("git", "init", "-q"), cwd=repo, check=True)
    subprocess.run(("git", "add", "one.py"), cwd=repo, check=True)
    subprocess.run(
        (
            "git",
            "-c",
            "user.name=A12 Test",
            "-c",
            "user.email=a12@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ),
        cwd=repo,
        check=True,
    )
    monkeypatch.setattr(operator, "_REPO_ROOT", repo)
    monkeypatch.setattr(operator, "_CHECKOUT_MODULE_PREFIXES", ())
    monkeypatch.setattr(operator.gate, "REQUIRED_SOURCE_PATHS", ("one.py",))
    monkeypatch.setitem(
        sys.modules,
        "external.relative_origin",
        SimpleNamespace(
            __file__=None,
            __spec__=SimpleNamespace(origin="_ops.py"),
        ),
    )
    monkeypatch.chdir(repo)

    operator._assert_tracked_tree_clean_before_repo_import()
    snapshot = operator._source_snapshot()
    assert snapshot["tracked_tree_clean"] is True
    assert snapshot["file_sha256"] == {
        "one.py": gate.sha256_bytes(b"one\n")
    }
    assert snapshot["files_sha256"] == gate.sha256_json(snapshot["file_sha256"])
    assert snapshot["python_isolated"] is (sys.flags.isolated == 1)
    assert snapshot["bytecode_disabled"] is sys.dont_write_bytecode

    source.write_text("drift\n", encoding="utf-8")
    with pytest.raises(ValueError, match="before repository import"):
        operator._assert_tracked_tree_clean_before_repo_import()
    with pytest.raises(ValueError, match="clean tracked source"):
        operator._source_snapshot()


def test_replay_cli_resolves_artifact_directory_to_raw(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seen = []

    def replay(path: Path, *, assert_gate: bool):
        seen.append((path, assert_gate))
        return {"passed": True}

    monkeypatch.setattr(operator.gate, "replay_raw", replay)

    assert operator.main(["replay", "--artifact", str(tmp_path), "--assert-gate"]) == 0
    assert seen == [(tmp_path / gate.RAW_NAME, True)]
    assert capsys.readouterr().out == '{"passed":true}\n'


def _subprocess(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        (sys.executable, *arguments),
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_help_reexecs_but_evidence_rejects_nonisolated_initial_python() -> None:
    help_result = _subprocess(str(SCRIPT), "--help")
    assert help_result.returncode == 0
    assert "run-native" in help_result.stdout

    evidence = _subprocess(
        str(SCRIPT),
        "replay",
        "--artifact",
        "/tmp/does-not-exist-a12",
    )
    assert evidence.returncode != 0
    assert "must start as: python -I -B" in evidence.stderr


def test_module_evidence_is_rejected_and_direct_isolated_help_succeeds() -> None:
    module = _subprocess(
        "-m",
        "verification.l1.correctness.batch_invariance_bench",
        "replay",
        "--artifact",
        "/tmp/does-not-exist-a12",
    )
    assert module.returncode != 0
    assert "direct checkout path" in module.stderr

    direct = _subprocess("-I", "-B", str(SCRIPT), "--help")
    assert direct.returncode == 0
    assert "verify" in direct.stdout


def test_import_path_keeps_engine_and_torch_lazy() -> None:
    code = (
        "import sys;"
        f"sys.path.insert(0,{str(REPO_ROOT)!r});"
        "import verification.l1.correctness.batch_invariance_bench;"
        "print(int('torch' in sys.modules),"
        "int(any(name in sys.modules for name in ("
        "'kairyu.engine.kairyu_backend',"
        "'kairyu.engine.core.worker',"
        "'kairyu.engine.core.model_runner'))))"
    )
    result = _subprocess("-I", "-B", "-c", code)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "0 0"
