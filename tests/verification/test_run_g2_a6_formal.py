"""CPU contracts for the committed G2 A6 formal operator."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pytest

from verification.l1.performance import g2_a6_vllm_bench as a6
from verification.l1.performance import run_g2_a6_formal as runner


def test_vllm_package_pin_matches_the_immutable_cuda_image() -> None:
    args = runner.build_parser().parse_args([])

    assert a6.VLLM_VERSION == "0.26.0+cu129"
    assert args.vllm_version == a6.VLLM_VERSION
    assert args.vllm_image == a6.VLLM_IMAGE_REF


@pytest.mark.parametrize(
    "status",
    [
        b"",
        b"?? .claude/worktrees/\0",
        b"?? .claude/worktrees/agent/file.py\0",
        (
            b"?? .claude/worktrees/agent/a.py\0"
            b"?? .claude/worktrees/other/b.py\0"
        ),
    ],
)
def test_source_status_allows_only_protected_untracked_worktrees(
    status: bytes,
) -> None:
    assert runner.unexpected_source_status_entries(status) == []


@pytest.mark.parametrize(
    "status",
    [
        b" M verification/l1/performance/g2_a6_vllm_bench.py\0",
        b"?? .claude/worktrees2/file.py\0",
        b"?? other/file.py\0",
        b"R  old.py\0new.py\0",
        b"?? .claude/worktrees/agent/../escape.py\0",
    ],
)
def test_source_status_rejects_every_other_entry(status: bytes) -> None:
    assert runner.unexpected_source_status_entries(status)


def test_source_status_rejects_non_nul_framing() -> None:
    with pytest.raises(runner.FormalRunError, match="invalid framing"):
        runner.unexpected_source_status_entries(
            b"?? .claude/worktrees/agent/file.py"
        )


def _args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        port=18080,
        model_volume="model-volume",
        repo=tmp_path,
        kairyu_image="kairyu:test",
        vllm_image=a6.VLLM_IMAGE_REF,
        vllm_version=a6.VLLM_VERSION,
        kairyu_cuda_version=a6.KAIRYU_CUDA_VERSION,
        kairyu_nccl_version=a6.KAIRYU_NCCL_VERSION,
        vllm_cuda_version=a6.VLLM_CUDA_VERSION,
        vllm_nccl_version=a6.VLLM_NCCL_VERSION,
    )


def test_formal_commands_pin_http_runtime_scheduler_and_launch_modes(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    kairyu = runner.Scenario(
        kind="binding",
        tp=4,
        arm="kairyu",
        workload="sharegpt",
        repeat=0,
    )
    stock = runner.Scenario(
        kind="binding",
        tp=4,
        arm="vllm-stock",
        workload="sharegpt",
        repeat=0,
    )
    kairyu_command = runner.kairyu_docker_command(
        args=args,
        scenario=kairyu,
        name="g2a6-test-k",
        config_path=tmp_path / "config.yaml",
        image_id=f"sha256:{'1' * 64}",
    )
    stock_command = runner.vllm_docker_command(
        a6=a6,
        args=args,
        scenario=stock,
        name="g2a6-test-v",
        image_id=a6.VLLM_IMAGE_ID,
    )

    assert kairyu_command[-2:] == ["serve", "/etc/kairyu/config.yaml"]
    assert f"sha256:{'1' * 64}" in kairyu_command
    assert args.kairyu_image not in kairyu_command
    assert "KAIRYU_ATTENTION_BACKEND=flashinfer" in kairyu_command
    expected_stock_argv = a6.expected_engine_argv(
        arm="vllm-stock",
        tp=4,
    )
    assert a6.VLLM_IMAGE_ID in stock_command
    assert args.vllm_image not in stock_command
    assert stock_command[-len(expected_stock_argv) :] == expected_stock_argv
    for flag in (
        "--async-scheduling",
        "--disable-uvicorn-access-log",
        "--no-enable-log-requests",
        "--disable-custom-all-reduce",
        "--attention-backend",
        "--distributed-executor-backend",
        "--compilation-config",
    ):
        assert flag in stock_command


def test_arm_config_uses_equal_usable_cache_and_truthful_reservations() -> None:
    kairyu = runner.Scenario(
        kind="binding",
        tp=8,
        arm="kairyu",
        workload="sharegpt",
        repeat=0,
    )
    stock = runner.Scenario(
        kind="binding",
        tp=8,
        arm="vllm-stock",
        workload="sharegpt",
        repeat=0,
    )
    kairyu_config = runner.configuration(a6, kairyu)
    stock_config = runner.configuration(a6, stock)

    assert kairyu_config["allocated_cache_blocks"] == 8193
    assert kairyu_config["reserved_graph_scratch_blocks"] == 1
    assert kairyu_config["reserved_null_blocks"] == 0
    assert stock_config["allocated_cache_blocks"] == 8193
    assert stock_config["reserved_graph_scratch_blocks"] == 0
    assert stock_config["reserved_null_blocks"] == 1
    assert (
        kairyu_config["usable_cache_blocks"]
        == stock_config["usable_cache_blocks"]
        == 8192
    )
    assert kairyu_config["pipeline_depth"] == 5
    assert stock_config["async_scheduling"] is True
    assert kairyu_config["custom_all_reduce"] is False
    assert stock_config["custom_all_reduce"] is False


def test_kairyu_template_pins_context_cache_pipeline_and_access_log(
    tmp_path: Path,
) -> None:
    del tmp_path
    template = (
        a6._REPO_ROOT
        / "deploy/verification/qwen3-32b-multi-gpu/g2-a6-kairyu.template.yaml"
    ).read_text(encoding="utf-8")
    rendered = runner.render_kairyu_config(a6, template, 8)

    for marker in (
        "access_log: false",
        "admission_wait_timeout_s: 600",
        "num_pages: 8193",
        "max_model_len: 8192",
        "pipeline_depth: 5",
    ):
        assert marker in rendered
    assert "__TENSOR_PARALLEL_SIZE__" not in rendered
    source = a6.kairyu_configuration_source(rendered, tp=8)
    assert source["value"]["parsed"] == a6.expected_kairyu_config(8)


def test_runtime_probe_requires_http_acceleration_packages() -> None:
    code = runner.runtime_probe_code("vllm-stock")
    for distribution in (
        "uvloop",
        "httptools",
        "uvicorn",
        "flashinfer",
        "vllm",
    ):
        assert distribution in code


def test_dry_checkpoint_attestation_binds_all_pinned_files() -> None:
    descriptor = runner.dry_checkpoint_attestation(a6)
    assert a6._checkpoint_attestation_valid(descriptor)
    checkpoint = descriptor["value"]["checkpoint"]
    assert len(checkpoint["weights"]) == 17
    assert checkpoint["weights_sha256"] == a6.MODEL_WEIGHTS_ROLLUP


def _retained_runtime_identities() -> dict[str, dict[str, object]]:
    return {
        "kairyu": {
            "cuda_version": a6.KAIRYU_CUDA_VERSION,
            "nccl_version": a6.KAIRYU_NCCL_VERSION,
            "package_versions": {
                "torch": "2.9.0",
                "flashinfer": "0.6.6",
                "uvloop": a6.UVLOOP_VERSION,
                "httptools": a6.HTTPTOOLS_VERSION,
                "uvicorn": a6.KAIRYU_UVICORN_VERSION,
                "kairyu": "0.1.0",
            },
        },
        "vllm-stock": {
            "cuda_version": a6.VLLM_CUDA_VERSION,
            "nccl_version": a6.VLLM_NCCL_VERSION,
            "package_versions": {
                "torch": "2.9.0",
                "flashinfer": "0.6.6",
                "uvloop": a6.UVLOOP_VERSION,
                "httptools": a6.HTTPTOOLS_VERSION,
                "uvicorn": a6.VLLM_UVICORN_VERSION,
                "vllm": a6.VLLM_VERSION,
            },
        },
    }


def test_dry_resume_reuses_retained_live_identities(tmp_path: Path) -> None:
    args = _args(tmp_path)
    checkpoint = runner.dry_checkpoint_attestation(a6)
    runtime = _retained_runtime_identities()
    retained = {
        "checkpoint": checkpoint,
        "runtime": runtime,
    }

    actual_checkpoint, actual_runtime = runner.dry_attestation_identities(
        a6=a6,
        args=args,
        version="0.1.0",
        resume_state=retained,
    )

    assert actual_checkpoint == checkpoint
    assert actual_runtime == runtime
    assert all(
        package != "dry-attested"
        for identity in actual_runtime.values()
        for package in identity["package_versions"].values()
    )


def test_dry_fixture_satisfies_the_full_32_plus_20_live_schema(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    env_path = a6._REPO_ROOT / "bench/results/env-2026-07-30.json"
    parsed_env = runner.parse_environment(env_path)
    inventory = runner.dry_gpu_inventory(parsed_env)
    checkpoint = runner.dry_checkpoint_attestation(a6)
    runtime = _retained_runtime_identities()
    state = {
        "repo_commit": "1" * 40,
        "images": {
            "kairyu": {
                "ref": args.kairyu_image,
                "id": f"sha256:{'2' * 64}",
            },
            "vllm-stock": {
                "ref": a6.VLLM_IMAGE_REF,
                "id": a6.VLLM_IMAGE_ID,
            },
        },
        "versions": {
            "kairyu": "0.1.0",
            "vllm-stock": a6.VLLM_VERSION,
        },
        "checkpoint": checkpoint,
        "runtime": runtime,
    }
    template = (
        a6._REPO_ROOT
        / "deploy/verification/qwen3-32b-multi-gpu/g2-a6-kairyu.template.yaml"
    ).read_text(encoding="utf-8")

    runner.dry_validate_provenance(
        a6=a6,
        binding=runner.binding_plan(a6),
        sweep=runner.sweep_plan(a6),
        state=state,
        parsed_env=parsed_env,
        inventory=inventory,
        repo=a6._REPO_ROOT,
        env_path=env_path,
        kairyu_template=template,
        model_volume=args.model_volume,
        published_host_port=args.port,
    )


def test_live_container_attestation_binds_image_mounts_and_imports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path)
    (tmp_path / "kairyu").mkdir()
    template = (
        a6._REPO_ROOT
        / "deploy/verification/qwen3-32b-multi-gpu/g2-a6-kairyu.template.yaml"
    ).read_text(encoding="utf-8")
    rendered = runner.render_kairyu_config(a6, template, 4)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(rendered, encoding="utf-8")
    configuration_source = a6.kairyu_configuration_source(rendered, tp=4)
    image_id = f"sha256:{'2' * 64}"
    expected = a6.expected_container_launch(arm="kairyu", tp=4)
    inspected = [
        {
            "Id": "3" * 64,
            "Image": image_id,
            "Config": {
                "Image": image_id,
                "Entrypoint": expected["entrypoint"],
                "Cmd": expected["argv"],
                "Env": [
                    "PYTHONDONTWRITEBYTECODE=1",
                    "KAIRYU_ATTENTION_BACKEND=flashinfer",
                ],
                "WorkingDir": "/app",
            },
            "HostConfig": {
                "IpcMode": "host",
                "Ulimits": [{"Name": "memlock", "Soft": -1, "Hard": -1}],
                "PortBindings": {
                    "8000/tcp": [
                        {"HostIp": "127.0.0.1", "HostPort": "18080"}
                    ]
                },
                "DeviceRequests": [
                    {
                        "Driver": "nvidia",
                        "Capabilities": [["gpu"]],
                        "DeviceIDs": ["0", "1", "2", "3"],
                    }
                ],
            },
            "Mounts": [
                {
                    "Type": "volume",
                    "Name": args.model_volume,
                    "Source": "/var/lib/docker/volumes/model-volume/_data",
                    "Destination": "/models",
                    "RW": False,
                },
                {
                    "Type": "bind",
                    "Source": str(tmp_path.resolve()),
                    "Destination": "/workspace",
                    "RW": False,
                },
                {
                    "Type": "bind",
                    "Source": str((tmp_path / "kairyu").resolve()),
                    "Destination": "/app/kairyu",
                    "RW": False,
                },
                {
                    "Type": "bind",
                    "Source": str(config_path.resolve()),
                    "Destination": "/etc/kairyu/config.yaml",
                    "RW": False,
                },
            ],
        }
    ]
    imported = {
        "package": {
            "file": "/app/kairyu/__init__.py",
            "sha256": "4" * 64,
        },
        "engine": {
            "file": "/app/kairyu/engine/engine_loop.py",
            "sha256": "5" * 64,
        },
    }
    config_probe = {
        "sha256": configuration_source["value"]["text_sha256"],
        "text": rendered,
    }

    def fake_run(command, **kwargs):
        del kwargs
        if tuple(command[:2]) == ("docker", "inspect"):
            stdout = json.dumps(inspected)
        elif "/etc/kairyu/config.yaml" in str(command[-1]):
            stdout = json.dumps(config_probe)
        else:
            stdout = json.dumps(imported)
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")

    monkeypatch.setattr(runner, "run_command", fake_run)
    descriptor = runner.container_launch_attestation(
        a6=a6,
        args=args,
        name="g2a6-fixture",
        arm="kairyu",
        tp=4,
        image_id=image_id,
        configuration_source=configuration_source,
        config_path=config_path,
    )

    assert a6._container_launch_valid(
        descriptor,
        arm="kairyu",
        tp=4,
        image_id=image_id,
        configuration_source=configuration_source,
    )
    assert descriptor["value"]["container_id"] == "3" * 64
    assert descriptor["value"]["image_id"] == image_id


def test_vllm_startup_attestation_uses_real_v026_message_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_message = (
        "Initializing a V1 LLM engine (v0.26.0) with config: "
        f"max_seq_len={a6.MAX_MODEL_LEN}, tensor_parallel_size=4, "
        "disable_custom_all_reduce=True, enable_prefix_caching=True, "
        "enable_chunked_prefill=True, compilation_config={"
        "'mode': <CompilationMode.VLLM_COMPILE: 3>, "
        "'cudagraph_mode': <CUDAGraphMode.FULL_AND_PIECEWISE: (2, 1)>, "
        f"'cudagraph_capture_sizes': {list(a6.VLLM_CUDA_GRAPH_CAPTURE_SIZES)}"
        "}"
    )
    log = "\n".join(
        (
            "INFO [vllm/config/vllm.py] Asynchronous scheduling is enabled.",
            f"INFO [vllm/v1/engine/core.py] {config_message}",
            (
                "INFO [vllm/platforms/cuda.py] "
                f"{a6.VLLM_ATTENTION_BACKEND_LOG_MARKER}"
            ),
            (
                "INFO [vllm/v1/attention/backends/flash_attn.py] "
                f"Using FlashAttention version {a6.VLLM_FLASH_ATTN_VERSION}"
            ),
        )
    )

    def fake_run(command, **kwargs):
        del kwargs
        return subprocess.CompletedProcess(command, 0, stdout=log, stderr="")

    monkeypatch.setattr(runner, "run_command", fake_run)
    descriptor = runner.vllm_startup_log_attestation(
        a6=a6,
        name="g2a6-vllm-fixture",
        tp=4,
    )

    assert a6._vllm_startup_log_valid(descriptor, tp=4)
    assert descriptor["value"]["engine_config_message"] == config_message
    assert descriptor["value"]["async_scheduling_messages"] == [
        "Asynchronous scheduling is enabled."
    ]
    assert descriptor["value"]["attention_version_messages"] == [
        f"Using FlashAttention version {a6.VLLM_FLASH_ATTN_VERSION}"
    ]
