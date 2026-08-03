"""m13: backend contract, MLA equivalence, fake-flashinfer pins, selector."""

import hashlib
import json
import os
import sys
import types

import pytest
import torch

from kairyu.engine.core.attention import (
    TorchAttentionBackend,
    select_backend,
    select_backend_decision,
    select_backend_name,
)
from kairyu.engine.core.attention.mla_torch import mla_absorbed, mla_decompress, mla_scale
from kairyu.engine.core.attention_selector import (
    attention_backend_execution_identity,
    attention_backend_identity,
    compose_backend_decisions,
)
from kairyu.engine.core.hw_profile import HardwareProfile
from kairyu.engine.core.kv_pool import PagedKVPool

PAGE = 4


def _pool(
    layers=1,
    kv_heads=2,
    head_dim=8,
    dtype=torch.float32,
) -> PagedKVPool:
    return PagedKVPool(layers, 16, PAGE, kv_heads, head_dim, dtype=dtype)


class TestTorchBackend:
    def test_matches_naive_attention(self):
        torch.manual_seed(0)
        pool = _pool()
        seq_len, heads = 10, 4
        keys = torch.randn(seq_len, 2, 8)
        values = torch.randn(seq_len, 2, 8)
        pool.write(0, [0, 1, 2], torch.arange(seq_len), keys, values)
        query = torch.randn(3, heads, 8)  # chunk positions 7..10
        out = TorchAttentionBackend().attend(query, pool, 0, [0, 1, 2], seq_len, 7)
        # naive oracle with repeated kv heads
        k_rep = keys.repeat_interleave(2, dim=1)
        v_rep = values.repeat_interleave(2, dim=1)
        scores = torch.einsum("thd,shd->hts", query, k_rep) * 8**-0.5
        mask = torch.arange(seq_len)[None, :] <= (torch.arange(3)[:, None] + 7)
        scores = scores.masked_fill(~mask[None], float("-inf"))
        naive = torch.einsum("hts,shd->thd", torch.softmax(scores, -1), v_rep)
        assert torch.allclose(out, naive.reshape(3, -1), atol=1e-6)


class TestMlaEquivalence:
    @pytest.mark.parametrize("d_v", [8, 12])  # d_nope != d_v exercises fold direction
    def test_decompress_equals_absorbed_and_oracle(self, d_v):
        torch.manual_seed(1)
        heads, d_nope, d_rope, rank, seq, chunk = 3, 8, 4, 16, 9, 9
        q_nope = torch.randn(chunk, heads, d_nope)
        q_pe = torch.randn(chunk, heads, d_rope)
        c_kv = torch.randn(seq, rank)
        k_pe = torch.randn(seq, d_rope)  # shared single head, post-RoPE
        w_uk = torch.randn(heads, rank, d_nope)
        w_uv = torch.randn(heads, rank, d_v)
        scale = mla_scale(d_nope, d_rope)
        a = mla_decompress(q_nope, q_pe, c_kv, k_pe, w_uk, w_uv, scale)
        b = mla_absorbed(q_nope, q_pe, c_kv, k_pe, w_uk, w_uv, scale)
        assert torch.allclose(a, b, atol=1e-5)
        # naive full-materialization oracle: per-head K = [k_nope || k_pe]
        k_nope = torch.einsum("sr,hrd->hsd", c_kv, w_uk)
        k_full = torch.cat([k_nope, k_pe[None].expand(heads, -1, -1)], dim=-1)
        q_full = torch.cat([q_nope, q_pe], dim=-1)
        scores = torch.einsum("thd,hsd->hts", q_full, k_full) * scale
        mask = torch.arange(seq)[None, :] <= torch.arange(chunk)[:, None]
        scores = scores.masked_fill(~mask[None], float("-inf"))
        values = torch.einsum("sr,hrd->hsd", c_kv, w_uv)
        oracle = torch.einsum("hts,hsd->thd", torch.softmax(scores, -1), values)
        assert torch.allclose(a, oracle, atol=1e-5)

    def test_wrong_scale_breaks_equivalence_gate(self):
        # guards against default-vs-default silently passing (review B1)
        torch.manual_seed(2)
        q_nope = torch.randn(4, 2, 8)
        q_pe = torch.randn(4, 2, 4)
        c_kv = torch.randn(4, 16)
        k_pe = torch.randn(4, 4)
        w_uk = torch.randn(2, 16, 8)
        w_uv = torch.randn(2, 16, 8)
        right = mla_decompress(q_nope, q_pe, c_kv, k_pe, w_uk, w_uv, mla_scale(8, 4))
        wrong = mla_decompress(q_nope, q_pe, c_kv, k_pe, w_uk, w_uv, (16 + 4) ** -0.5)
        assert not torch.allclose(right, wrong, atol=1e-5)


class _FakeWrapper:
    def __init__(
        self,
        workspace,
        kv_layout=None,
        use_tensor_cores=None,
        use_cuda_graph=False,
        paged_kv_indptr_buffer=None,
        paged_kv_indices_buffer=None,
        paged_kv_last_page_len_buffer=None,
    ):
        assert workspace.dtype == torch.uint8 and workspace.numel() >= 1
        self.workspace = workspace
        self.kv_layout = kv_layout
        self.use_tensor_cores = use_tensor_cores
        self.use_cuda_graph = use_cuda_graph
        self.plans: list[dict] = []
        self.runs: list[tuple] = []
        self.run_kwargs: list[dict] = []
        if use_cuda_graph:
            # cudagraph mode: the wrapper OWNS these buffers for its lifetime —
            # plan() copies into them in place and run() reads them, which is
            # the only reason a captured run can see a later plan
            # (flashinfer/decode.py BatchDecodeWithPagedKVCacheWrapper).
            for buffer in (
                paged_kv_indptr_buffer,
                paged_kv_indices_buffer,
                paged_kv_last_page_len_buffer,
            ):
                assert torch.is_tensor(buffer) and buffer.dtype == torch.int32
            assert len(paged_kv_indptr_buffer) == len(paged_kv_last_page_len_buffer) + 1
        self.indptr_buffer = paged_kv_indptr_buffer
        self.indices_buffer = paged_kv_indices_buffer
        self.last_page_len_buffer = paged_kv_last_page_len_buffer

    def plan(self, *args, **kwargs):
        self.plans.append({"args": args, "kwargs": kwargs})
        if self.use_cuda_graph:
            indptr, indices, last_page_len = args[:3]
            assert len(indices) <= len(self.indices_buffer)
            self.indptr_buffer.copy_(indptr)
            self.indices_buffer[: len(indices)].copy_(indices)
            self.last_page_len_buffer.copy_(last_page_len)

    def run(self, query, paged_kv, **kwargs):
        assert self.plans, "run() before plan() (contract violation)"
        self.runs.append((query, paged_kv))
        self.run_kwargs.append(kwargs)
        return query.clone()


@pytest.fixture()
def fake_flashinfer(monkeypatch):
    from kairyu.engine.core.attention import flashinfer_gpu

    module = types.ModuleType("flashinfer")
    module.BatchPrefillWithPagedKVCacheWrapper = _FakeWrapper
    module.BatchDecodeWithPagedKVCacheWrapper = _FakeWrapper
    monkeypatch.setitem(sys.modules, "flashinfer", module)
    monkeypatch.setattr(flashinfer_gpu, "_WORKSPACE_BYTES", 64)
    return module


class TestFlashInferAdapterContract:
    def _backend(self):
        from kairyu.engine.core.attention.flashinfer_gpu import FlashInferBackend

        return FlashInferBackend(device="cpu")

    def test_missing_import_has_gpu_extra_install_action(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "flashinfer", None)

        with pytest.raises(RuntimeError, match=r"uv sync --extra gpu"):
            self._backend()

    def test_wrapper_workspaces_are_zero_initialized(self, fake_flashinfer, monkeypatch):
        real_empty = torch.empty

        def dirty_empty(*args, **kwargs):
            return real_empty(*args, **kwargs).fill_(0xA5)

        monkeypatch.setattr(torch, "empty", dirty_empty)

        backend = self._backend()

        for wrapper in (backend._prefill, backend._decode):
            assert wrapper.workspace.numel() == 64
            assert torch.count_nonzero(wrapper.workspace).item() == 0
        assert backend._prefill.workspace is backend._decode.workspace

    def test_direct_venv_entrypoint_exposes_its_sibling_ninja(
        self, tmp_path, monkeypatch
    ):
        from kairyu.engine.core.attention import flashinfer_gpu

        python_bin = tmp_path / "venv" / "bin"
        python_bin.mkdir(parents=True)
        ninja = python_bin / "ninja"
        ninja.write_text("#!/bin/sh\nexit 0\n")
        ninja.chmod(0o755)
        monkeypatch.setattr(sys, "executable", str(python_bin / "python"))
        monkeypatch.setattr(flashinfer_gpu.shutil, "which", lambda *a, **k: None)
        monkeypatch.setenv("PATH", "/usr/local/sbin")

        flashinfer_gpu._ensure_python_bin_on_path()

        assert os.environ["PATH"].split(os.pathsep)[0] == str(python_bin)

    def test_runtime_preflight_plans_exact_prefill_and_decode_shapes(
        self, fake_flashinfer
    ):
        backend = self._backend()
        pool = _pool(dtype=torch.bfloat16)
        config = types.SimpleNamespace(num_attention_heads=4)

        backend.preflight_runtime(config, pool, q_dtype=torch.bfloat16)

        prefill = backend._prefill.plans[-1]
        decode = backend._decode.plans[-1]
        assert prefill["args"][4:6] == (4, 2)
        assert prefill["kwargs"] == {
            "head_dim_qk": 8,
            "page_size": PAGE,
            "causal": True,
            "q_data_type": torch.bfloat16,
            "kv_data_type": torch.bfloat16,
        }
        assert decode["args"][3:7] == (4, 2, 8, PAGE)
        assert decode["kwargs"] == {
            "q_data_type": torch.bfloat16,
            "kv_data_type": torch.bfloat16,
        }

    def test_runtime_preflight_reports_jit_failure_before_readiness(
        self, fake_flashinfer
    ):
        backend = self._backend()
        backend._prefill.plan = lambda *a, **k: (_ for _ in ()).throw(
            FileNotFoundError("ninja")
        )

        with pytest.raises(
            RuntimeError,
            match="preflight failed before serving became ready.*ninja",
        ):
            backend.preflight_runtime(
                types.SimpleNamespace(num_attention_heads=4),
                _pool(),
                q_dtype=torch.float32,
            )

    def test_prefill_plan_pins_indptr_math_and_kwargs(self, fake_flashinfer):
        backend = self._backend()
        pool = _pool()
        query = torch.randn(5, 4, 8)
        backend.attend(query, pool, 0, [3, 1, 2], seq_len=10, chunk_start=5)
        plan = backend._prefill.plans[-1]
        qo_indptr, kv_indptr, kv_indices, last_page_len = plan["args"][:4]
        assert qo_indptr.tolist() == [0, 5] and qo_indptr.dtype == torch.int32
        assert kv_indptr.tolist() == [0, 3]  # ceil(10/4) pages
        assert kv_indices.tolist() == [3, 1, 2] and kv_indices.dtype == torch.int32
        assert last_page_len.tolist() == [2]  # (10-1) % 4 + 1
        # prefill spelling: head_dim_qk, NOT head_dim (review A1)
        assert plan["kwargs"]["head_dim_qk"] == 8
        assert "head_dim" not in plan["kwargs"]
        assert plan["kwargs"]["causal"] is True
        assert plan["kwargs"]["q_data_type"] == query.dtype  # explicit (A3)
        assert plan["kwargs"]["kv_data_type"] == pool.k.dtype

    def test_decode_path_uses_decode_wrapper(self, fake_flashinfer):
        backend = self._backend()
        pool = _pool()
        query = torch.randn(1, 4, 8)
        backend.attend(query, pool, 0, [0, 1], seq_len=6, chunk_start=5)
        assert backend._decode.plans and not backend._prefill.plans
        assert backend._decode.use_tensor_cores is True

    def test_plan_cached_across_layers(self, fake_flashinfer):
        backend = self._backend()
        pool = _pool(layers=3)
        query = torch.randn(2, 4, 8)
        for layer in range(3):
            backend.attend(query, pool, layer, [0, 1], seq_len=6, chunk_start=4)
        assert len(backend._prefill.plans) == 1  # layer 0 plans; 1..2 reuse (A8)
        assert len(backend._prefill.runs) == 3

    def test_query_and_kv_dtypes_are_part_of_plan_identity(
        self, fake_flashinfer
    ):
        backend = self._backend()
        bf16_query = torch.randn(2, 4, 8, dtype=torch.bfloat16)

        backend.attend(
            bf16_query,
            _pool(dtype=torch.bfloat16),
            0,
            [0],
            seq_len=2,
            chunk_start=0,
        )
        backend.attend(
            bf16_query.to(torch.float16),
            _pool(dtype=torch.bfloat16),
            0,
            [0],
            seq_len=2,
            chunk_start=0,
        )
        backend.attend(
            bf16_query.to(torch.float16),
            _pool(dtype=torch.float8_e4m3fn),
            0,
            [0],
            seq_len=2,
            chunk_start=0,
        )

        assert len(backend._prefill.plans) == 3
        assert (
            backend._prefill.plans[0]["kwargs"]["q_data_type"]
            == torch.bfloat16
        )
        assert (
            backend._prefill.plans[1]["kwargs"]["q_data_type"]
            == torch.float16
        )
        assert backend._prefill.plans[0]["kwargs"]["kv_data_type"] == torch.bfloat16
        assert backend._prefill.plans[1]["kwargs"]["kv_data_type"] == torch.bfloat16
        assert (
            backend._prefill.plans[2]["kwargs"]["kv_data_type"]
            == torch.float8_e4m3fn
        )

    def test_every_list_run_path_forwards_kv_scales(self, fake_flashinfer):
        backend = self._backend()
        fp8_pool = _pool(dtype=torch.float8_e4m3fn)
        query = torch.randn(2, 4, 8, dtype=torch.bfloat16)
        scales = {"k_scale": 1.0, "v_scale": 1.0}

        assert backend.supports_fp8_kv is True
        backend.attend(query, fp8_pool, 0, [0], 2, 0)
        assert backend._prefill.run_kwargs[-1] == scales

        backend.attend(query[:1], fp8_pool, 0, [0], 1, 0)
        assert backend._decode.run_kwargs[-1] == scales

        backend.attend_batched(
            [query[:1], query[:1]],
            fp8_pool,
            0,
            [[0], [1]],
            [1, 1],
            [0, 0],
        )
        assert backend._decode.run_kwargs[-1] == scales

        backend.attend_batched(
            [query, query[:1]],
            fp8_pool,
            0,
            [[0], [1]],
            [2, 1],
            [0, 0],
        )
        assert backend._prefill.run_kwargs[-1] == scales

        backend.attend(
            query,
            _pool(dtype=torch.bfloat16),
            0,
            [0],
            2,
            0,
        )
        assert backend._prefill.run_kwargs[-1] == {
            "k_scale": None,
            "v_scale": None,
        }

    def test_non_tail_chunk_asserts(self, fake_flashinfer):
        backend = self._backend()
        pool = _pool()
        with pytest.raises(AssertionError, match="bottom-right"):
            backend.attend(torch.randn(2, 4, 8), pool, 0, [0, 1], seq_len=8, chunk_start=2)

    def test_run_receives_nhd_tuple(self, fake_flashinfer):
        backend = self._backend()
        pool = _pool(layers=2)
        backend.attend(torch.randn(2, 4, 8), pool, 1, [0], seq_len=2, chunk_start=0)
        _, paged_kv = backend._prefill.runs[-1]
        assert paged_kv[0].shape == (16, PAGE, 2, 8)  # pool.k[layer] NHD slice
        assert torch.equal(paged_kv[0], pool.k[1])

    def test_batched_decode_plans_and_runs_once_with_csr_pages(self, fake_flashinfer):
        backend = self._backend()
        pool = _pool()
        queries = [torch.randn(1, 4, 8), torch.randn(1, 4, 8)]

        contexts = backend.attend_batched(
            queries,
            pool,
            0,
            [[3, 1], [7, 6, 5]],
            [6, 9],
            [5, 8],
        )

        assert len(backend._decode.plans) == 1
        assert len(backend._decode.runs) == 1
        plan = backend._decode.plans[0]
        kv_indptr, kv_indices, last_page_len = plan["args"][:3]
        assert kv_indptr.tolist() == [0, 2, 5] and kv_indptr.dtype == torch.int32
        assert kv_indices.tolist() == [3, 1, 7, 6, 5]
        assert kv_indices.dtype == torch.int32
        assert last_page_len.tolist() == [2, 1]
        assert last_page_len.dtype == torch.int32
        assert plan["args"][3:7] == (4, 2, 8, PAGE)
        assert plan["kwargs"]["q_data_type"] == queries[0].dtype
        assert plan["kwargs"]["kv_data_type"] == pool.k.dtype
        run_query, _ = backend._decode.runs[0]
        assert run_query.shape == (2, 4, 8)
        assert [context.shape for context in contexts] == [(1, 32), (1, 32)]
        assert torch.equal(contexts[0], queries[0].reshape(1, -1))
        assert torch.equal(contexts[1], queries[1].reshape(1, -1))

    def test_batched_decode_plan_cached_across_layers(self, fake_flashinfer):
        backend = self._backend()
        pool = _pool(layers=3)
        queries = [torch.randn(1, 4, 8), torch.randn(1, 4, 8)]
        page_tables = [[3, 1], [7, 6, 5]]
        seq_lens = [6, 9]
        chunk_starts = [5, 8]

        for layer in range(3):
            backend.attend_batched(queries, pool, layer, page_tables, seq_lens, chunk_starts)

        assert len(backend._decode.plans) == 1
        assert len(backend._decode.runs) == 3

    def test_batched_decode_replans_after_prefill(self, fake_flashinfer):
        backend = self._backend()
        pool = _pool()
        backend.attend(torch.randn(2, 4, 8), pool, 0, [3, 1], seq_len=6, chunk_start=4)

        backend.attend_batched(
            [torch.randn(1, 4, 8)],
            pool,
            0,
            [[3, 1]],
            [6],
            [5],
        )

        assert len(backend._prefill.plans) == 1
        assert len(backend._decode.plans) == 1
        assert len(backend._decode.runs) == 1

    def test_single_sequence_batch_equals_single_decode(self, fake_flashinfer):
        backend = self._backend()
        pool = _pool()
        query = torch.randn(1, 4, 8)

        batched = backend.attend_batched([query], pool, 0, [[3, 1]], [6], [5])[0]
        single = backend.attend(query, pool, 0, [3, 1], 6, 5)

        assert torch.equal(batched, single)

    def test_mixed_query_lengths_use_one_batched_prefill_plan(self, fake_flashinfer):
        backend = self._backend()
        pool = _pool()
        queries = [torch.randn(2, 4, 8), torch.randn(1, 4, 8)]

        contexts = backend.attend_batched(
            queries,
            pool,
            0,
            [[3, 1], [7, 6, 5]],
            [6, 9],
            [4, 8],
        )

        assert len(backend._prefill.runs) == 1
        assert len(backend._decode.runs) == 0
        plan = backend._prefill.plans[0]
        qo_indptr, kv_indptr, kv_indices, last_page_len = plan["args"][:4]
        assert qo_indptr.tolist() == [0, 2, 3]
        assert kv_indptr.tolist() == [0, 2, 5]
        assert kv_indices.tolist() == [3, 1, 7, 6, 5]
        assert last_page_len.tolist() == [2, 1]
        run_query, _ = backend._prefill.runs[0]
        assert run_query.shape == (3, 4, 8)
        assert [context.shape for context in contexts] == [(2, 32), (1, 32)]

    @pytest.mark.parametrize(
        ("page_tables", "seq_lens", "chunk_starts"),
        [
            ([[0]], [1, 1], [0, 0]),
            ([[0], [1]], [1], [0, 0]),
            ([[0], [1]], [1, 1], [0]),
        ],
    )
    def test_batched_decode_rejects_parallel_list_length_mismatch(
        self, fake_flashinfer, page_tables, seq_lens, chunk_starts
    ):
        backend = self._backend()
        queries = [torch.randn(1, 4, 8), torch.randn(1, 4, 8)]

        with pytest.raises(ValueError, match="parallel batch inputs"):
            backend.attend_batched(
                queries,
                _pool(),
                0,
                page_tables,
                seq_lens,
                chunk_starts,
            )

        assert backend._decode.plans == []
        assert backend._decode.runs == []

    def test_batched_decode_empty_batch_is_a_noop(self, fake_flashinfer):
        backend = self._backend()

        contexts = backend.attend_batched([], _pool(), 0, [], [], [])

        assert contexts == []
        assert backend._decode.plans == []
        assert backend._decode.runs == []

    def test_batched_decode_non_tail_query_asserts(self, fake_flashinfer):
        backend = self._backend()

        with pytest.raises(AssertionError, match="bottom-right"):
            backend.attend_batched(
                [torch.randn(1, 4, 8)],
                _pool(),
                0,
                [[0, 1]],
                [6],
                [4],
            )


def _forbid_host_sync(monkeypatch) -> None:
    """Make every host-synchronizing tensor read explode.

    `.tolist()` / `.cpu()` / `.item()` are exactly what a CUDA graph capture
    rejects ("Cannot copy between CPU and CUDA tensors during CUDA graph
    capture"), so a decode that survives this is a decode a graph can capture.
    """

    def fail(name):
        def method(self, *args, **kwargs):
            raise AssertionError(f"host synchronization in the capture region: .{name}()")

        return method

    for name in ("tolist", "cpu", "item", "numpy"):
        monkeypatch.setattr(torch.Tensor, name, fail(name))


class TestFlashInferTensorDecode:
    """The m17 D1 tensor contract on the kernel a GPU deployment selects.

    FlashInfer splits into a HOST plan and a device run — `plan()` "cannot be
    used in Cuda Graph" (its own docstring; it copies indptr to the CPU). So
    `plan_decode` is the per-step host phase and `attend_decode` is the
    capture-safe half, and these pin that the halves stay on their own side.
    """

    def _backend(self):
        from kairyu.engine.core.attention.flashinfer_gpu import FlashInferBackend

        return FlashInferBackend(device="cpu")

    def _inputs(self, batch=2, heads=4):
        query = torch.randn(batch, heads, 8)
        page_tables = torch.tensor([[3, 1], [7, 6]][:batch], dtype=torch.int32)
        seq_lens = torch.tensor([6, 5][:batch], dtype=torch.int32)
        return query, page_tables, seq_lens

    def _capturing(self, monkeypatch, value=True):
        from kairyu.engine.core.attention import flashinfer_gpu

        # raising=False so this test class states the property even against an
        # implementation that has no capture guard at all
        monkeypatch.setattr(flashinfer_gpu, "_is_capturing", lambda: value, raising=False)

    def test_the_capture_region_never_touches_the_host(self, fake_flashinfer, monkeypatch):
        """THE regression gate, in the shape a capture really happens: warm up
        eagerly (which plans), then decode with the host off-limits.

        A `.tolist()` or `.cpu()` here is what a real capture rejects with
        "Cannot copy between CPU and CUDA tensors during CUDA graph capture".
        """
        backend = self._backend()
        pool = _pool(layers=2)
        query, page_tables, seq_lens = self._inputs()
        backend.attend_decode(query, pool, 0, page_tables, seq_lens)  # warmup
        wrapper = backend._decode_tensor_wrapper
        plans_before = len(wrapper.plans)

        self._capturing(monkeypatch)
        _forbid_host_sync(monkeypatch)
        contexts = [
            backend.attend_decode(query, pool, layer, page_tables, seq_lens) for layer in range(2)
        ]

        assert len(wrapper.plans) == plans_before  # planning stayed outside
        assert len(wrapper.runs) == 3  # warmup + both captured layers
        assert [context.shape for context in contexts] == [(2, 32), (2, 32)]

    def test_capture_without_a_plan_fails_loudly(self, fake_flashinfer, monkeypatch):
        """Never silently bake capture-time pages into the graph."""
        backend = self._backend()
        query, page_tables, seq_lens = self._inputs()
        self._capturing(monkeypatch)

        with pytest.raises(RuntimeError, match="plan_decode"):
            backend.attend_decode(query, _pool(), 0, page_tables, seq_lens)

    def test_a_plan_for_another_shape_does_not_count_as_this_step_s(
        self, fake_flashinfer, monkeypatch
    ):
        backend = self._backend()
        pool = _pool()
        query, page_tables, seq_lens = self._inputs()
        backend.plan_decode(pool, page_tables, seq_lens, num_qo_heads=4, q_dtype=query.dtype)
        self._capturing(monkeypatch)

        with pytest.raises(RuntimeError, match="plan_decode"):
            backend.attend_decode(
                torch.randn(1, 4, 8),
                pool,
                0,
                page_tables[:1],
                seq_lens[:1],
            )

    def test_plan_decode_refuses_to_run_inside_a_capture(self, fake_flashinfer, monkeypatch):
        backend = self._backend()
        query, page_tables, seq_lens = self._inputs()
        self._capturing(monkeypatch)

        with pytest.raises(RuntimeError, match="cannot run inside a CUDA graph"):
            backend.plan_decode(_pool(), page_tables, seq_lens, num_qo_heads=4, q_dtype=query.dtype)

    def test_the_plan_lands_in_persistent_cudagraph_buffers(self, fake_flashinfer):
        """Plain (non-cudagraph) mode REBINDS its paged buffers on every plan, so
        a captured run would keep reading the capture-time allocation."""
        backend = self._backend()
        pool = _pool()
        query, page_tables, seq_lens = self._inputs()

        backend.plan_decode(pool, page_tables, seq_lens, num_qo_heads=4, q_dtype=query.dtype)
        wrapper = backend._decode_tensor_wrapper
        buffers = (
            wrapper.indptr_buffer,
            wrapper.indices_buffer,
            wrapper.last_page_len_buffer,
        )
        assert wrapper.use_cuda_graph is True
        # ceil(6/4)=2 and ceil(5/4)=2 pages; last page holds 2 and 1 entries
        assert wrapper.indptr_buffer.tolist() == [0, 2, 4]
        assert wrapper.indices_buffer[:4].tolist() == [3, 1, 7, 6]
        assert wrapper.last_page_len_buffer.tolist() == [2, 1]

        page_tables.copy_(torch.tensor([[5, 4], [2, 0]], dtype=torch.int32))
        seq_lens.copy_(torch.tensor([4, 8], dtype=torch.int32))
        backend.plan_decode(pool, page_tables, seq_lens, num_qo_heads=4, q_dtype=query.dtype)

        assert backend._decode_tensor_wrapper is wrapper  # same wrapper, same shape
        assert (
            wrapper.indptr_buffer,
            wrapper.indices_buffer,
            wrapper.last_page_len_buffer,
        ) == buffers  # rewritten IN PLACE, not rebound
        assert wrapper.indptr_buffer.tolist() == [0, 1, 3]
        assert wrapper.indices_buffer[:3].tolist() == [5, 2, 0]
        assert wrapper.last_page_len_buffer.tolist() == [4, 4]

    def test_fast_replay_uses_host_schedule_and_refreshes_persistent_buffers(
        self, fake_flashinfer, monkeypatch
    ):
        from kairyu.engine.core.attention import flashinfer_gpu

        backend = self._backend()
        pool = _pool()
        query, page_tables, seq_lens = self._inputs()
        backend.plan_decode(
            pool,
            page_tables,
            seq_lens,
            num_qo_heads=4,
            q_dtype=query.dtype,
        )
        wrapper = backend._decode_tensor_wrapper
        stock_plans = len(wrapper.plans)
        fast_calls: list[tuple] = []

        def cpu_pack(
            tables, lengths, indptr, indices, last_page_len, *, page_size
        ):
            counts = [
                (int(length) + page_size - 1) // page_size
                for length in lengths.tolist()
            ]
            offsets = [0]
            packed = []
            for row, count in enumerate(counts):
                offsets.append(offsets[-1] + count)
                packed.extend(tables[row, :count].tolist())
            indptr.copy_(torch.tensor(offsets, dtype=torch.int32))
            indices[: len(packed)].copy_(torch.tensor(packed, dtype=torch.int32))
            last_page_len.copy_(
                torch.tensor(
                    [(int(length) - 1) % page_size + 1 for length in lengths],
                    dtype=torch.int32,
                )
            )

        def fast_plan(wrapper_arg, indptr, indices, last_page_len, *args, **kwargs):
            fast_calls.append(
                (wrapper_arg, indptr.clone(), indices.clone(), last_page_len.clone(), args, kwargs)
            )

        monkeypatch.setattr(flashinfer_gpu, "pack_flashinfer_decode_metadata", cpu_pack)
        backend._fast_decode_plan = fast_plan
        page_tables.copy_(torch.tensor([[5, 4], [2, 0]], dtype=torch.int32))
        seq_lens.copy_(torch.tensor([4, 8], dtype=torch.int32))

        backend.plan_decode(
            pool,
            page_tables,
            seq_lens,
            num_qo_heads=4,
            q_dtype=query.dtype,
            replay=True,
            host_seq_lens=(4, 8),
        )

        assert len(wrapper.plans) == stock_plans
        assert wrapper.indptr_buffer.tolist() == [0, 1, 3]
        assert wrapper.indices_buffer[:3].tolist() == [5, 2, 0]
        assert wrapper.last_page_len_buffer.tolist() == [4, 4]
        assert len(fast_calls) == 1
        _, cpu_indptr, _, cpu_last, _, kwargs = fast_calls[0]
        assert cpu_indptr.device.type == cpu_last.device.type == "cpu"
        assert cpu_indptr.tolist() == [0, 1, 3]
        assert cpu_last.tolist() == [4, 4]
        assert kwargs["global_override_indptr_cpu"].tolist() == [0, 1, 3]
        stats = backend.decode_execution_stats()
        assert stats["fast_replay_plans"] == 1
        assert stats["stock_replay_fallbacks"] == 0
        assert stats["replay_fallback_reason"] is None

    def test_missing_fast_api_truthfully_falls_back_to_stock_plan(
        self, fake_flashinfer
    ):
        backend = self._backend()
        pool = _pool()
        query, page_tables, seq_lens = self._inputs()
        backend.plan_decode(
            pool,
            page_tables,
            seq_lens,
            num_qo_heads=4,
            q_dtype=query.dtype,
        )
        wrapper = backend._decode_tensor_wrapper
        page_tables.copy_(torch.tensor([[5, 4], [2, 0]], dtype=torch.int32))
        seq_lens.copy_(torch.tensor([4, 8], dtype=torch.int32))

        backend.plan_decode(
            pool,
            page_tables,
            seq_lens,
            num_qo_heads=4,
            q_dtype=query.dtype,
            replay=True,
            host_seq_lens=(4, 8),
        )

        assert len(wrapper.plans) == 2
        assert wrapper.indptr_buffer.tolist() == [0, 1, 3]
        assert wrapper.indices_buffer[:3].tolist() == [5, 2, 0]
        stats = backend.decode_execution_stats()
        assert stats["fast_replay_plans"] == 0
        assert stats["stock_replay_fallbacks"] == 1
        assert "unavailable" in stats["replay_fallback_reason"]

    @pytest.mark.parametrize(
        ("error_type", "message"),
        (
            (TypeError, "simulated 0.6.x signature drift"),
            (Exception, "simulated Triton compilation failure"),
        ),
        ids=("signature-drift", "triton-compiler"),
    )
    def test_incompatible_fast_api_rewrites_a_partially_packed_buffer_with_stock(
        self, fake_flashinfer, monkeypatch, error_type, message
    ):
        from kairyu.engine.core.attention import flashinfer_gpu

        backend = self._backend()
        pool = _pool()
        query, page_tables, seq_lens = self._inputs()
        backend.plan_decode(
            pool,
            page_tables,
            seq_lens,
            num_qo_heads=4,
            q_dtype=query.dtype,
        )
        wrapper = backend._decode_tensor_wrapper

        def partial_pack(
            _tables, _lengths, indptr, indices, last_page_len, *, page_size
        ):
            assert page_size == PAGE
            indptr.fill_(-91)
            indices.fill_(-92)
            last_page_len.fill_(-93)

        def incompatible_fast(*args, **kwargs):
            raise error_type(message)

        monkeypatch.setattr(
            flashinfer_gpu, "pack_flashinfer_decode_metadata", partial_pack
        )
        backend._fast_decode_plan = incompatible_fast
        page_tables.copy_(torch.tensor([[5, 4], [2, 0]], dtype=torch.int32))
        seq_lens.copy_(torch.tensor([4, 8], dtype=torch.int32))

        backend.plan_decode(
            pool,
            page_tables,
            seq_lens,
            num_qo_heads=4,
            q_dtype=query.dtype,
            replay=True,
            host_seq_lens=(4, 8),
        )

        assert len(wrapper.plans) == 2
        assert wrapper.indptr_buffer.tolist() == [0, 1, 3]
        assert wrapper.indices_buffer[:3].tolist() == [5, 2, 0]
        assert wrapper.last_page_len_buffer.tolist() == [4, 4]
        stats = backend.decode_execution_stats()
        assert stats["fast_replay_plans"] == 0
        assert stats["stock_replay_fallbacks"] == 1
        assert message in stats["replay_fallback_reason"]

    def test_fast_api_is_resolved_from_the_flashinfer_decode_module(
        self, fake_flashinfer, monkeypatch
    ):
        decode_module = types.ModuleType("flashinfer.decode")

        def fast_decode_plan(*args, **kwargs):
            return None

        decode_module.fast_decode_plan = fast_decode_plan
        monkeypatch.setitem(sys.modules, "flashinfer.decode", decode_module)

        backend = self._backend()

        assert backend._fast_decode_plan is fast_decode_plan

    def test_flashinfer_satisfies_the_graph_capture_contract(self, fake_flashinfer):
        """The declaration the runner's fail-fast gate reads (#138 review [P1]).

        Splitting the host plan out of the captured region is exactly what
        earns this: before it, ``attend_decode`` existed but planned inline, so
        a runner that only checked the METHOD would have built a graph path
        whose first capture died on a D2H copy.
        """
        from kairyu.engine.core.attention import graph_capture_gap

        backend = self._backend()
        assert backend.supports_graph_capture is True
        assert graph_capture_gap(backend) is None

    def test_eager_replans_once_per_step_not_once_per_layer(self, fake_flashinfer):
        backend = self._backend()
        pool = _pool(layers=3)
        query, page_tables, seq_lens = self._inputs()

        for layer in range(3):
            backend.attend_decode(query, pool, layer, page_tables, seq_lens)
        wrapper = backend._decode_tensor_wrapper
        assert len(wrapper.plans) == 1  # layer 0 plans, 1..2 reuse it

        for layer in range(3):  # next step: same buffers, new contents
            backend.attend_decode(query, pool, layer, page_tables, seq_lens)

        assert len(wrapper.plans) == 2
        assert len(wrapper.runs) == 6

    def test_the_tensor_path_leaves_the_list_paths_plan_alone(self, fake_flashinfer):
        backend = self._backend()
        pool = _pool()
        query, page_tables, seq_lens = self._inputs()

        backend.attend_decode(query, pool, 0, page_tables, seq_lens)
        backend.attend_batched([query[:1]], pool, 0, [[3, 1]], [6], [5])

        assert len(backend._decode.plans) == 1  # the list path replanned once
        assert len(backend._decode_tensor_wrapper.plans) == 1

    def test_plan_run_dtypes_and_kwargs_are_the_pinned_ones(self, fake_flashinfer):
        backend = self._backend()
        pool = _pool(kv_heads=2, head_dim=8)
        query, page_tables, seq_lens = self._inputs()

        backend.attend_decode(query, pool, 0, page_tables, seq_lens)

        plan = backend._decode_tensor_wrapper.plans[0]
        indptr, indices, last_page_len = plan["args"][:3]
        assert indptr.dtype == indices.dtype == last_page_len.dtype == torch.int32
        assert plan["args"][3:7] == (4, 2, 8, PAGE)
        assert plan["kwargs"]["q_data_type"] == query.dtype
        assert plan["kwargs"]["kv_data_type"] == pool.k.dtype
        run_query, paged_kv = backend._decode_tensor_wrapper.runs[0]
        assert torch.equal(run_query, query)
        assert torch.equal(paged_kv[0], pool.k[0])

    def test_tensor_decode_forwards_fp8_kv_scales(self, fake_flashinfer):
        backend = self._backend()
        pool = _pool(dtype=torch.float8_e4m3fn)
        query, page_tables, seq_lens = self._inputs()

        backend.attend_decode(query, pool, 0, page_tables, seq_lens)

        assert backend._decode_tensor_wrapper.run_kwargs == [
            {"k_scale": 1.0, "v_scale": 1.0}
        ]


class TestSelector:
    def test_cpu_profile_selects_torch(self):
        assert isinstance(select_backend(None), TorchAttentionBackend)
        cpu = HardwareProfile(arch="cpu")
        assert isinstance(select_backend(cpu), TorchAttentionBackend)

    def test_env_override_torch(self, monkeypatch):
        monkeypatch.setenv("KAIRYU_ATTENTION_BACKEND", "torch")
        sm120 = HardwareProfile(arch="cuda", sm=120)
        assert isinstance(select_backend(sm120), TorchAttentionBackend)

    def test_env_override_invalid_fails_loudly(self, monkeypatch):
        monkeypatch.setenv("KAIRYU_ATTENTION_BACKEND", "bogus")
        with pytest.raises(ValueError, match="bogus"):
            select_backend(None)

    def test_gpu_tier_selects_flashinfer(self, fake_flashinfer, monkeypatch):
        from kairyu.engine.core.attention.flashinfer_gpu import FlashInferBackend

        selected = []
        monkeypatch.setattr(
            FlashInferBackend,
            "__init__",
            lambda self, device="cuda": selected.append(device),
        )
        sm120 = HardwareProfile(arch="cuda", sm=120)
        assert isinstance(select_backend(sm120, device="cuda:3"), FlashInferBackend)
        assert selected == ["cuda:3"]

    def test_auto_gpu_falls_back_to_torch_when_flashinfer_cannot_construct(self, monkeypatch):
        from kairyu.engine.core.attention.flashinfer_gpu import FlashInferBackend

        monkeypatch.delenv("KAIRYU_ATTENTION_BACKEND", raising=False)

        def unavailable(self, device="cuda"):
            raise ModuleNotFoundError("flashinfer is not installed")

        monkeypatch.setattr(FlashInferBackend, "__init__", unavailable)
        backend = select_backend(HardwareProfile(arch="cuda", sm=120))

        assert isinstance(backend, TorchAttentionBackend)
        decision = backend.selection_decision
        assert decision.requested == "auto"
        assert decision.resolved == "torch"
        assert decision.components == {
            "prefill": "torch",
            "decode": "torch",
            "kv_mode": "tensor-gather",
        }
        assert "ModuleNotFoundError" in decision.rationale

    def test_explicit_flashinfer_constructor_failure_remains_strict(self, monkeypatch):
        from kairyu.engine.core.attention.flashinfer_gpu import FlashInferBackend

        monkeypatch.setenv("KAIRYU_ATTENTION_BACKEND", "flashinfer")

        def unavailable(self, device="cuda"):
            raise ModuleNotFoundError("flashinfer is not installed")

        monkeypatch.setattr(FlashInferBackend, "__init__", unavailable)
        with pytest.raises(RuntimeError, match=r"uv sync --extra gpu"):
            select_backend(HardwareProfile(arch="cuda", sm=120))


class TestSelectBackendName:
    """select_backend_name mirrors select_backend's precedence but returns the
    NAME without importing/instantiating flashinfer (CPU/macOS safe)."""

    def test_cpu_profile_and_none_are_torch(self):
        assert select_backend_name(None) == "torch"
        assert select_backend_name(HardwareProfile(arch="cpu")) == "torch"

    def test_sm120_is_flashinfer_without_importing(self):
        # no fake_flashinfer / monkeypatch: name resolution never imports the backend
        assert select_backend_name(HardwareProfile(arch="cuda", sm=120)) == "flashinfer"

    def test_env_override_wins(self, monkeypatch):
        monkeypatch.setenv("KAIRYU_ATTENTION_BACKEND", "torch")
        assert select_backend_name(HardwareProfile(arch="cuda", sm=120)) == "torch"
        monkeypatch.setenv("KAIRYU_ATTENTION_BACKEND", "flashinfer")
        assert select_backend_name(HardwareProfile(arch="cpu")) == "flashinfer"
        monkeypatch.setenv("KAIRYU_ATTENTION_BACKEND", "flashattention3")
        assert select_backend_name(HardwareProfile(arch="cuda", sm=90)) == "flashattention3"
        monkeypatch.setenv("KAIRYU_ATTENTION_BACKEND", "flashattention4")
        assert select_backend_name(HardwareProfile(arch="cuda", sm=120)) == "flashattention4"

    def test_explicit_auto_uses_the_profile_policy(self, monkeypatch):
        monkeypatch.setenv("KAIRYU_ATTENTION_BACKEND", "auto")
        assert select_backend_name(HardwareProfile(arch="cpu")) == "torch"
        assert select_backend_name(HardwareProfile(arch="cuda", sm=120)) == "flashinfer"

    def test_invalid_env_fails_loudly(self, monkeypatch):
        monkeypatch.setenv("KAIRYU_ATTENTION_BACKEND", "bogus")
        with pytest.raises(ValueError, match="bogus"):
            select_backend_name(None)


class TestBackendDecision:
    def test_sm120_fa4_materialization_and_decode_delegate_are_visible(self, monkeypatch):
        monkeypatch.setenv("KAIRYU_ATTENTION_BACKEND", "flashattention4")

        decision = select_backend_decision(HardwareProfile(arch="cuda", sm=120))

        assert decision.requested == decision.resolved == "flashattention4"
        assert decision.source == "env"
        assert decision.components == {
            "prefill": "flashattention4",
            "decode": "flashinfer",
            "kv_mode": "paged-materialized",
        }

    @pytest.mark.parametrize("sm", [90, 100, 110])
    def test_fa4_supported_direct_paged_profiles_are_reported(self, monkeypatch, sm):
        monkeypatch.setenv("KAIRYU_ATTENTION_BACKEND", "flashattention4")
        decision = select_backend_decision(HardwareProfile(arch="cuda", sm=sm))
        assert decision.components["kv_mode"] == "paged-direct"
        assert decision.architecture["sm"] == sm
        assert decision.architecture["arch"] == "cuda"

    def test_tp_identity_is_canonical_and_excludes_free_form_rationale(self, monkeypatch):
        monkeypatch.setenv("KAIRYU_ATTENTION_BACKEND", "flashattention4")
        decision = select_backend_decision(
            HardwareProfile(
                arch="cuda",
                device_name="NVIDIA H100",
                sm=90,
            )
        )
        equivalent = type(decision)(
            requested=decision.requested,
            resolved=decision.resolved,
            source=decision.source,
            components=dict(reversed(tuple(decision.components.items()))),
            rationale="diagnostic wording changed",
            architecture=dict(reversed(tuple(decision.architecture.items()))),
        )

        assert attention_backend_identity(decision) == attention_backend_identity(equivalent)

    def test_execution_identity_binds_the_loaded_flashinfer_version(self, monkeypatch):
        from kairyu.engine.core import attention_selector

        monkeypatch.setenv("KAIRYU_ATTENTION_BACKEND", "flashinfer")
        record = "flashinfer_aot.so,sha256=test,123\n"
        distribution = types.SimpleNamespace(
            version="0.6.4+aot",
            read_text=lambda name: record if name == "RECORD" else None,
        )
        monkeypatch.setattr(
            attention_selector.metadata,
            "distribution",
            lambda name: distribution,
        )
        decision = select_backend_decision(HardwareProfile(arch="cuda", sm=120))
        backend = types.SimpleNamespace(
            _flashinfer=types.SimpleNamespace(__version__="0.6.4")
        )

        identity = json.loads(attention_backend_execution_identity(decision, backend))

        assert identity["component_versions"] == {
            "decode": "0.6.4",
            "prefill": "0.6.4",
        }
        assert identity["component_builds"] == {
            "flashinfer-jit-cache": {
                "installed": True,
                "version": "0.6.4+aot",
                "record_sha256": hashlib.sha256(record.encode()).hexdigest(),
            }
        }

    def test_execution_identity_binds_the_torch_version(self):
        decision = select_backend_decision(HardwareProfile(arch="cpu"))
        backend = TorchAttentionBackend()

        identity = json.loads(attention_backend_execution_identity(decision, backend))

        assert identity["component_versions"] == {
            "decode": str(torch.__version__),
            "prefill": str(torch.__version__),
        }

    def test_execution_identity_binds_fa_and_delegated_decode_versions(
        self,
        monkeypatch,
    ):
        monkeypatch.setenv("KAIRYU_ATTENTION_BACKEND", "flashattention4")
        decision = select_backend_decision(HardwareProfile(arch="cuda", sm=120))
        backend = types.SimpleNamespace(
            component_versions={"prefill": "4.0.0b24", "decode": "unknown"},
            _decode_backend=types.SimpleNamespace(
                _flashinfer=types.SimpleNamespace(__version__="0.6.4")
            ),
        )

        identity = json.loads(attention_backend_execution_identity(decision, backend))

        assert identity["component_versions"] == {
            "decode": "0.6.4",
            "prefill": "4.0.0b24",
        }

    def test_execution_identity_rejects_an_unversioned_accelerator_backend(
        self,
        monkeypatch,
    ):
        monkeypatch.setenv("KAIRYU_ATTENTION_BACKEND", "flashinfer")
        decision = select_backend_decision(HardwareProfile(arch="cuda", sm=120))
        backend = types.SimpleNamespace(_flashinfer=types.SimpleNamespace())

        with pytest.raises(RuntimeError, match="exact runtime __version__"):
            attention_backend_execution_identity(decision, backend)

    def test_pd_role_decision_reports_both_architectures(self, monkeypatch):
        monkeypatch.setenv("KAIRYU_ATTENTION_BACKEND", "flashattention4")
        prefill = select_backend_decision(HardwareProfile(arch="cuda", device_name="H100", sm=90))
        decode = select_backend_decision(
            HardwareProfile(arch="cuda", device_name="RTX PRO 6000", sm=120)
        )

        decision = compose_backend_decisions(prefill, decode)

        assert decision.resolved == "flashattention4"
        assert decision.components["kv_mode"] == ("prefill:paged-direct;decode:paged-materialized")
        assert decision.architecture["prefill"]["sm"] == 90
        assert decision.architecture["decode"]["sm"] == 120

    def test_automatic_selection_reports_stable_fallback_reason(self):
        decision = select_backend_decision(HardwareProfile(arch="cuda", sm=120))
        assert decision.requested == "auto"
        assert decision.resolved == "flashinfer"
        assert decision.source == "hw_profile"
        assert "stable" in decision.rationale
