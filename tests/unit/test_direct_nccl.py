"""CPU-only contract tests for the attention-DP direct NCCL transport."""

from __future__ import annotations

import ctypes
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from kairyu.engine.core import direct_nccl
from kairyu.engine.core.dist_comm import TorchDistCommunicator
from kairyu.models.moe_parallel import EpMoeBlock


class _Symbol:
    def __init__(self, callback=None) -> None:
        self.callback = callback or (lambda *_args: 0)
        self.calls: list[tuple[object, ...]] = []
        self.restype = None
        self.argtypes = None

    def __call__(self, *args):
        self.calls.append(args)
        return self.callback(*args)


def _fake_cdll() -> SimpleNamespace:
    symbols = {
        declaration.name: _Symbol()
        for declaration in direct_nccl._FUNCTIONS
    }

    def get_version(pointer) -> int:
        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_int)).contents.value = 22907
        return 0

    def get_unique_id(pointer) -> int:
        unique_id = ctypes.cast(
            pointer,
            ctypes.POINTER(direct_nccl._NcclUniqueId),
        ).contents
        for index in range(direct_nccl._UNIQUE_ID_BYTES):
            unique_id.internal[index] = (index % 256) - 128
        return 0

    def init_rank(pointer, _world_size, _unique_id, _rank) -> int:
        ctypes.cast(pointer, ctypes.POINTER(ctypes.c_void_p)).contents.value = 0xCAFE
        return 0

    symbols["ncclGetErrorString"] = _Symbol(lambda _result: b"mock failure")
    symbols["ncclGetVersion"] = _Symbol(get_version)
    symbols["ncclGetUniqueId"] = _Symbol(get_unique_id)
    symbols["ncclCommInitRank"] = _Symbol(init_rank)
    return SimpleNamespace(**symbols)


def test_nccl_binding_declares_checked_minimal_abi() -> None:
    direct_nccl.NcclLibrary._cache.clear()
    cdll = _fake_cdll()
    binding = direct_nccl.NcclLibrary(
        "/mock/libnccl.so.2",
        loader=lambda _path: cdll,
    )

    assert binding.version() == "2.29.7"
    unique_id = binding.unique_id()
    assert len(unique_id) == 128
    communicator = binding.init_rank(4, unique_id, 2)
    assert communicator.value == 0xCAFE
    binding.group_start()
    binding.all_gather(11, 22, 32, torch.uint8, communicator, 33)
    binding.reduce_scatter(44, 55, 64, torch.bfloat16, communicator, 66)
    binding.broadcast(77, 88, 12, torch.int32, 3, communicator, 99)
    binding.reduce(111, 122, 14, torch.float32, 2, communicator, 133)
    binding.group_end()
    binding.destroy(communicator)

    assert cdll.ncclAllGather.calls[0][2] == 32
    assert cdll.ncclAllGather.calls[0][3] == 1
    assert cdll.ncclReduceScatter.calls[0][2] == 64
    assert cdll.ncclReduceScatter.calls[0][3] == 9
    assert cdll.ncclReduceScatter.calls[0][4] == 0
    assert cdll.ncclBroadcast.calls[0][2:5] == (12, 2, 3)
    assert cdll.ncclReduce.calls[0][2:6] == (14, 7, 0, 2)
    for declaration in direct_nccl._FUNCTIONS:
        symbol = getattr(cdll, declaration.name)
        assert symbol.restype is declaration.restype
        assert symbol.argtypes == list(declaration.argtypes)


def test_nccl_binding_reports_operation_and_library_error() -> None:
    direct_nccl.NcclLibrary._cache.clear()
    cdll = _fake_cdll()
    cdll.ncclGroupStart.callback = lambda: 5
    binding = direct_nccl.NcclLibrary(
        "/mock/error-libnccl.so.2",
        loader=lambda _path: cdll,
    )

    with pytest.raises(RuntimeError, match="ncclGroupStart.*mock failure"):
        binding.group_start()


class _ControlComm:
    rank = 0
    world_size = 4

    @staticmethod
    def all_gather(value):
        return tuple({**value, "rank": rank} for rank in range(4))

    @staticmethod
    def broadcast(value, src: int):
        assert src == 0
        return value


class _FakeLibrary:
    path = "/mock/libnccl.so.2"

    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []
        self.destroyed = False
        self.aborted = False

    @staticmethod
    def version() -> str:
        return "2.29.7"

    @staticmethod
    def unique_id() -> bytes:
        return bytes(range(128))

    def init_rank(self, world_size: int, unique_id: bytes, rank: int):
        self.calls.append(("init", world_size, unique_id, rank))
        return ctypes.c_void_p(123)

    def group_start(self) -> None:
        self.calls.append(("group_start",))

    def group_end(self) -> None:
        self.calls.append(("group_end",))

    def all_gather(self, *args) -> None:
        self.calls.append(("all_gather", *args))

    def reduce_scatter(self, *args) -> None:
        self.calls.append(("reduce_scatter", *args))

    def broadcast(self, *args) -> None:
        self.calls.append(("broadcast", *args))

    def reduce(self, *args) -> None:
        self.calls.append(("reduce", *args))

    def destroy(self, _communicator) -> None:
        self.destroyed = True

    def abort(self, _communicator) -> None:
        self.aborted = True


def test_direct_nccl_creation_is_collective_and_reports_active_only_after_init(
    monkeypatch,
) -> None:
    library = _FakeLibrary()
    monkeypatch.setattr(direct_nccl.torch.cuda, "device", lambda _device: nullcontext())

    communicator, metadata = direct_nccl.DirectNcclCommunicator.try_create(
        _ControlComm(),
        "cuda:0",
        library_factory=lambda _path: library,
    )

    assert communicator is not None
    assert metadata == {
        "selected_backend": "direct_nccl_ctypes",
        "fallback_backend": "torch.distributed:nccl",
        "direct_nccl_active": True,
        "direct_nccl_library": "/mock/libnccl.so.2",
        "direct_nccl_version": "2.29.7",
        "selection_reason": (
            "attention-DP CUDA/NCCL hot path initialized on every rank"
        ),
    }
    assert library.calls[0][:2] == ("init", 4)
    communicator.close()
    assert library.destroyed is True


def test_direct_nccl_library_failure_selects_torch_fallback_with_reason() -> None:
    def unavailable(_path):
        raise OSError("library absent")

    communicator, metadata = direct_nccl.DirectNcclCommunicator.try_create(
        _ControlComm(),
        "cuda:0",
        library_factory=unavailable,
    )

    assert communicator is None
    assert metadata["selected_backend"] == "torch.distributed:nccl"
    assert metadata["direct_nccl_active"] is False
    assert metadata["direct_nccl_library"] is None
    assert metadata["direct_nccl_version"] is None
    assert "rank 0: OSError: library absent" in metadata["selection_reason"]


def test_direct_collectives_group_gathers_and_use_reduce_scatter_receive_count() -> None:
    library = _FakeLibrary()
    communicator = direct_nccl.DirectNcclCommunicator(
        rank=0,
        world_size=4,
        device=torch.device("cpu"),
        library=library,
        communicator=ctypes.c_void_p(123),
        version="2.29.7",
        stream_resolver=lambda _device: SimpleNamespace(cuda_stream=77),
    )
    first = torch.arange(6, dtype=torch.uint8).reshape(2, 3)
    second = torch.arange(4, dtype=torch.int32).reshape(2, 2)

    gathered = communicator.all_gather_many((first, second))
    reduced = communicator.reduce_scatter(torch.ones(8, 5, dtype=torch.bfloat16))

    assert tuple(tensor.shape for tensor in gathered) == ((8, 3), (8, 2))
    assert tuple(reduced.shape) == (2, 5)
    operations = [call[0] for call in library.calls]
    assert operations == [
        "group_start",
        "all_gather",
        "all_gather",
        "group_end",
        "reduce_scatter",
    ]
    reduce_call = library.calls[-1]
    assert reduce_call[3] == reduced.numel()
    assert reduce_call[-1] == 77


def test_direct_collective_failure_aborts_instead_of_reusing_broken_sequence() -> None:
    class _FailingLibrary(_FakeLibrary):
        def all_gather(self, *args) -> None:
            super().all_gather(*args)
            raise RuntimeError("injected collective failure")

    library = _FailingLibrary()
    communicator = direct_nccl.DirectNcclCommunicator(
        rank=0,
        world_size=4,
        device=torch.device("cpu"),
        library=library,
        communicator=ctypes.c_void_p(123),
        version="2.29.7",
        stream_resolver=lambda _device: SimpleNamespace(cuda_stream=77),
    )

    with pytest.raises(RuntimeError, match="injected collective failure"):
        communicator.all_gather_many((torch.ones(2, 2),))
    assert library.aborted is True
    assert communicator.active is False
    with pytest.raises(RuntimeError, match="closed"):
        communicator.reduce_scatter(torch.ones(4, 2))


def test_direct_close_aborts_if_orderly_destroy_fails() -> None:
    class _DestroyFailingLibrary(_FakeLibrary):
        def destroy(self, _communicator) -> None:
            raise RuntimeError("injected destroy failure")

    library = _DestroyFailingLibrary()
    communicator = direct_nccl.DirectNcclCommunicator(
        rank=0,
        world_size=4,
        device=torch.device("cpu"),
        library=library,
        communicator=ctypes.c_void_p(123),
        version="2.29.7",
        stream_resolver=lambda _device: SimpleNamespace(cuda_stream=77),
    )

    with pytest.raises(RuntimeError, match="injected destroy failure"):
        communicator.close()
    assert communicator.active is False
    assert library.aborted is True


def test_torch_wrapper_never_reports_an_aborted_direct_transport_as_active(
    monkeypatch,
) -> None:
    direct = SimpleNamespace(active=False)
    communicator = object.__new__(TorchDistCommunicator)
    communicator._group = object()
    communicator._attention_dp_direct_nccl = direct
    communicator._attention_dp_collective_metadata = {
        "selected_backend": "direct_nccl_ctypes",
        "fallback_backend": "torch.distributed:nccl",
        "direct_nccl_active": True,
        "direct_nccl_library": "libnccl.so.2",
        "direct_nccl_version": "2.29.7",
        "selection_reason": "initialized",
    }
    monkeypatch.setattr(
        "kairyu.engine.core.dist_comm.dist.get_backend",
        lambda _group: "nccl",
    )

    assert communicator.attention_dp_direct_nccl_active is False
    metadata = communicator.attention_dp_collective_metadata()
    assert metadata["selected_backend"] == "unavailable"
    assert metadata["direct_nccl_active"] is False
    assert "fallback is unsafe" in metadata["selection_reason"]


def test_direct_collective_workspaces_grow_and_reuse_on_the_same_stream() -> None:
    capturing = False
    library = _FakeLibrary()
    communicator = direct_nccl.DirectNcclCommunicator(
        rank=0,
        world_size=4,
        device=torch.device("cpu"),
        library=library,
        communicator=ctypes.c_void_p(123),
        version="2.29.7",
        stream_resolver=lambda _device: SimpleNamespace(cuda_stream=77),
        capture_resolver=lambda: capturing,
    )

    first_gather = communicator.all_gather_many((torch.ones(3, 5),))[0]
    first_reduce = communicator.reduce_scatter(torch.ones(12, 5))
    smaller_gather = communicator.all_gather_many((torch.ones(2, 5),))[0]
    smaller_reduce = communicator.reduce_scatter(torch.ones(8, 5))

    assert smaller_gather.data_ptr() == first_gather.data_ptr()
    assert smaller_reduce.data_ptr() == first_reduce.data_ptr()
    grown_gather = communicator.all_gather_many((torch.ones(4, 5),))[0]
    grown_reduce = communicator.reduce_scatter(torch.ones(16, 5))
    assert grown_gather.data_ptr() != first_gather.data_ptr()
    assert grown_reduce.data_ptr() != first_reduce.data_ptr()
    assert communicator._captured_gather_workspaces == {}
    assert communicator._captured_reduce_scatter_workspaces == {}

    capturing = True
    captured_small_gather = communicator.all_gather_many((torch.ones(1, 5),))[0]
    captured_small_reduce = communicator.reduce_scatter(torch.ones(4, 5))
    captured_large_gather = communicator.all_gather_many((torch.ones(4, 5),))[0]
    captured_large_reduce = communicator.reduce_scatter(torch.ones(16, 5))
    assert captured_small_gather.data_ptr() != captured_large_gather.data_ptr()
    assert captured_small_reduce.data_ptr() != captured_large_reduce.data_ptr()
    assert {
        item.data_ptr()
        for item in communicator._captured_gather_workspaces.values()
    } == {
        captured_small_gather.data_ptr(),
        captured_large_gather.data_ptr(),
    }
    assert {
        item.data_ptr()
        for item in communicator._captured_reduce_scatter_workspaces.values()
    } == {
        captured_small_reduce.data_ptr(),
        captured_large_reduce.data_ptr(),
    }


class _RaggedControlComm:
    world_size = 4

    def __init__(
        self,
        rank: int,
        *,
        peer_layouts: tuple[tuple[int, ...], ...] | None = None,
    ) -> None:
        self.rank = rank
        self.peer_layouts = peer_layouts
        self.calls = 0

    def all_gather(self, value):
        self.calls += 1
        key = "layouts" if "layouts" in value else "row_sizes"
        layouts = self.peer_layouts or (value[key],) * self.world_size
        return tuple(
            {"rank": rank, key: layout}
            for rank, layout in enumerate(layouts)
        )


def _ragged_direct(
    *,
    rank: int = 0,
    control: _RaggedControlComm | None = None,
) -> tuple[direct_nccl.DirectNcclCommunicator, _FakeLibrary, _RaggedControlComm]:
    library = _FakeLibrary()
    resolved_control = control or _RaggedControlComm(rank)
    communicator = direct_nccl.DirectNcclCommunicator(
        rank=rank,
        world_size=4,
        device=torch.device("cpu"),
        library=library,
        communicator=ctypes.c_void_p(123),
        version="2.29.7",
        control_comm=resolved_control,
        stream_resolver=lambda _device: SimpleNamespace(cuda_stream=77),
    )
    return communicator, library, resolved_control


def _ragged_inputs(rows: int) -> tuple[torch.Tensor, ...]:
    return (
        torch.arange(rows * 4, dtype=torch.uint8).reshape(rows, 4),
        torch.arange(rows * 2, dtype=torch.uint8).reshape(rows, 2),
        torch.arange(rows * 2, dtype=torch.int32).reshape(rows, 2),
        torch.arange(rows * 2, dtype=torch.float32).reshape(rows, 2),
    )


def test_direct_ragged_collectives_group_rank_major_broadcasts_and_reduces() -> None:
    communicator, library, control = _ragged_direct()
    row_sizes = (3, 1, 0, 7)

    gathered = communicator.all_gatherv_many(_ragged_inputs(3), row_sizes)
    source = torch.ones(sum(row_sizes), 5, dtype=torch.bfloat16)
    reduced = communicator.reduce_scatterv(source, row_sizes)

    assert tuple(tensor.shape for tensor in gathered) == (
        (11, 4),
        (11, 2),
        (11, 2),
        (11, 2),
    )
    assert tuple(reduced.shape) == (3, 5)
    assert control.calls == 1
    broadcasts = [call for call in library.calls if call[0] == "broadcast"]
    reductions = [call for call in library.calls if call[0] == "reduce"]
    assert len(broadcasts) == 12
    assert [call[5] for call in broadcasts] == [0] * 4 + [1] * 4 + [3] * 4
    assert [call[3] for call in broadcasts] == [
        12,
        6,
        6,
        6,
        4,
        2,
        2,
        2,
        28,
        14,
        14,
        14,
    ]
    assert len(reductions) == 3
    assert [call[5] for call in reductions] == [0, 1, 3]
    assert [call[3] for call in reductions] == [15, 5, 35]

    first_output = gathered[0]
    first_tensor_broadcasts = (broadcasts[0], broadcasts[4], broadcasts[8])
    assert [call[2] - first_output.data_ptr() for call in first_tensor_broadcasts] == [
        0,
        12,
        16,
    ]
    assert [call[1] - source.data_ptr() for call in reductions] == [0, 30, 40]


def test_direct_ragged_zero_row_owner_uses_non_null_reduce_receive_workspace() -> None:
    communicator, library, _control = _ragged_direct(rank=2)
    row_sizes = (3, 1, 0, 7)

    gathered = communicator.all_gatherv_many(_ragged_inputs(0), row_sizes)
    reduced = communicator.reduce_scatterv(
        torch.ones(sum(row_sizes), 3, dtype=torch.bfloat16),
        row_sizes,
    )

    assert all(tensor.shape[0] == sum(row_sizes) for tensor in gathered)
    assert tuple(reduced.shape) == (0, 3)
    reductions = [call for call in library.calls if call[0] == "reduce"]
    assert reductions
    assert all(call[2] != 0 for call in reductions)


def test_direct_ragged_workspaces_are_current_stream_grow_only() -> None:
    communicator, _library, control = _ragged_direct()
    large_sizes = (3, 1, 0, 7)
    small_sizes = (2, 1, 0, 5)

    large_gather = communicator.all_gatherv_many(
        _ragged_inputs(3),
        large_sizes,
    )[0]
    large_reduce = communicator.reduce_scatterv(
        torch.ones(sum(large_sizes), 5),
        large_sizes,
    )
    small_gather = communicator.all_gatherv_many(
        _ragged_inputs(2),
        small_sizes,
    )[0]
    small_reduce = communicator.reduce_scatterv(
        torch.ones(sum(small_sizes), 5),
        small_sizes,
    )

    assert small_gather.data_ptr() == large_gather.data_ptr()
    assert small_reduce.data_ptr() == large_reduce.data_ptr()
    assert control.calls == 2


def test_direct_ragged_step_sync_repairs_local_cache_drift_before_hot_path() -> None:
    communicator, _library, control = _ragged_direct()
    stale = (1, 1, 1, 2)
    current = (3, 1, 0, 7)
    communicator._validated_ragged_layouts.add(stale)

    assert communicator.synchronize_ragged_layouts((current,)) == (current,)
    assert communicator._validated_ragged_layouts == {current}
    assert control.calls == 1

    communicator.all_gatherv_many(_ragged_inputs(3), current)
    communicator.reduce_scatterv(torch.ones(sum(current), 5), current)
    assert control.calls == 1


def test_direct_ragged_step_sync_rejects_cross_rank_plan_drift() -> None:
    current = ((3, 1, 0, 7),)
    control = _RaggedControlComm(
        0,
        peer_layouts=(
            current,
            current,
            ((3, 2, 0, 6),),
            current,
        ),
    )
    communicator, library, _control = _ragged_direct(control=control)

    with pytest.raises(RuntimeError, match="disagree across ranks"):
        communicator.synchronize_ragged_layouts(current)
    assert library.calls == []


def test_direct_ragged_rejects_cross_rank_layout_disagreement_before_nccl() -> None:
    row_sizes = (3, 1, 0, 7)
    control = _RaggedControlComm(
        0,
        peer_layouts=(row_sizes, row_sizes, (3, 2, 0, 6), row_sizes),
    )
    communicator, library, _control = _ragged_direct(control=control)

    with pytest.raises(RuntimeError, match="disagree across ranks"):
        communicator.all_gatherv_many(_ragged_inputs(3), row_sizes)
    assert library.calls == []
    assert communicator.active is True


@pytest.mark.parametrize(
    ("row_sizes", "match"),
    [
        ([3, 1, 0, 7], "must be a tuple"),
        ((3, 1, 0), "length must equal"),
        ((3, -1, 0, 7), "non-negative integers"),
        ((3, True, 0, 7), "non-negative integers"),
    ],
)
def test_direct_ragged_rejects_malformed_row_sizes(row_sizes, match) -> None:
    communicator, _library, _control = _ragged_direct()

    with pytest.raises((TypeError, ValueError), match=match):
        communicator.all_gatherv_many(_ragged_inputs(3), row_sizes)


def test_direct_ragged_rejects_local_shape_and_missing_control_plane() -> None:
    communicator, _library, _control = _ragged_direct()
    with pytest.raises(ValueError, match="local rows"):
        communicator.all_gatherv_many(_ragged_inputs(2), (3, 1, 0, 7))
    with pytest.raises(ValueError, match=r"sum\(row_sizes\)"):
        communicator.reduce_scatterv(torch.ones(10, 5), (3, 1, 0, 7))

    no_control = direct_nccl.DirectNcclCommunicator(
        rank=0,
        world_size=4,
        device=torch.device("cpu"),
        library=_FakeLibrary(),
        communicator=ctypes.c_void_p(123),
        version="2.29.7",
        stream_resolver=lambda _device: SimpleNamespace(cuda_stream=77),
    )
    with pytest.raises(RuntimeError, match="gloo control communicator"):
        no_control.all_gatherv_many(_ragged_inputs(3), (3, 1, 0, 7))


def test_torch_ragged_api_explicitly_rejects_unimplemented_fallback() -> None:
    communicator = object.__new__(TorchDistCommunicator)
    communicator._attention_dp_direct_nccl = None

    with pytest.raises(RuntimeError, match="fallback is not implemented"):
        communicator.tensor_all_gatherv_many(_ragged_inputs(3), (3, 1, 0, 7))
    with pytest.raises(RuntimeError, match="fallback is not implemented"):
        communicator.tensor_reduce_scatterv(torch.ones(11, 5), (3, 1, 0, 7))


class _DirectWorkspaceComm:
    rank = 0
    world_size = 4
    attention_dp_direct_nccl_active = True

    @staticmethod
    def tensor_all_gather(tensor):
        return tensor

    @staticmethod
    def tensor_reduce_scatter(tensor):
        return tensor


class _FourExpertMoe(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.experts = nn.ModuleList(nn.Identity() for _ in range(4))

    @staticmethod
    def _route(hidden):
        return (
            torch.zeros(hidden.shape[0], 1, dtype=torch.int64),
            torch.ones(hidden.shape[0], 1, dtype=torch.float32),
        )


def test_attention_dp_block_local_partial_workspace_is_per_stream_and_grow_only(
    monkeypatch,
) -> None:
    block = EpMoeBlock(
        _FourExpertMoe(),
        _DirectWorkspaceComm(),
        ep_rank=0,
        ep_size=4,
        attention_dp=True,
    )
    stream = SimpleNamespace(cuda_stream=77)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _device: stream)

    first = block._attention_dp_local_partial(
        rows=12,
        hidden_size=5,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )
    smaller = block._attention_dp_local_partial(
        rows=8,
        hidden_size=5,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )
    assert smaller.data_ptr() == first.data_ptr()

    stream.cuda_stream = 88
    other_stream = block._attention_dp_local_partial(
        rows=8,
        hidden_size=5,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )
    assert other_stream.data_ptr() != first.data_ptr()

    stream.cuda_stream = 77
    block._attention_dp_capture_resolver = lambda: True
    captured_small = block._attention_dp_local_partial(
        rows=8,
        hidden_size=5,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )
    captured_large = block._attention_dp_local_partial(
        rows=16,
        hidden_size=5,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )
    assert captured_small.data_ptr() != captured_large.data_ptr()
    assert {
        item.data_ptr()
        for item in (
            block._attention_dp_captured_local_partial_workspaces.values()
        )
    } == {captured_small.data_ptr(), captured_large.data_ptr()}
