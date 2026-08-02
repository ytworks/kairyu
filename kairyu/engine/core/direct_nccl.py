# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright Kairyu contributors
"""Minimal direct NCCL binding for the attention-DP MoE hot path.

The ctypes approach and ABI declarations are adapted from the Apache-2.0
vLLM/SGLang PyNccl communicator:
https://github.com/sgl-project/sglang/tree/v0.5.16/python/sglang/srt/distributed/device_communicators

Kairyu deliberately owns a much smaller surface: communicator construction,
grouped equal-size or rank-ragged all-gather, SUM reduce-scatter, and explicit
close/abort. The existing torch.distributed process groups remain the control
plane, fallback transport, and lifecycle authority.
"""

from __future__ import annotations

import ctypes
import os
from collections.abc import Callable
from contextlib import nullcontext, suppress
from dataclasses import dataclass
from typing import Any

import torch

_UNIQUE_ID_BYTES = 128
_DEFAULT_LIBRARY = "libnccl.so.2"
_LIBRARY_ENV = "KAIRYU_NCCL_SO_PATH"
_RAGGED_LAYOUT_CACHE_LIMIT = 128


class _NcclUniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_byte * _UNIQUE_ID_BYTES)]


_NcclResult = ctypes.c_int
_NcclComm = ctypes.c_void_p
_CudaStream = ctypes.c_void_p
_Buffer = ctypes.c_void_p
_NcclDataType = ctypes.c_int
_NcclRedOp = ctypes.c_int

_NCCL_SUM = 0
_NCCL_DTYPES = {
    torch.int8: 0,
    torch.uint8: 1,
    torch.int32: 2,
    torch.int64: 4,
    torch.float16: 6,
    torch.float32: 7,
    torch.float64: 8,
    torch.bfloat16: 9,
}


@dataclass(frozen=True)
class _Function:
    name: str
    restype: Any
    argtypes: tuple[Any, ...]


_FUNCTIONS = (
    _Function("ncclGetErrorString", ctypes.c_char_p, (_NcclResult,)),
    _Function("ncclGetVersion", _NcclResult, (ctypes.POINTER(ctypes.c_int),)),
    _Function("ncclGetUniqueId", _NcclResult, (ctypes.POINTER(_NcclUniqueId),)),
    _Function(
        "ncclCommInitRank",
        _NcclResult,
        (ctypes.POINTER(_NcclComm), ctypes.c_int, _NcclUniqueId, ctypes.c_int),
    ),
    _Function(
        "ncclAllGather",
        _NcclResult,
        (_Buffer, _Buffer, ctypes.c_size_t, _NcclDataType, _NcclComm, _CudaStream),
    ),
    _Function(
        "ncclReduceScatter",
        _NcclResult,
        (
            _Buffer,
            _Buffer,
            ctypes.c_size_t,
            _NcclDataType,
            _NcclRedOp,
            _NcclComm,
            _CudaStream,
        ),
    ),
    _Function(
        "ncclBroadcast",
        _NcclResult,
        (
            _Buffer,
            _Buffer,
            ctypes.c_size_t,
            _NcclDataType,
            ctypes.c_int,
            _NcclComm,
            _CudaStream,
        ),
    ),
    _Function(
        "ncclReduce",
        _NcclResult,
        (
            _Buffer,
            _Buffer,
            ctypes.c_size_t,
            _NcclDataType,
            _NcclRedOp,
            ctypes.c_int,
            _NcclComm,
            _CudaStream,
        ),
    ),
    _Function("ncclGroupStart", _NcclResult, ()),
    _Function("ncclGroupEnd", _NcclResult, ()),
    _Function("ncclCommDestroy", _NcclResult, (_NcclComm,)),
    _Function("ncclCommAbort", _NcclResult, (_NcclComm,)),
)


class NcclLibrary:
    """Small checked wrapper over the NCCL C ABI.

    No Python object finalizer destroys communicators: destruction order during
    interpreter teardown is nondeterministic, while ``ncclCommDestroy`` can
    coordinate with peers. The launcher calls :meth:`DirectNcclCommunicator.close`
    only after its normal all-rank rendezvous, or ``abort`` on failed startup.
    """

    _cache: dict[str, tuple[Any, dict[str, Any]]] = {}

    def __init__(
        self,
        library_path: str | None = None,
        *,
        loader: Callable[[str], Any] = ctypes.CDLL,
    ) -> None:
        path = library_path or os.environ.get(_LIBRARY_ENV) or _DEFAULT_LIBRARY
        if not isinstance(path, str) or not path:
            raise ValueError(f"{_LIBRARY_ENV} must be a non-empty path")
        cached = self._cache.get(path)
        if cached is None:
            library = loader(path)
            functions: dict[str, Any] = {}
            for declaration in _FUNCTIONS:
                function = getattr(library, declaration.name)
                function.restype = declaration.restype
                function.argtypes = list(declaration.argtypes)
                functions[declaration.name] = function
            cached = (library, functions)
            self._cache[path] = cached
        self._library, self._functions = cached
        self.path = path

    def _check(self, operation: str, result: int) -> None:
        if result == 0:
            return
        raw = self._functions["ncclGetErrorString"](result)
        detail = raw.decode("utf-8", errors="replace") if raw else "unknown error"
        raise RuntimeError(f"{operation} failed with NCCL {result}: {detail}")

    def version_raw(self) -> int:
        version = ctypes.c_int()
        self._check(
            "ncclGetVersion",
            self._functions["ncclGetVersion"](ctypes.byref(version)),
        )
        return version.value

    def version(self) -> str:
        raw = self.version_raw()
        return f"{raw // 10_000}.{(raw % 10_000) // 100}.{raw % 100}"

    def unique_id(self) -> bytes:
        unique_id = _NcclUniqueId()
        self._check(
            "ncclGetUniqueId",
            self._functions["ncclGetUniqueId"](ctypes.byref(unique_id)),
        )
        return bytes(value & 0xFF for value in unique_id.internal)

    def init_rank(self, world_size: int, unique_id_bytes: bytes, rank: int) -> _NcclComm:
        if len(unique_id_bytes) != _UNIQUE_ID_BYTES:
            raise ValueError(
                f"NCCL unique id must be {_UNIQUE_ID_BYTES} bytes, "
                f"got {len(unique_id_bytes)}"
            )
        unique_id = _NcclUniqueId.from_buffer_copy(unique_id_bytes)
        communicator = _NcclComm()
        self._check(
            "ncclCommInitRank",
            self._functions["ncclCommInitRank"](
                ctypes.byref(communicator),
                world_size,
                unique_id,
                rank,
            ),
        )
        return communicator

    def all_gather(
        self,
        send_ptr: int,
        receive_ptr: int,
        count: int,
        dtype: torch.dtype,
        communicator: _NcclComm,
        stream: int,
    ) -> None:
        self._check(
            "ncclAllGather",
            self._functions["ncclAllGather"](
                _Buffer(send_ptr),
                _Buffer(receive_ptr),
                count,
                _nccl_dtype(dtype),
                communicator,
                _CudaStream(stream),
            ),
        )

    def reduce_scatter(
        self,
        send_ptr: int,
        receive_ptr: int,
        receive_count: int,
        dtype: torch.dtype,
        communicator: _NcclComm,
        stream: int,
    ) -> None:
        self._check(
            "ncclReduceScatter",
            self._functions["ncclReduceScatter"](
                _Buffer(send_ptr),
                _Buffer(receive_ptr),
                receive_count,
                _nccl_dtype(dtype),
                _NCCL_SUM,
                communicator,
                _CudaStream(stream),
            ),
        )

    def broadcast(
        self,
        send_ptr: int,
        receive_ptr: int,
        count: int,
        dtype: torch.dtype,
        root: int,
        communicator: _NcclComm,
        stream: int,
    ) -> None:
        self._check(
            "ncclBroadcast",
            self._functions["ncclBroadcast"](
                _Buffer(send_ptr),
                _Buffer(receive_ptr),
                count,
                _nccl_dtype(dtype),
                root,
                communicator,
                _CudaStream(stream),
            ),
        )

    def reduce(
        self,
        send_ptr: int,
        receive_ptr: int,
        count: int,
        dtype: torch.dtype,
        root: int,
        communicator: _NcclComm,
        stream: int,
    ) -> None:
        self._check(
            "ncclReduce",
            self._functions["ncclReduce"](
                _Buffer(send_ptr),
                _Buffer(receive_ptr),
                count,
                _nccl_dtype(dtype),
                _NCCL_SUM,
                root,
                communicator,
                _CudaStream(stream),
            ),
        )

    def group_start(self) -> None:
        self._check("ncclGroupStart", self._functions["ncclGroupStart"]())

    def group_end(self) -> None:
        self._check("ncclGroupEnd", self._functions["ncclGroupEnd"]())

    def destroy(self, communicator: _NcclComm) -> None:
        self._check(
            "ncclCommDestroy",
            self._functions["ncclCommDestroy"](communicator),
        )

    def abort(self, communicator: _NcclComm) -> None:
        self._check(
            "ncclCommAbort",
            self._functions["ncclCommAbort"](communicator),
        )


def _nccl_dtype(dtype: torch.dtype) -> int:
    try:
        return _NCCL_DTYPES[dtype]
    except KeyError as error:
        raise TypeError(f"direct NCCL does not support dtype {dtype}") from error


def _error_text(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"[:512]


def _fallback_metadata(reason: str) -> dict[str, object]:
    return {
        "selected_backend": "torch.distributed:nccl",
        "fallback_backend": "torch.distributed:nccl",
        "direct_nccl_active": False,
        "direct_nccl_library": None,
        "direct_nccl_version": None,
        "selection_reason": reason[:1024],
    }


class DirectNcclCommunicator:
    """One rank of Kairyu's attention-DP-only NCCL communicator."""

    def __init__(
        self,
        *,
        rank: int,
        world_size: int,
        device: torch.device,
        library: NcclLibrary,
        communicator: _NcclComm,
        version: str,
        control_comm=None,
        stream_resolver: Callable[[torch.device], Any] = torch.cuda.current_stream,
        capture_resolver: Callable[[], bool] | None = None,
    ) -> None:
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.library = library
        self.version = version
        self._control_comm = control_comm
        self._communicator: _NcclComm | None = communicator
        self._stream_resolver = stream_resolver
        self._capture_resolver = capture_resolver
        # These buffers are safe to reuse only because the attention-DP block
        # consumes each result immediately on the same stream. The stream is
        # part of every key; capacity only grows for a stable tail geometry.
        self._gather_workspaces: dict[tuple[object, ...], torch.Tensor] = {}
        self._reduce_scatter_workspaces: dict[
            tuple[object, ...], torch.Tensor
        ] = {}
        # Captured NCCL nodes store raw addresses.  Keep one exact-shape buffer
        # per captured geometry, separate from the eager grow-only cache: this
        # preserves old graph pointers without retaining every historical eager
        # allocation on a long-lived server.
        self._captured_gather_workspaces: dict[
            tuple[object, ...], torch.Tensor
        ] = {}
        self._captured_reduce_scatter_workspaces: dict[
            tuple[object, ...], torch.Tensor
        ] = {}
        self._validated_ragged_layouts: set[tuple[int, ...]] = set()

    @classmethod
    def try_create(
        cls,
        control_comm,
        device: str | torch.device,
        *,
        library_path: str | None = None,
        library_factory: Callable[[str | None], NcclLibrary] = NcclLibrary,
    ) -> tuple[DirectNcclCommunicator | None, dict[str, object]]:
        """Collectively create, or collectively select the Torch NCCL fallback."""

        resolved_device = torch.device(device)
        if resolved_device.type != "cuda":
            return None, _fallback_metadata(
                f"direct NCCL requires CUDA, got {resolved_device.type}"
            )
        rank = control_comm.rank
        world_size = control_comm.world_size
        library: NcclLibrary | None = None
        version: str | None = None
        try:
            library = library_factory(library_path)
            version = library.version()
            local_probe = {"rank": rank, "ok": True, "error": None}
        except Exception as error:
            local_probe = {"rank": rank, "ok": False, "error": _error_text(error)}
        probes = control_comm.all_gather(local_probe)
        failures = tuple(
            probe
            for probe in probes
            if not isinstance(probe, dict) or probe.get("ok") is not True
        )
        if failures:
            details = "; ".join(
                f"rank {probe.get('rank', '?')}: {probe.get('error', 'malformed probe')}"
                if isinstance(probe, dict)
                else f"malformed probe {type(probe).__name__}"
                for probe in failures
            )
            return None, _fallback_metadata(f"direct NCCL library unavailable: {details}")
        if library is None or version is None:  # defensive; probes established this
            raise RuntimeError("direct NCCL library probe succeeded without a binding")

        if rank == 0:
            try:
                unique_id_payload: tuple[bool, bytes | str] = (
                    True,
                    library.unique_id(),
                )
            except Exception as error:
                unique_id_payload = (False, _error_text(error))
        else:
            unique_id_payload = (False, "not-root")
        unique_id_payload = control_comm.broadcast(unique_id_payload, src=0)
        if (
            not isinstance(unique_id_payload, tuple)
            or len(unique_id_payload) != 2
            or type(unique_id_payload[0]) is not bool
        ):
            raise RuntimeError("direct NCCL unique-id broadcast was malformed")
        if not unique_id_payload[0]:
            return None, _fallback_metadata(
                f"direct NCCL unique-id creation failed: {unique_id_payload[1]}"
            )
        unique_id = unique_id_payload[1]
        if not isinstance(unique_id, bytes) or len(unique_id) != _UNIQUE_ID_BYTES:
            raise RuntimeError("direct NCCL unique-id broadcast had an invalid payload")

        communicator: _NcclComm | None = None
        try:
            cuda_context = torch.cuda.device(resolved_device)
            with cuda_context:
                communicator = library.init_rank(world_size, unique_id, rank)
            local_init = {"rank": rank, "ok": True, "error": None}
        except Exception as error:
            local_init = {"rank": rank, "ok": False, "error": _error_text(error)}
        init_rows = control_comm.all_gather(local_init)
        init_failures = tuple(
            row
            for row in init_rows
            if not isinstance(row, dict) or row.get("ok") is not True
        )
        if init_failures:
            if communicator is not None:
                with suppress(Exception):
                    library.abort(communicator)
            details = "; ".join(
                f"rank {row.get('rank', '?')}: {row.get('error', 'malformed result')}"
                if isinstance(row, dict)
                else f"malformed result {type(row).__name__}"
                for row in init_failures
            )
            return None, _fallback_metadata(
                f"direct NCCL communicator initialization failed: {details}"
            )
        if communicator is None:
            raise RuntimeError("direct NCCL initialization succeeded without a communicator")
        direct = cls(
            rank=rank,
            world_size=world_size,
            device=resolved_device,
            library=library,
            communicator=communicator,
            version=version,
            control_comm=control_comm,
        )
        return direct, {
            "selected_backend": "direct_nccl_ctypes",
            "fallback_backend": "torch.distributed:nccl",
            "direct_nccl_active": True,
            "direct_nccl_library": library.path,
            "direct_nccl_version": version,
            "selection_reason": "attention-DP CUDA/NCCL hot path initialized on every rank",
        }

    def _live_communicator(self) -> _NcclComm:
        if self._communicator is None:
            raise RuntimeError("direct NCCL communicator is closed")
        return self._communicator

    @property
    def active(self) -> bool:
        return self._communicator is not None

    def _validate_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("direct NCCL inputs must be tensors")
        if tensor.device != self.device:
            raise ValueError(
                f"direct NCCL communicator is bound to {self.device}, "
                f"got tensor on {tensor.device}"
            )
        _nccl_dtype(tensor.dtype)
        return tensor.contiguous()

    def _stream(self) -> int:
        stream = self._stream_resolver(self.device)
        handle = getattr(stream, "cuda_stream", None)
        if type(handle) is not int or handle < 0:
            raise RuntimeError("current CUDA stream does not expose a valid handle")
        return handle

    def _validate_ragged_row_sizes(
        self,
        row_sizes: tuple[int, ...],
    ) -> tuple[int, ...]:
        if type(row_sizes) is not tuple:
            raise TypeError("direct NCCL ragged row_sizes must be a tuple")
        if len(row_sizes) != self.world_size:
            raise ValueError(
                "direct NCCL ragged row_sizes length must equal "
                f"world_size={self.world_size}, got {len(row_sizes)}"
            )
        if any(type(rows) is not int or rows < 0 for rows in row_sizes):
            raise ValueError(
                "direct NCCL ragged row_sizes must contain non-negative integers"
            )
        if row_sizes in self._validated_ragged_layouts:
            return row_sizes
        if self._control_comm is None:
            raise RuntimeError(
                "ragged direct NCCL requires the gloo control communicator for "
                "all-rank row-size validation"
            )
        local = {"rank": self.rank, "row_sizes": row_sizes}
        gathered = self._control_comm.all_gather(local)
        if not isinstance(gathered, (tuple, list)) or len(gathered) != self.world_size:
            raise RuntimeError(
                "ragged direct NCCL row-size validation returned a malformed "
                "rank count"
            )
        layouts: list[tuple[int, ...]] = []
        for slot, record in enumerate(gathered):
            if (
                not isinstance(record, dict)
                or set(record) != {"rank", "row_sizes"}
                or record.get("rank") != slot
                or not isinstance(record.get("row_sizes"), tuple)
            ):
                raise RuntimeError(
                    "ragged direct NCCL row-size validation returned a malformed "
                    f"record in rank slot {slot}"
                )
            layouts.append(record["row_sizes"])
        if any(layout != row_sizes for layout in layouts):
            raise RuntimeError(
                "ragged direct NCCL row_sizes disagree across ranks: "
                f"{tuple(layouts)!r}"
            )
        if len(self._validated_ragged_layouts) >= _RAGGED_LAYOUT_CACHE_LIMIT:
            self._validated_ragged_layouts.clear()
        self._validated_ragged_layouts.add(row_sizes)
        return row_sizes

    def synchronize_ragged_layouts(
        self,
        layouts: tuple[tuple[int, ...], ...],
    ) -> tuple[tuple[int, ...], ...]:
        """Validate one worker step's ragged layouts on every rank.

        A local cache hit must never let one rank skip a control collective
        while a peer with a cold cache enters it.  Production calls this once
        at the worker boundary, before any model/NCCL operation.  Replacing the
        cache with the agreed step set also repairs any prior local cache drift;
        the per-layer hot path then performs only membership checks.
        """

        if type(layouts) is not tuple or not layouts:
            raise TypeError(
                "direct NCCL synchronized ragged layouts must be a non-empty tuple"
            )
        normalized: list[tuple[int, ...]] = []
        for row_sizes in layouts:
            if type(row_sizes) is not tuple:
                raise TypeError(
                    "direct NCCL synchronized row_sizes must be tuples"
                )
            if len(row_sizes) != self.world_size:
                raise ValueError(
                    "direct NCCL synchronized row_sizes length must equal "
                    f"world_size={self.world_size}, got {len(row_sizes)}"
                )
            if any(type(rows) is not int or rows < 0 for rows in row_sizes):
                raise ValueError(
                    "direct NCCL synchronized row_sizes must contain "
                    "non-negative integers"
                )
            if row_sizes not in normalized:
                normalized.append(row_sizes)
        agreed = tuple(normalized)
        if self._control_comm is None:
            raise RuntimeError(
                "ragged direct NCCL synchronization requires the gloo "
                "control communicator"
            )
        local = {"rank": self.rank, "layouts": agreed}
        gathered = self._control_comm.all_gather(local)
        if not isinstance(gathered, (tuple, list)) or len(gathered) != self.world_size:
            raise RuntimeError(
                "ragged direct NCCL layout synchronization returned a malformed "
                "rank count"
            )
        peer_layouts: list[tuple[tuple[int, ...], ...]] = []
        for slot, record in enumerate(gathered):
            if (
                not isinstance(record, dict)
                or set(record) != {"rank", "layouts"}
                or record.get("rank") != slot
                or not isinstance(record.get("layouts"), tuple)
                or any(
                    not isinstance(layout, tuple)
                    for layout in record["layouts"]
                )
            ):
                raise RuntimeError(
                    "ragged direct NCCL layout synchronization returned a "
                    f"malformed record in rank slot {slot}"
                )
            peer_layouts.append(record["layouts"])
        if any(peer != agreed for peer in peer_layouts):
            raise RuntimeError(
                "ragged direct NCCL synchronized layouts disagree across ranks: "
                f"{tuple(peer_layouts)!r}"
            )
        self._validated_ragged_layouts = set(agreed)
        return agreed

    @staticmethod
    def _workspace(
        cache: dict[tuple[object, ...], torch.Tensor],
        captured_cache: dict[tuple[object, ...], torch.Tensor],
        key: tuple[object, ...],
        *,
        capturing: bool,
        rows: int,
        tail_shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        if capturing:
            captured_key = (*key, rows)
            buffer = captured_cache.get(captured_key)
            if buffer is None:
                buffer = torch.empty(
                    (rows, *tail_shape),
                    dtype=dtype,
                    device=device,
                )
                captured_cache[captured_key] = buffer
            return buffer
        buffer = cache.get(key)
        if buffer is None or buffer.shape[0] < rows:
            buffer = torch.empty(
                (rows, *tail_shape),
                dtype=dtype,
                device=device,
            )
            cache[key] = buffer
        return buffer if buffer.shape[0] == rows else buffer[:rows]

    def _capturing(self) -> bool:
        value = (
            self._capture_resolver()
            if callable(self._capture_resolver)
            else (
                torch.cuda.is_current_stream_capturing()
                if self.device.type == "cuda"
                else False
            )
        )
        if type(value) is not bool:
            raise RuntimeError(
                "CUDA graph capture resolver must return a boolean"
            )
        return value

    def all_gather_many(
        self,
        tensors: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        if type(tensors) is not tuple or not tensors:
            raise ValueError("direct NCCL grouped all-gather requires a non-empty tuple")
        inputs = tuple(self._validate_tensor(tensor) for tensor in tensors)
        if any(tensor.ndim < 1 for tensor in inputs):
            raise ValueError("direct NCCL grouped all-gather inputs need at least one dim")
        stream = self._stream()
        capturing = self._capturing()
        outputs = tuple(
            self._workspace(
                self._gather_workspaces,
                self._captured_gather_workspaces,
                (
                    slot,
                    str(tensor.device),
                    tensor.dtype,
                    tuple(tensor.shape[1:]),
                    stream,
                ),
                capturing=capturing,
                rows=self.world_size * tensor.shape[0],
                tail_shape=tuple(tensor.shape[1:]),
                dtype=tensor.dtype,
                device=tensor.device,
            )
            for slot, tensor in enumerate(inputs)
        )
        communicator = self._live_communicator()
        try:
            self.library.group_start()
            for source, output in zip(inputs, outputs, strict=True):
                self.library.all_gather(
                    source.data_ptr(),
                    output.data_ptr(),
                    source.numel(),
                    source.dtype,
                    communicator,
                    stream,
                )
            self.library.group_end()
        except BaseException:
            with suppress(Exception):
                self.abort()
            raise
        return outputs

    def all_gatherv_many(
        self,
        tensors: tuple[torch.Tensor, ...],
        row_sizes: tuple[int, ...],
    ) -> tuple[torch.Tensor, ...]:
        """Gather four rank-major tensors with a variable row count per rank."""

        if type(tensors) is not tuple or len(tensors) != 4:
            raise ValueError(
                "direct NCCL grouped ragged all-gather requires exactly four tensors"
            )
        sizes = self._validate_ragged_row_sizes(row_sizes)
        inputs = tuple(self._validate_tensor(tensor) for tensor in tensors)
        if any(tensor.ndim < 1 for tensor in inputs):
            raise ValueError(
                "direct NCCL grouped ragged all-gather inputs need at least one dim"
            )
        local_rows = sizes[self.rank]
        if any(tensor.shape[0] != local_rows for tensor in inputs):
            raise ValueError(
                "direct NCCL grouped ragged all-gather local rows must equal "
                f"row_sizes[{self.rank}]={local_rows}"
            )
        total_rows = sum(sizes)
        stream = self._stream()
        capturing = self._capturing()
        outputs = tuple(
            self._workspace(
                self._gather_workspaces,
                self._captured_gather_workspaces,
                (
                    slot,
                    str(tensor.device),
                    tensor.dtype,
                    tuple(tensor.shape[1:]),
                    stream,
                ),
                capturing=capturing,
                rows=total_rows,
                tail_shape=tuple(tensor.shape[1:]),
                dtype=tensor.dtype,
                device=tensor.device,
            )
            for slot, tensor in enumerate(inputs)
        )
        if total_rows == 0:
            return outputs
        communicator = self._live_communicator()
        try:
            self.library.group_start()
            row_offset = 0
            for root, root_rows in enumerate(sizes):
                if root_rows:
                    for source, output in zip(inputs, outputs, strict=True):
                        row_width = output.numel() // total_rows
                        receive_ptr = (
                            output.data_ptr()
                            + row_offset * row_width * output.element_size()
                        )
                        self.library.broadcast(
                            source.data_ptr(),
                            receive_ptr,
                            root_rows * row_width,
                            source.dtype,
                            root,
                            communicator,
                            stream,
                        )
                row_offset += root_rows
            self.library.group_end()
        except BaseException:
            with suppress(Exception):
                self.abort()
            raise
        return outputs

    def reduce_scatter(self, tensor: torch.Tensor) -> torch.Tensor:
        source = self._validate_tensor(tensor)
        if source.ndim < 1 or source.shape[0] % self.world_size != 0:
            raise ValueError(
                "direct NCCL reduce-scatter requires dim 0 divisible by "
                f"world_size={self.world_size}"
            )
        output_rows = source.shape[0] // self.world_size
        stream = self._stream()
        capturing = self._capturing()
        tail_shape = tuple(source.shape[1:])
        output = self._workspace(
            self._reduce_scatter_workspaces,
            self._captured_reduce_scatter_workspaces,
            (str(source.device), source.dtype, tail_shape, stream),
            capturing=capturing,
            rows=output_rows,
            tail_shape=tail_shape,
            dtype=source.dtype,
            device=source.device,
        )
        try:
            self.library.reduce_scatter(
                source.data_ptr(),
                output.data_ptr(),
                output.numel(),
                source.dtype,
                self._live_communicator(),
                stream,
            )
        except BaseException:
            with suppress(Exception):
                self.abort()
            raise
        return output

    def reduce_scatterv(
        self,
        tensor: torch.Tensor,
        row_sizes: tuple[int, ...],
    ) -> torch.Tensor:
        """SUM-reduce rank-major variable chunks to their owner ranks."""

        sizes = self._validate_ragged_row_sizes(row_sizes)
        source = self._validate_tensor(tensor)
        if source.ndim < 1:
            raise ValueError("direct NCCL ragged reduce-scatter input needs one dim")
        total_rows = sum(sizes)
        if source.shape[0] != total_rows:
            raise ValueError(
                "direct NCCL ragged reduce-scatter dim 0 must equal sum(row_sizes)="
                f"{total_rows}, got {source.shape[0]}"
            )
        output_rows = sizes[self.rank]
        stream = self._stream()
        capturing = self._capturing()
        tail_shape = tuple(source.shape[1:])
        # Keep a non-null receive pointer even when this rank owns zero rows;
        # NCCL ignores it for every non-root operation and the zero-root call is
        # skipped consistently on all ranks.
        workspace = self._workspace(
            self._reduce_scatter_workspaces,
            self._captured_reduce_scatter_workspaces,
            (str(source.device), source.dtype, tail_shape, stream),
            capturing=capturing,
            rows=max(1, output_rows),
            tail_shape=tail_shape,
            dtype=source.dtype,
            device=source.device,
        )
        output = workspace[:output_rows]
        if total_rows == 0:
            return output
        row_width = source.numel() // total_rows
        communicator = self._live_communicator()
        try:
            self.library.group_start()
            row_offset = 0
            for root, root_rows in enumerate(sizes):
                if root_rows:
                    send_ptr = (
                        source.data_ptr()
                        + row_offset * row_width * source.element_size()
                    )
                    self.library.reduce(
                        send_ptr,
                        workspace.data_ptr(),
                        root_rows * row_width,
                        source.dtype,
                        root,
                        communicator,
                        stream,
                    )
                row_offset += root_rows
            self.library.group_end()
        except BaseException:
            with suppress(Exception):
                self.abort()
            raise
        return output

    def close(self) -> None:
        communicator = self._communicator
        if communicator is None:
            return
        self._communicator = None
        self._gather_workspaces.clear()
        self._reduce_scatter_workspaces.clear()
        self._captured_gather_workspaces.clear()
        self._captured_reduce_scatter_workspaces.clear()
        self._validated_ragged_layouts.clear()
        context = (
            torch.cuda.device(self.device)
            if self.device.type == "cuda"
            else nullcontext()
        )
        with context:
            try:
                self.library.destroy(communicator)
            except BaseException:
                with suppress(Exception):
                    self.library.abort(communicator)
                raise

    def abort(self) -> None:
        communicator = self._communicator
        if communicator is None:
            return
        self._communicator = None
        self._gather_workspaces.clear()
        self._reduce_scatter_workspaces.clear()
        self._captured_gather_workspaces.clear()
        self._captured_reduce_scatter_workspaces.clear()
        self._validated_ragged_layouts.clear()
        context = (
            torch.cuda.device(self.device)
            if self.device.type == "cuda"
            else nullcontext()
        )
        with context:
            self.library.abort(communicator)
