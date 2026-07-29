"""Side-stream KV handoff and physical completion ownership (m18 D3).

``CudaStreamProvider`` owns the streams needed by a copy. Same-device transfer
uses one side stream. Cross-device P2P uses a source-side copy stream and a
destination-side completion stream because PyTorch's ``copy_`` consults the
current stream on both devices; omitting either silently serializes role
compute.

With ``defer=True``, ``StreamCopyKVHandoff`` returns an opaque completion while
the copy is live. The producer may queue another forward, but destination
publication, decode adoption, source release, and retry cleanup remain forbidden
until ``event.query()`` proves physical completion. ``gate_pending()`` is only a
low-level stream-ordering primitive; a queued wait is not evidence that host
visible ownership is safe to change.

``PDCoordinator`` holds both page leases across that interval and calls
``finalize_completion()`` after the physical poll succeeds. Lost completion or
failed cleanup retains the allocation and fails closed instead of recycling
pages that may still be in use.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from typing import Protocol

from kairyu.engine.core.pd import HandoffCompletionLostError
from kairyu.engine.core.radix_kv import KVAllocation


class StreamProvider(Protocol):
    def begin(self) -> None: ...

    def synchronize(self) -> None: ...

    def record(self) -> object: ...

    def gate(self, event: object) -> None: ...


class CpuNoopEvent:
    """What ``CpuNoopStream`` hands back: waiting on it is a no-op."""

    def __init__(self, events: list[str]) -> None:
        self._events = events
        self.waited = False

    def wait(self, stream=None) -> None:  # noqa: ARG002 - mirrors torch.cuda.Event
        self.waited = True
        self._events.append("wait")

    def synchronize(self) -> None:
        self.waited = True
        self._events.append("wait")

    def query(self) -> bool:
        if not self.waited:
            self.waited = True
            self._events.append("complete")
        return True


class CpuNoopStream:
    """CPU: no streams; records call order for the contract tests."""

    def __init__(self) -> None:
        self.events: list[str] = []

    def begin(self) -> None:
        self.events.append("begin")

    def synchronize(self) -> None:
        self.events.append("synchronize")

    def record(self) -> CpuNoopEvent:
        self.events.append("record")
        return CpuNoopEvent(self.events)

    def gate(self, event: CpuNoopEvent) -> None:
        event.wait()


class _CudaTransferEvent:
    """Keep both halves of a cross-device transfer dependency alive."""

    def __init__(self, done: object, dependencies: tuple[object, ...]) -> None:
        self._done = done
        self._dependencies = dependencies

    def wait(self, stream: object) -> None:
        self._done.wait(stream)

    def synchronize(self) -> None:
        self._done.synchronize()

    def query(self) -> bool:
        return bool(self._done.query())


@dataclass(frozen=True)
class HandoffCompletion:
    """Opaque ownership token for one asynchronously queued handoff."""

    sequence: int


@dataclass
class _PendingTransfer:
    event: object | None
    allocation: object | None
    error: BaseException | None


class CudaStreamProvider:
    """The GPU half of the seam (m18 D3): copy work uses dedicated streams.

    ``begin()`` makes the source copy stream wait for producer work already
    queued, then makes the destination and source transfer streams current.
    Cross-device ``copy_`` therefore stays off both role compute streams.
    ``synchronize()`` closes that window and blocks until both stream tails have
    completed.

    A full ``stream.synchronize()`` is the BLOCKING form's deliberate choice: the
    handoff's commit point publishes the allocation to other threads, and the m18
    D3 ordering rule is that it must never run ahead of the copy. ``record()``
    bridges the source tail into one destination completion event. ``gate()`` can
    order later device work but does not authorize publication; callers must
    still query or synchronize the recorded event.

    What this buys, stated precisely because two earlier versions of this
    docstring got it wrong. ``begin()`` waits on everything ALREADY queued on the
    caller's stream — including work unrelated to the pages, which the copy
    therefore still follows. What it gains is the other direction: work the
    caller queues AFTER that point runs independently of the copy, because the
    two are on separate streams.
    """

    def __init__(
        self,
        device: object | None = None,
        *,
        transfer_device: object | None = None,
        gate_devices: tuple[object, ...] = (),
        require_peer_access: bool = False,
    ) -> None:
        import torch

        if not torch.cuda.is_available():  # pragma: no cover - deploy-day only
            raise RuntimeError("CudaStreamProvider requires CUDA")
        self._torch = torch
        source = torch.device(f"cuda:{torch.cuda.current_device()}" if device is None else device)
        transfer = torch.device(source if transfer_device is None else transfer_device)
        if source.type != "cuda" or transfer.type != "cuda":
            raise ValueError(
                "CudaStreamProvider devices must be CUDA devices "
                f"(source={source}, transfer={transfer})"
            )
        if source.index is None:
            source = torch.device("cuda", torch.cuda.current_device())
        if transfer.index is None:
            transfer = torch.device("cuda", torch.cuda.current_device())
        if (
            require_peer_access
            and source != transfer
            and not torch.cuda.can_device_access_peer(source.index, transfer.index)
        ):
            raise ValueError(
                "deferred cross-device KV handoff requires CUDA peer access; "
                f"{source} cannot access {transfer}"
            )
        self._source_device = source
        self._stream = torch.cuda.Stream(device=transfer)
        self._source_stream = (
            self._stream if source == transfer else torch.cuda.Stream(device=source)
        )
        ordered_devices = (source, transfer, *(torch.device(item) for item in gate_devices))
        self._gate_devices = tuple(dict.fromkeys(ordered_devices))
        self._window: ExitStack | None = None
        self._source_ready: object | None = None

    def begin(self) -> None:
        if self._window is not None:  # pragma: no cover - misuse guard
            raise RuntimeError("CudaStreamProvider.begin() is already open")
        producer = self._torch.cuda.current_stream(device=self._source_device)
        ready = self._torch.cuda.Event()
        ready.record(producer)
        ready.wait(self._source_stream)
        window = ExitStack()
        try:
            # Cross-device Tensor.copy_ consults BOTH devices' current streams.
            # Keep the destination dependency stream current, then leave the
            # source-copy stream as the innermost/current device while transfer()
            # queues its P2P DMA. Otherwise PyTorch silently uses the source
            # compute stream and serializes the next prefill forward behind it.
            if self._source_stream is not self._stream:
                window.enter_context(self._torch.cuda.stream(self._stream))
            window.enter_context(self._torch.cuda.stream(self._source_stream))
        except BaseException:
            window.close()
            self._source_ready = None
            raise
        self._window = window
        self._source_ready = ready

    def _close_window(self) -> None:
        window, self._window = self._window, None
        if window is not None:
            # ExitStack unwinds every entered device context in reverse order,
            # including when transfer() raised.
            window.close()

    def synchronize(self) -> None:
        self._close_window()
        try:
            self._source_stream.synchronize()
            if self._stream is not self._source_stream:
                self._stream.synchronize()
        finally:
            self._source_ready = None

    def record(self) -> object:
        """Close the window and return an EVENT instead of blocking the host.

        This is what makes overlap real: the producer can queue its next step
        immediately, and whoever consumes the copied pages waits on the event.
        ``synchronize()`` remains for callers that want the simple ordering.
        """
        self._close_window()
        dependencies = tuple(
            dependency for dependency in (self._source_ready,) if dependency is not None
        )
        if self._source_stream is not self._stream:
            # Tensor.copy_ normally installs its own cross-device barriers, but
            # the provider contract also permits source-only work. Bridge the
            # source tail explicitly so one destination event covers both.
            source_done = self._torch.cuda.Event()
            source_done.record(self._source_stream)
            source_done.wait(self._stream)
            dependencies += (source_done,)
        event = self._torch.cuda.Event()
        event.record(self._stream)
        self._source_ready = None
        return _CudaTransferEvent(event, dependencies)

    def gate(self, event) -> None:
        """Order everything the caller queues NEXT after ``event``.

        The host is not stopped: the dependency is expressed on the caller's
        stream, so both halves of the m6 D4 rule hold for stream-ordered work —
        reads of the destination pages and rewrites of the released source pages
        alike are queued behind the copy.
        """
        for device in self._gate_devices:
            event.wait(self._torch.cuda.current_stream(device=device))


class StreamCopyKVHandoff:
    """Wrap a KVHandoff with blocking or completion-owned stream execution.

    ``defer=True`` returns without blocking the host and records the copy's
    completion event. The producer can then queue its next step while the copy is
    live. ``completion_ready()`` is the non-blocking physical poll and
    ``finalize_completion()`` is the only operation that may publish or discard
    its destination allocation. ``wait_for_pending()`` is the blocking
    exceptional/control path. ``gate_pending()`` merely installs device-stream
    waits and deliberately leaves every completion owned.

    Default stays ``defer=False``: block before returning, so the commit point
    cannot run ahead of the copy. Deferring moves that responsibility to the
    caller, and a caller that forgets it reads half-written KV — or lets the next
    forward overwrite the pages the copy is reading.

    Events accumulate: a prefill step transfers every prompt that finished in it,
    so one deferred event per transfer is pending until the consumer settles them
    together. Keeping only the last would silently drop the ordering for every
    earlier copy in the step.
    """

    def __init__(self, inner, provider: StreamProvider, *, defer: bool = False) -> None:
        self._inner = inner
        self._provider = provider
        self._defer = defer
        self._pending: dict[HandoffCompletion, _PendingTransfer] = {}
        # Finalization is a two-phase host mutation.  The coordinator first
        # publishes/discards here, then acknowledges after advancing its own
        # state.  Keeping the outcome until that acknowledgement makes a
        # success-then-raise hook safely re-enterable instead of turning into a
        # missing-token KeyError.
        self._settled: dict[HandoffCompletion, BaseException | None] = {}
        self._allocation_completions: dict[int, HandoffCompletion] = {}
        self._poisoned: list[_PendingTransfer] = []
        self._sequence = 0
        managed = getattr(inner, "manage_completion_externally", None)
        if managed is not None:
            managed()

    @property
    def defers(self) -> bool:
        """True when ``transfer()`` returns before the copy has completed.

        A deferring caller must physically finalize its completion before
        reading the destination or releasing the source.
        """
        return self._defer

    @property
    def pending_event(self) -> object | None:
        """The most recent unsettled completion event, or None."""
        if not self._pending:
            return None
        return self._pending[next(reversed(self._pending))].event

    @property
    def pending_events(self) -> tuple[object, ...]:
        return tuple(
            pending.event for pending in self._pending.values() if pending.event is not None
        )

    @property
    def poisoned_allocations(self) -> tuple[object, ...]:
        """Allocations retained because safe cleanup could not be completed."""
        return tuple(
            pending.allocation
            for pending in self._poisoned
            if pending.allocation is not None
        )

    def _poison(
        self,
        allocation: object | None,
        error: BaseException,
        message: str,
    ) -> HandoffCompletionLostError:
        """Retain an unreachable owner and return a fail-closed marker."""
        self._poisoned.append(_PendingTransfer(None, allocation, error))
        return HandoffCompletionLostError(message, allocation=allocation)

    @staticmethod
    def _allocation_was_released(allocation: object | None) -> bool:
        """Whether a raised cleanup hook nevertheless completed its mutation."""
        return allocation is not None and getattr(allocation, "_freed", False) is True

    def _settle_interrupted_transfer(
        self,
        allocation: object | None,
        interruption: BaseException,
    ) -> None:
        """Make a BaseException safe to propagate, or retain ownership."""
        try:
            self._provider.synchronize()
        except BaseException as sync_error:
            lost = self._poison(
                allocation,
                sync_error,
                "handoff interruption was followed by synchronization failure; "
                "ownership remains retained",
            )
            raise lost from interruption
        if allocation is None:
            return
        try:
            self.discard(allocation)
        except BaseException as cleanup_error:
            if not self._allocation_was_released(allocation):
                lost = self._poison(
                    allocation,
                    cleanup_error,
                    "interrupted handoff could not release its destination allocation",
                )
                raise lost from interruption

    @staticmethod
    def _normalize_transfer_error(
        caught: Exception,
        allocation: object | None,
    ) -> Exception:
        """Make every protocol-level transfer failure retryable by one type."""
        from kairyu.engine.core.pd import KVHandoffError

        if isinstance(caught, KVHandoffError):
            return caught
        normalized = KVHandoffError(
            f"handoff transfer failed: {caught}",
            allocation=allocation,
        )
        normalized.__cause__ = caught
        normalized.__suppress_context__ = True
        return normalized

    @staticmethod
    def _normalize_publication_error(caught: BaseException) -> Exception:
        """Turn a safely cleaned publication failure into a retry outcome."""
        from kairyu.engine.core.pd import KVHandoffError

        if isinstance(caught, KVHandoffError):
            caught.allocation = None
            return caught
        normalized = KVHandoffError(
            f"handoff publication failed: {type(caught).__name__}",
        )
        normalized.__cause__ = caught
        normalized.__suppress_context__ = True
        return normalized

    def _propagate_inner_completion_loss(
        self,
        caught: Exception,
        allocation: object | None,
    ) -> None:
        """Unwind the outer stream without weakening an inner fail-closed mark."""
        try:
            self._provider.synchronize()
        except BaseException as sync_error:
            lost = self._poison(
                allocation,
                sync_error,
                "nested handoff completion was lost and the outer stream could "
                "not synchronize; ownership remains retained",
            )
            raise lost from caught
        self._poisoned.append(_PendingTransfer(None, allocation, caught))

    def transfer(
        self, tokens: tuple[int, ...], first_token: int, pages: tuple[int, ...] = ()
    ) -> KVAllocation:
        if not self._defer:
            return self._transfer_blocking(tokens, first_token, pages)

        self._provider.begin()
        allocation = None
        error = None
        try:
            allocation = self._inner.transfer(tokens, first_token, pages)
        except Exception as caught:
            allocation = getattr(caught, "allocation", None)
            if getattr(caught, "handoff_completion_lost", False):
                self._propagate_inner_completion_loss(caught, allocation)
                raise
            error = self._normalize_transfer_error(caught, allocation)
        except BaseException as interruption:
            # Cancellation/termination cannot be represented as a deferred
            # retry, but it must still unwind the active stream contexts and
            # reclaim any destination allocation made before it arrived.
            allocation = getattr(interruption, "allocation", None)
            self._settle_interrupted_transfer(allocation, interruption)
            raise
        # A raising transfer may already have queued part of the copy.  Its
        # completion therefore owns both allocations just like a successful
        # transfer; cleanup is forbidden until query() reports true.
        try:
            event = self._provider.record()
        except BaseException as record_error:
            # There is no usable ownership token. Fall back to the blocking
            # contract before releasing any destination pages; otherwise a
            # queued partial copy could write into a reallocated page.
            try:
                self._provider.synchronize()
            except BaseException as sync_error:
                lost = self._poison(
                    allocation,
                    sync_error,
                    "handoff event recording and fallback synchronization both failed",
                )
                raise lost from record_error
            if allocation is not None:
                try:
                    self.discard(allocation)
                except BaseException as cleanup_error:
                    if not self._allocation_was_released(allocation):
                        lost = self._poison(
                            allocation,
                            cleanup_error,
                            "handoff event recording failed and destination cleanup "
                            "could not release the allocation",
                        )
                        raise lost from cleanup_error
            if not isinstance(record_error, Exception):
                raise
            if error is not None:
                raise error from record_error
            raise
        pending_before = set(self._pending)
        try:
            completion = self._register(event, allocation, error)
        except BaseException as registration_error:
            registered = tuple(
                token for token in self._pending if token not in pending_before
            )
            if len(registered) > 1:  # pragma: no cover - state invariant
                lost = self._poison(
                    allocation,
                    registration_error,
                    "handoff registration created multiple completion owners",
                )
                raise lost from registration_error
            if registered:
                # The token exists despite an outer success-then-raise hook.
                # Bind a retry outcome to it before surfacing the exception so
                # PD can retain the source lease and settle exactly this copy.
                completion = registered[0]
                pending = self._pending[completion]
                if pending.error is None:
                    pending.error = self._normalize_publication_error(
                        registration_error,
                    )
                surfaced: BaseException
                if isinstance(registration_error, Exception):
                    surfaced = pending.error
                else:
                    surfaced = registration_error
                try:
                    surfaced.allocation = allocation  # type: ignore[attr-defined]
                    surfaced.handoff_completion = completion  # type: ignore[attr-defined]
                except Exception:
                    lost = HandoffCompletionLostError(
                        "handoff registration interruption could not retain "
                        "its completion token",
                        allocation=allocation,
                    )
                    lost.handoff_completion = completion  # type: ignore[attr-defined]
                    raise lost from registration_error
                if surfaced is registration_error:
                    raise
                raise surfaced from registration_error

            # No token was installed, but record() returned an event we still
            # own. Close it synchronously before cleanup; otherwise the source
            # could be retried and released while this unregistered copy reads.
            try:
                event.synchronize()
            except BaseException as sync_error:
                lost = self._poison(
                    allocation,
                    sync_error,
                    "handoff registration failed and its recorded event could "
                    "not be synchronized",
                )
                raise lost from registration_error
            if allocation is not None:
                try:
                    self.discard(allocation)
                except BaseException as cleanup_error:
                    if not self._allocation_was_released(allocation):
                        lost = self._poison(
                            allocation,
                            cleanup_error,
                            "handoff registration failed and destination cleanup "
                            "could not release the allocation",
                        )
                        raise lost from registration_error
            if error is not None:
                error.allocation = None  # type: ignore[attr-defined]
                raise error from registration_error
            if isinstance(registration_error, Exception):
                raise self._normalize_publication_error(
                    registration_error,
                ) from registration_error
            raise
        if error is not None:
            error.handoff_completion = completion  # type: ignore[attr-defined]
            raise error
        return allocation

    def _transfer_blocking(
        self,
        tokens: tuple[int, ...],
        first_token: int,
        pages: tuple[int, ...],
    ):
        self._provider.begin()
        allocation = None
        error = None
        try:
            allocation = self._inner.transfer(tokens, first_token, pages)
        except Exception as caught:
            allocation = getattr(caught, "allocation", None)
            if getattr(caught, "handoff_completion_lost", False):
                self._propagate_inner_completion_loss(caught, allocation)
                raise
            error = self._normalize_transfer_error(caught, allocation)
        except BaseException as interruption:
            allocation = getattr(interruption, "allocation", None)
            self._settle_interrupted_transfer(allocation, interruption)
            raise
        # Publication and failed-allocation cleanup happen only after this
        # synchronization.  mark_computed-before-sync exposes partial GPU KV.
        try:
            self._provider.synchronize()
        except BaseException as sync_error:
            lost = self._poison(
                allocation,
                sync_error,
                "blocking handoff synchronization failed; allocation remains owned",
            )
            raise lost from sync_error
        if error is not None:
            if allocation is not None:
                try:
                    self.discard(allocation)
                except BaseException as cleanup_error:
                    if not self._allocation_was_released(allocation):
                        lost = self._poison(
                            allocation,
                            cleanup_error,
                            "failed blocking handoff could not release its destination "
                            "allocation",
                        )
                        raise lost from cleanup_error
            raise error
        if allocation is not None:
            try:
                self.publish(allocation)
            except BaseException as publish_error:
                try:
                    self.discard(allocation)
                except BaseException as cleanup_error:
                    if not self._allocation_was_released(allocation):
                        lost = self._poison(
                            allocation,
                            cleanup_error,
                            "failed publication could not release its destination "
                            "allocation",
                        )
                        raise lost from publish_error
                if isinstance(publish_error, Exception):
                    raise self._normalize_publication_error(
                        publish_error,
                    ) from publish_error
                raise
        return allocation

    def _register(
        self,
        event: object,
        allocation: object | None,
        error: BaseException | None,
    ) -> HandoffCompletion:
        self._sequence += 1
        completion = HandoffCompletion(self._sequence)
        self._pending[completion] = _PendingTransfer(event, allocation, error)
        if allocation is not None:
            self._allocation_completions[id(allocation)] = completion
        return completion

    def completion_for(self, allocation: object) -> HandoffCompletion:
        """Return the token that owns ``allocation`` until physical completion."""
        try:
            return self._allocation_completions[id(allocation)]
        except KeyError:
            raise RuntimeError("handoff allocation has no pending completion") from None

    def completion_ready(self, completion: HandoffCompletion) -> bool:
        """Poll without blocking the host or changing ownership."""
        if completion in self._settled:
            return True
        pending = self._pending[completion]
        if pending.event is None:  # pragma: no cover - poisoned items are separate
            raise HandoffCompletionLostError(
                "handoff completion event was lost",
                allocation=pending.allocation,
            )
        query = getattr(pending.event, "query", None)
        if query is None:
            raise RuntimeError("handoff completion event does not support query()")
        return bool(query())

    def wait_completion(self, completion: HandoffCompletion) -> None:
        """Exceptional-path wait for abort/forget of an in-flight request."""
        if completion in self._settled:
            return
        pending = self._pending[completion]
        if pending.event is None:  # pragma: no cover - poisoned items are separate
            raise HandoffCompletionLostError(
                "handoff completion event was lost",
                allocation=pending.allocation,
            )
        pending.event.synchronize()

    def _forget_completion(self, completion: HandoffCompletion) -> None:
        pending = self._pending.pop(completion)
        if pending.allocation is not None:
            self._allocation_completions.pop(id(pending.allocation), None)

    def finalize_completion(self, completion: HandoffCompletion) -> bool:
        """Publish/discard after physical completion; replay until acknowledged."""
        if completion in self._settled:
            outcome = self._settled[completion]
            if outcome is not None:
                raise outcome
            return True
        if not self.completion_ready(completion):
            return False
        pending = self._pending[completion]
        if pending.error is not None:
            cleanup_error = None
            if pending.allocation is not None:
                # Remove ownership only after cleanup succeeds.  A cleanup
                # exception leaves the completion available for diagnosis/retry.
                try:
                    self.discard(pending.allocation)
                except BaseException as caught:
                    if not self._allocation_was_released(pending.allocation):
                        raise
                    cleanup_error = caught
            self._forget_completion(completion)
            self._settled[completion] = pending.error
            if cleanup_error is not None:
                raise cleanup_error from pending.error
            raise pending.error
        try:
            if pending.allocation is not None:
                self.publish(pending.allocation)
        except BaseException as publish_error:
            cleanup_error = None
            try:
                if pending.allocation is not None:
                    self.discard(pending.allocation)
            except BaseException as caught:
                # Keep the token: cleanup can be retried and the allocation
                # remains reachable instead of becoming a permanent owner lock.
                if not self._allocation_was_released(pending.allocation):
                    raise caught from publish_error
                cleanup_error = caught
            outcome = self._normalize_publication_error(publish_error)
            self._forget_completion(completion)
            self._settled[completion] = outcome
            if cleanup_error is not None:
                raise cleanup_error from publish_error
            if isinstance(publish_error, Exception):
                raise outcome from publish_error
            raise
        self._forget_completion(completion)
        self._settled[completion] = None
        return True

    def acknowledge_completion(self, completion: HandoffCompletion) -> None:
        """Forget a replay outcome after the caller advanced its own phase."""
        if completion not in self._settled:
            raise RuntimeError("handoff completion was not finalized")
        self._settled.pop(completion)

    def wait_for_pending(self) -> None:
        """Block and finalize every deferred copy; a no-op when none is."""
        first_error: BaseException | None = None
        for completion in tuple(self._pending):
            try:
                self.wait_completion(completion)
                self.finalize_completion(completion)
            except BaseException as error:
                if first_error is None:
                    first_error = error
            finally:
                if completion in self._settled:
                    self.acknowledge_completion(completion)
        if first_error is not None:
            raise first_error

    def gate_pending(self) -> None:
        """Install stream waits without claiming host-visible completion.

        A stream wait is useful for low-level stream composition, but it does not
        make an allocation safe to publish or release.  Consequently this method
        deliberately retains every completion; ``finalize_completion`` is the
        only operation that consumes ownership after ``query()`` is true.
        """
        gate = getattr(self._provider, "gate", None)
        for pending in tuple(self._pending.values()):
            if pending.event is None:  # pragma: no cover - poisoned items are separate
                continue
            if gate is not None:
                gate(pending.event)
            else:  # pragma: no cover - every shipped provider has gate()
                pending.event.synchronize()

    def publish(self, allocation: KVAllocation) -> None:
        """Publish destination KV after physical completion."""
        publish = getattr(self._inner, "publish", None)
        if publish is not None:
            publish(allocation)

    def discard(self, allocation: KVAllocation) -> None:
        """Release a failed destination allocation after physical completion."""
        discard = getattr(self._inner, "discard", None)
        if discard is None:
            raise TypeError("wrapped KV handoff does not implement discard()")
        discard(allocation)
