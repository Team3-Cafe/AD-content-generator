from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass
from queue import Queue
from threading import Lock, Thread
from typing import Any, Callable
from uuid import uuid4
from weakref import finalize

_STOP = object()


def _run_image_pipeline(**kwargs):
    # Keep queue import light; load the full pipeline only when a job starts.
    from adcg.pipelines.image import run_image_pipeline

    return run_image_pipeline(**kwargs)


@dataclass(frozen=True)
class QueueSubmission:
    job_id: str
    queued_ahead: int
    future: Future


@dataclass(frozen=True)
class _QueueJob:
    job_id: str
    pipeline_kwargs: dict[str, Any]
    future: Future


@dataclass
class _QueueState:
    lock: Lock
    unfinished: int = 0
    closed: bool = False
    model_status: str = "loading"
    model_error: BaseException | None = None


def _complete_job(state: _QueueState) -> None:
    with state.lock:
        state.unfinished -= 1


def _model_initialization_error(error: BaseException) -> RuntimeError:
    return RuntimeError(f"GPU 모델 초기화에 실패했습니다: {error}")


def _fail_jobs_after_model_error(
    jobs: Queue,
    state: _QueueState,
    error: BaseException,
) -> None:
    while True:
        job = jobs.get()

        if job is _STOP:
            jobs.task_done()
            return

        try:
            if job.future.set_running_or_notify_cancel():
                job.future.set_exception(
                    _model_initialization_error(error)
                )
        finally:
            _complete_job(state)
            jobs.task_done()


def _worker_loop(
    jobs: Queue,
    state: _QueueState,
    pipe,
    pipe_factory: Callable[[], Any] | None,
    runner: Callable[..., Any],
) -> None:
    if pipe is None:
        try:
            pipe = pipe_factory()
        except BaseException as error:
            with state.lock:
                state.model_status = "failed"
                state.model_error = error

            _fail_jobs_after_model_error(jobs, state, error)
            return
        else:
            with state.lock:
                state.model_status = "ready"

    while True:
        job = jobs.get()

        if job is _STOP:
            jobs.task_done()
            return

        try:
            if job.future.set_running_or_notify_cancel():
                kwargs = dict(job.pipeline_kwargs)
                kwargs["diffusion_pipe"] = pipe

                try:
                    result = runner(**kwargs)
                except BaseException as error:
                    job.future.set_exception(error)
                else:
                    job.future.set_result(result)
        finally:
            _complete_job(state)
            jobs.task_done()


class InferenceQueue:
    """Run image-pipeline jobs in FIFO order on one shared model."""

    def __init__(
        self,
        pipe=None,
        runner: Callable[..., Any] = _run_image_pipeline,
        pipe_factory: Callable[[], Any] | None = None,
    ) -> None:
        if pipe is None and pipe_factory is None:
            raise ValueError("pipe 또는 pipe_factory가 필요합니다.")

        self._jobs = Queue()
        self._state = _QueueState(
            lock=Lock(),
            model_status="ready" if pipe is not None else "loading",
        )
        self._worker = Thread(
            target=_worker_loop,
            args=(
                self._jobs,
                self._state,
                pipe,
                pipe_factory,
                runner,
            ),
            name="adcg-gpu-worker",
            daemon=True,
        )
        self._worker.start()
        self._finalizer = finalize(self, self._jobs.put, _STOP)

    @property
    def unfinished(self) -> int:
        with self._state.lock:
            return self._state.unfinished

    @property
    def model_status(self) -> str:
        with self._state.lock:
            if self._state.closed:
                return "closed"
            return self._state.model_status

    @property
    def model_error(self) -> str | None:
        with self._state.lock:
            error = self._state.model_error
            return str(error) if error is not None else None

    def submit(
        self,
        pipeline_kwargs: dict[str, Any],
        job_id: str | None = None,
    ) -> QueueSubmission:
        with self._state.lock:
            if self._state.closed:
                raise RuntimeError("추론 큐가 종료되었습니다.")

            submission = QueueSubmission(
                job_id=str(job_id or uuid4().hex),
                queued_ahead=self._state.unfinished,
                future=Future(),
            )

            if self._state.model_error is not None:
                submission.future.set_exception(
                    _model_initialization_error(
                        self._state.model_error
                    )
                )
                return submission

            self._state.unfinished += 1
            self._jobs.put(
                _QueueJob(
                    job_id=submission.job_id,
                    pipeline_kwargs=dict(pipeline_kwargs),
                    future=submission.future,
                )
            )

        return submission

    def close(self, wait: bool = True) -> None:
        with self._state.lock:
            if self._state.closed:
                return
            self._state.closed = True

        if self._finalizer.alive:
            self._finalizer()

        if wait:
            self._worker.join()
