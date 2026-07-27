from __future__ import annotations

from concurrent.futures import (
    CancelledError,
    Future,
    InvalidStateError,
    ThreadPoolExecutor,
)
from dataclasses import dataclass, field
from functools import partial
from queue import Queue
from threading import Lock, RLock, Thread
from typing import Any, Callable
from uuid import uuid4
from weakref import finalize

_STOP = object()


def _prepare_image_pipeline(**kwargs):
    # Load pipeline modules only after a job is submitted.
    from adcg.pipelines.image import prepare_image_pipeline

    return {"prepared": prepare_image_pipeline(**kwargs)}


def _run_prepared_image_pipeline(**kwargs):
    from adcg.pipelines.image import run_prepared_image_pipeline

    return run_prepared_image_pipeline(**kwargs)


@dataclass
class _JobProgress:
    lock: Lock = field(default_factory=Lock)
    stage: str = "preparing"

    def get(self) -> str:
        with self.lock:
            return self.stage

    def set(self, stage: str) -> None:
        with self.lock:
            self.stage = stage


@dataclass(frozen=True)
class QueueSubmission:
    job_id: str
    queued_ahead: int
    future: Future
    _progress: _JobProgress = field(repr=False, compare=False)

    @property
    def stage(self) -> str:
        """Current stage: preparing, waiting_gpu, running_gpu, or terminal."""
        return self._progress.get()


@dataclass(frozen=True)
class _QueueJob:
    job_id: str
    pipeline_kwargs: dict[str, Any]
    future: Future
    progress: _JobProgress


@dataclass
class _QueueState:
    lock: Any
    unfinished: int = 0
    closed: bool = False
    abandon_prepared: bool = False
    model_status: str = "loading"
    model_error: BaseException | None = None


def _complete_job(state: _QueueState) -> None:
    with state.lock:
        state.unfinished -= 1


def _set_future_exception(
    future: Future,
    error: BaseException,
) -> None:
    try:
        future.set_exception(error)
    except InvalidStateError:
        # The caller may cancel while preparation is still running.
        pass


def _model_initialization_error(error: BaseException) -> RuntimeError:
    return RuntimeError(f"GPU 모델 초기화에 실패했습니다: {error}")


def _preparation_finished(
    jobs: Queue,
    state: _QueueState,
    job: _QueueJob,
    preparation: Future,
) -> None:
    try:
        prepared_kwargs = preparation.result()
        if not isinstance(prepared_kwargs, dict):
            raise TypeError(
                "prepare_runner는 GPU runner용 인자 dict를 반환해야 합니다."
            )
    except CancelledError:
        job.future.cancel()
        job.progress.set("cancelled")
        _complete_job(state)
        return
    except BaseException as error:
        if job.future.cancelled():
            job.progress.set("cancelled")
        else:
            job.progress.set("failed")
            _set_future_exception(job.future, error)
        _complete_job(state)
        return

    with state.lock:
        if job.future.cancelled():
            terminal = "cancelled"
            error = None
        elif state.abandon_prepared:
            terminal = "cancelled"
            error = None
        elif state.model_error is not None:
            terminal = "failed"
            error = _model_initialization_error(state.model_error)
        else:
            job.progress.set("waiting_gpu")
            jobs.put(
                _QueueJob(
                    job_id=job.job_id,
                    pipeline_kwargs=dict(prepared_kwargs),
                    future=job.future,
                    progress=job.progress,
                )
            )
            return

    if terminal == "cancelled":
        job.future.cancel()
    elif error is not None:
        _set_future_exception(job.future, error)
    job.progress.set(terminal)
    _complete_job(state)


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
                job.progress.set("failed")
                job.future.set_exception(
                    _model_initialization_error(error)
                )
            else:
                job.progress.set("cancelled")
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

        kwargs = None
        try:
            if job.future.set_running_or_notify_cancel():
                job.progress.set("running_gpu")
                kwargs = dict(job.pipeline_kwargs)
                kwargs["diffusion_pipe"] = pipe

                try:
                    result = runner(**kwargs)
                except BaseException as error:
                    job.progress.set("failed")
                    job.future.set_exception(error)
                else:
                    job.progress.set("completed")
                    job.future.set_result(result)
            else:
                job.progress.set("cancelled")
        finally:
            _complete_job(state)
            jobs.task_done()
            # Do not retain prepared PIL artifacts while waiting for next job.
            kwargs = None
            job = None


def _finalize_inference_queue(
    jobs: Queue,
    state: _QueueState,
    preparation_executor: ThreadPoolExecutor,
) -> None:
    with state.lock:
        state.closed = True
        state.abandon_prepared = True

    preparation_executor.shutdown(
        wait=False,
        cancel_futures=True,
    )
    jobs.put(_STOP)


class InferenceQueue:
    """Prepare concurrently, then run GPU-ready jobs FIFO on one model."""

    def __init__(
        self,
        pipe=None,
        runner: Callable[..., Any] | None = None,
        pipe_factory: Callable[[], Any] | None = None,
        prepare_runner: Callable[..., dict[str, Any]] | None = None,
        prepare_workers: int = 4,
    ) -> None:
        if pipe is None and pipe_factory is None:
            raise ValueError("pipe 또는 pipe_factory가 필요합니다.")
        if prepare_workers < 1:
            raise ValueError("prepare_workers는 1 이상이어야 합니다.")

        if runner is None:
            runner = _run_prepared_image_pipeline
            if prepare_runner is None:
                prepare_runner = _prepare_image_pipeline

        self._jobs = Queue()
        self._state = _QueueState(
            lock=RLock(),
            model_status="ready" if pipe is not None else "loading",
        )
        self._prepare_runner = prepare_runner
        self._preparation_executor = ThreadPoolExecutor(
            max_workers=prepare_workers,
            thread_name_prefix="adcg-job-prepare",
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
        self._finalizer = finalize(
            self,
            _finalize_inference_queue,
            self._jobs,
            self._state,
            self._preparation_executor,
        )

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
        progress = _JobProgress()

        with self._state.lock:
            if self._state.closed:
                raise RuntimeError("추론 큐가 종료되었습니다.")

            submission = QueueSubmission(
                job_id=str(job_id or uuid4().hex),
                queued_ahead=self._state.unfinished,
                future=Future(),
                _progress=progress,
            )

            if self._state.model_error is not None:
                progress.set("failed")
                submission.future.set_exception(
                    _model_initialization_error(
                        self._state.model_error
                    )
                )
                return submission

            self._state.unfinished += 1
            job = _QueueJob(
                job_id=submission.job_id,
                pipeline_kwargs=dict(pipeline_kwargs),
                future=submission.future,
                progress=progress,
            )

            if self._prepare_runner is None:
                progress.set("waiting_gpu")
                self._jobs.put(job)
            else:
                preparation = self._preparation_executor.submit(
                    self._prepare_runner,
                    **job.pipeline_kwargs,
                )
                preparation.add_done_callback(
                    partial(
                        _preparation_finished,
                        self._jobs,
                        self._state,
                        job,
                    )
                )

        return submission

    def close(self, wait: bool = True) -> None:
        with self._state.lock:
            if self._state.closed:
                return
            self._state.closed = True
            if not wait:
                self._state.abandon_prepared = True

        if self._finalizer.alive:
            self._finalizer.detach()

        self._preparation_executor.shutdown(
            wait=wait,
            cancel_futures=not wait,
        )
        self._jobs.put(_STOP)

        if wait:
            self._worker.join()
