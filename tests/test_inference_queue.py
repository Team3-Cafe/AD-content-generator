from threading import Barrier, Event, Lock
import unittest
from unittest.mock import patch

from adcg.runtime import InferenceQueue


class InferenceQueueTests(unittest.TestCase):
    def test_default_mode_does_not_inject_a_shared_diffusion_pipe(self):
        with patch(
            "adcg.runtime.inference_queue._prepare_image_job",
            side_effect=lambda value: {"prepared": value},
        ) as prepare, patch(
            "adcg.runtime.inference_queue._run_prepared_image_job",
            side_effect=lambda prepared: f"result-{prepared}",
        ) as run:
            queue = InferenceQueue()
            try:
                submission = queue.submit(
                    job_id="safe-default",
                    pipeline_kwargs={"value": "image"},
                )
                self.assertEqual(
                    submission.future.result(timeout=2),
                    "result-image",
                )
            finally:
                queue.close()

        prepare.assert_called_once_with(value="image")
        run.assert_called_once_with(prepared="image")
        self.assertEqual(submission.stage, "completed")

    def test_safe_mode_prepares_concurrently_and_serializes_gpu_runner(self):
        preparation_barrier = Barrier(2)
        active_lock = Lock()
        active_preparations = 0
        max_preparations = 0
        active_gpu = 0
        max_gpu = 0
        runner_calls = []

        def prepare(value):
            nonlocal active_preparations, max_preparations
            with active_lock:
                active_preparations += 1
                max_preparations = max(
                    max_preparations,
                    active_preparations,
                )
            preparation_barrier.wait(timeout=2)
            with active_lock:
                active_preparations -= 1
            return {"value": value}

        def run_prepared(value):
            nonlocal active_gpu, max_gpu
            with active_lock:
                active_gpu += 1
                max_gpu = max(max_gpu, active_gpu)
                runner_calls.append(value)
            with active_lock:
                active_gpu -= 1
            return f"result-{value}"

        queue = InferenceQueue(
            runner=run_prepared,
            prepare_runner=prepare,
            prepare_workers=2,
        )
        try:
            first = queue.submit(
                job_id="safe-1",
                pipeline_kwargs={"value": "first"},
            )
            second = queue.submit(
                job_id="safe-2",
                pipeline_kwargs={"value": "second"},
            )

            self.assertEqual(
                {
                    first.future.result(timeout=2),
                    second.future.result(timeout=2),
                },
                {"result-first", "result-second"},
            )
            self.assertEqual(first.stage, "completed")
            self.assertEqual(second.stage, "completed")
        finally:
            queue.close()

        self.assertEqual(max_preparations, 2)
        self.assertEqual(max_gpu, 1)
        self.assertCountEqual(runner_calls, ["first", "second"])

    def test_jobs_run_fifo_on_one_loaded_model(self):
        shared_pipe = object()
        first_started = Event()
        release_first = Event()
        calls = []
        calls_lock = Lock()
        load_count = 0

        def load_pipe():
            nonlocal load_count
            load_count += 1
            return shared_pipe

        def run_image_job(value, diffusion_pipe):
            with calls_lock:
                calls.append((value, diffusion_pipe))

            if value == "first":
                first_started.set()
                if not release_first.wait(timeout=2):
                    raise TimeoutError("첫 번째 작업 해제 대기 시간 초과")

            return f"result-{value}"

        queue = InferenceQueue(
            pipe_factory=load_pipe,
            runner=run_image_job,
        )
        try:
            first = queue.submit(
                job_id="job-1",
                pipeline_kwargs={"value": "first"},
            )
            self.assertTrue(first_started.wait(timeout=2))

            second = queue.submit(
                job_id="job-2",
                pipeline_kwargs={"value": "second"},
            )
            self.assertEqual(first.queued_ahead, 0)
            self.assertEqual(second.queued_ahead, 1)

            release_first.set()

            self.assertEqual(
                first.future.result(timeout=2),
                "result-first",
            )
            self.assertEqual(
                second.future.result(timeout=2),
                "result-second",
            )
        finally:
            release_first.set()
            queue.close()

        self.assertEqual(load_count, 1)
        self.assertEqual(calls, [
            ("first", shared_pipe),
            ("second", shared_pipe),
        ])

    def test_dedicated_pipelines_load_once_and_route_to_each_job(self):
        background_pipe = object()
        identity_pipe = object()
        load_counts = {"background": 0, "identity": 0}
        calls = []

        def load_background_pipe():
            load_counts["background"] += 1
            return background_pipe

        def load_identity_pipe():
            load_counts["identity"] += 1
            return identity_pipe

        def run_image_job(
            prepared,
            background_pipe,
            identity_pipe,
        ):
            calls.append(
                (prepared, background_pipe, identity_pipe)
            )
            return f"result-{prepared}"

        queue = InferenceQueue(
            background_pipe_factory=load_background_pipe,
            identity_pipe_factory=load_identity_pipe,
            prepare_runner=lambda value: {"prepared": value},
            runner=run_image_job,
        )
        try:
            first = queue.submit(
                job_id="resident-1",
                pipeline_kwargs={"value": "first"},
            )
            second = queue.submit(
                job_id="resident-2",
                pipeline_kwargs={"value": "second"},
            )

            self.assertEqual(
                first.future.result(timeout=2),
                "result-first",
            )
            self.assertEqual(
                second.future.result(timeout=2),
                "result-second",
            )
        finally:
            queue.close()

        self.assertEqual(
            load_counts,
            {"background": 1, "identity": 1},
        )
        self.assertEqual(calls, [
            ("first", background_pipe, identity_pipe),
            ("second", background_pipe, identity_pipe),
        ])

    def test_model_loading_error_is_returned_to_submitted_job(self):
        model_attempted = Event()

        def fail_to_load():
            model_attempted.set()
            raise RuntimeError("out of memory")

        queue = InferenceQueue(
            pipe_factory=fail_to_load,
            runner=lambda **_kwargs: None,
        )
        try:
            self.assertTrue(model_attempted.wait(timeout=2))
            submission = queue.submit(pipeline_kwargs={})

            with self.assertRaisesRegex(
                RuntimeError,
                "GPU 모델 초기화.*out of memory",
            ):
                submission.future.result(timeout=2)
        finally:
            queue.close()


if __name__ == "__main__":
    unittest.main()
