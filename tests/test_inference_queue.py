from threading import Event, Lock
import unittest

from adcg.runtime import InferenceQueue


class InferenceQueueTests(unittest.TestCase):
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

    def test_gpu_runs_jobs_in_preparation_completion_order(self):
        shared_pipe = object()
        first_preparation_started = Event()
        release_first_preparation = Event()
        gpu_calls = []

        def prepare(value):
            if value == "first":
                first_preparation_started.set()
                if not release_first_preparation.wait(timeout=2):
                    raise TimeoutError("첫 번째 준비 단계 해제 시간 초과")
            return {"value": value}

        def run_gpu_job(value, diffusion_pipe):
            gpu_calls.append((value, diffusion_pipe))
            return f"result-{value}"

        queue = InferenceQueue(
            pipe=shared_pipe,
            runner=run_gpu_job,
            prepare_runner=prepare,
            prepare_workers=2,
        )
        try:
            first = queue.submit(
                job_id="job-1",
                pipeline_kwargs={"value": "first"},
            )
            self.assertTrue(first_preparation_started.wait(timeout=2))
            self.assertEqual(first.stage, "preparing")

            second = queue.submit(
                job_id="job-2",
                pipeline_kwargs={"value": "second"},
            )

            self.assertEqual(
                second.future.result(timeout=2),
                "result-second",
            )
            self.assertEqual(
                gpu_calls,
                [("second", shared_pipe)],
            )

            release_first_preparation.set()
            self.assertEqual(
                first.future.result(timeout=2),
                "result-first",
            )
        finally:
            release_first_preparation.set()
            queue.close()

        self.assertEqual(gpu_calls, [
            ("second", shared_pipe),
            ("first", shared_pipe),
        ])

    def test_preparation_failure_does_not_enter_gpu_queue(self):
        gpu_calls = []

        def prepare(value):
            if value == "invalid":
                raise ValueError("invalid input")
            return {"value": value}

        def run_gpu_job(value, diffusion_pipe):
            gpu_calls.append(value)
            return value

        queue = InferenceQueue(
            pipe=object(),
            runner=run_gpu_job,
            prepare_runner=prepare,
        )
        try:
            invalid = queue.submit(
                pipeline_kwargs={"value": "invalid"}
            )
            valid = queue.submit(
                pipeline_kwargs={"value": "valid"}
            )

            with self.assertRaisesRegex(ValueError, "invalid input"):
                invalid.future.result(timeout=2)
            self.assertEqual(valid.future.result(timeout=2), "valid")
        finally:
            queue.close()

        self.assertEqual(gpu_calls, ["valid"])


if __name__ == "__main__":
    unittest.main()
