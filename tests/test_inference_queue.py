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


if __name__ == "__main__":
    unittest.main()
