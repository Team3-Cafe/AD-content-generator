from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from threading import Barrier, Event, Lock
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image

from adcg.generation.control import create_depth_control
from adcg.generation.conditioned_diffusion import run_generation
from adcg.preprocessing.product import (
    _remove_background,
    get_rembg_session,
)
from adcg.refinement.identity import _prepare_identity_control_inputs


class ModelCacheTests(unittest.TestCase):
    def tearDown(self):
        get_rembg_session.cache_clear()

    def test_rembg_session_is_loaded_once_per_model(self):
        session = object()
        get_rembg_session.cache_clear()

        with patch(
            "adcg.preprocessing.product.ort.get_available_providers",
            return_value=["CPUExecutionProvider"],
        ), patch(
            "adcg.preprocessing.product.new_session",
            return_value=session,
        ) as load_session:
            first = get_rembg_session("u2net")
            second = get_rembg_session("u2net")

        self.assertIs(first, session)
        self.assertIs(second, session)
        load_session.assert_called_once_with(
            "u2net",
            providers=["CPUExecutionProvider"],
        )

    def test_rembg_prefers_cuda_when_provider_is_available(self):
        session = object()
        get_rembg_session.cache_clear()

        with patch(
            "adcg.preprocessing.product.ort.get_available_providers",
            return_value=[
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
        ), patch(
            "adcg.preprocessing.product.new_session",
            return_value=session,
        ) as load_session:
            selected = get_rembg_session("u2net")

        self.assertIs(selected, session)
        load_session.assert_called_once_with(
            "u2net",
            providers=[
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
        )

    def test_rembg_falls_back_to_cpu_when_cuda_initialization_fails(self):
        cpu_session = object()
        get_rembg_session.cache_clear()

        with patch(
            "adcg.preprocessing.product.ort.get_available_providers",
            return_value=[
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
        ), patch(
            "adcg.preprocessing.product.new_session",
            side_effect=[RuntimeError("CUDA unavailable"), cpu_session],
        ) as load_session, self.assertWarns(RuntimeWarning):
            selected = get_rembg_session("u2net")

        self.assertIs(selected, cpu_session)
        self.assertEqual(load_session.call_count, 2)
        self.assertEqual(
            load_session.call_args_list[1].kwargs["providers"],
            ["CPUExecutionProvider"],
        )

    def test_rembg_inference_is_limited_to_one_concurrent_call(self):
        workers_ready = Barrier(2)
        first_started = Event()
        release_first = Event()
        active_lock = Lock()
        active_calls = 0
        max_active_calls = 0
        call_count = 0

        def fake_remove(image, **_kwargs):
            nonlocal active_calls, max_active_calls, call_count
            with active_lock:
                call_count += 1
                call_number = call_count
                active_calls += 1
                max_active_calls = max(max_active_calls, active_calls)

            if call_number == 1:
                first_started.set()
                if not release_first.wait(timeout=2):
                    raise TimeoutError("첫 rembg 호출 해제 시간 초과")

            with active_lock:
                active_calls -= 1
            return image

        def run_remove():
            workers_ready.wait(timeout=2)
            return _remove_background("image", model="u2net")

        with patch(
            "adcg.preprocessing.product.get_rembg_session",
            return_value=object(),
        ), patch(
            "adcg.preprocessing.product.remove",
            side_effect=fake_remove,
        ):
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [
                    executor.submit(run_remove),
                    executor.submit(run_remove),
                ]
                self.assertTrue(first_started.wait(timeout=2))
                release_first.set()
                results = [
                    future.result(timeout=2)
                    for future in futures
                ]

        self.assertEqual(results, ["image", "image"])
        self.assertEqual(call_count, 2)
        self.assertEqual(max_active_calls, 1)

    def test_depth_inference_is_limited_to_one_concurrent_call(self):
        workers_ready = Barrier(2)
        first_started = Event()
        release_first = Event()
        active_lock = Lock()
        active_calls = 0
        max_active_calls = 0
        call_count = 0

        def fake_depth(**_kwargs):
            nonlocal active_calls, max_active_calls, call_count
            with active_lock:
                call_count += 1
                call_number = call_count
                active_calls += 1
                max_active_calls = max(max_active_calls, active_calls)

            if call_number == 1:
                first_started.set()
                if not release_first.wait(timeout=2):
                    raise TimeoutError("첫 Depth 호출 해제 시간 초과")

            with active_lock:
                active_calls -= 1
            return "depth"

        def run_depth():
            workers_ready.wait(timeout=2)
            return create_depth_control("image")

        with patch(
            "adcg.generation.control._create_depth_control",
            side_effect=fake_depth,
        ):
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [
                    executor.submit(run_depth),
                    executor.submit(run_depth),
                ]
                self.assertTrue(first_started.wait(timeout=2))
                release_first.set()
                results = [
                    future.result(timeout=2)
                    for future in futures
                ]

        self.assertEqual(results, ["depth", "depth"])
        self.assertEqual(call_count, 2)
        self.assertEqual(max_active_calls, 1)

    def test_generation_uses_injected_pipeline_without_reloading(self):
        shared_pipe = SimpleNamespace(
            controlnet=SimpleNamespace(
                nets=[object(), object()],
            )
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            prompt_path = root / "prompt.json"
            prompt_path.write_text(
                json.dumps({
                    "generation_prompt": {
                        "background_prompt": "warm bakery",
                        "negative_prompt": "distorted product",
                    }
                }),
                encoding="utf-8",
            )

            canvas_result = {
                "product_mask": "product-mask",
                "product_layer": "product-layer",
                "condition_canvas": "condition-canvas",
                "placement": {
                    "requested_product_scale": None,
                    "effective_product_scale": 0.5,
                    "sizing_mode": "auto",
                },
            }
            outputs = {
                "image": root / "generated.png",
                "product_mask": root / "mask.png",
            }

            with patch(
                "adcg.generation.conditioned_diffusion."
                "_load_product",
                return_value=SimpleNamespace(width=400, height=200),
            ), patch(
                "adcg.generation.conditioned_diffusion."
                "create_condition_canvas",
                return_value=canvas_result,
            ), patch(
                "adcg.generation.conditioned_diffusion."
                "create_background_inpaint_mask",
                return_value="inpaint-mask",
            ), patch(
                "adcg.generation.conditioned_diffusion."
                "create_canny_control",
                return_value="control-image",
            ), patch(
                "adcg.generation.conditioned_diffusion."
                "create_depth_control",
                return_value="depth-control-image",
            ), patch(
                "adcg.generation.conditioned_diffusion."
                "run_conditioned_inference",
                return_value=(
                    "generated-image",
                    0.1,
                    {
                        "everyday": {"prompt": "everyday"},
                        "studio": {"prompt": "studio"},
                        "negative_prompt": "negative",
                    },
                ),
            ) as inference, patch(
                "adcg.generation.conditioned_diffusion."
                "save_generation_result",
                return_value=outputs,
            ), patch(
                "adcg.generation.conditioned_diffusion."
                "load_generation_pipeline",
            ) as load_pipeline, patch(
                "adcg.generation.conditioned_diffusion."
                "torch.cuda.empty_cache",
            ) as empty_cache:
                result = run_generation(
                    product_image=root / "product.png",
                    prompt_json=prompt_path,
                    output_dir=root / "output",
                    pipe=shared_pipe,
                )

        self.assertEqual(result, outputs)
        self.assertIs(
            inference.call_args.kwargs["pipe"],
            shared_pipe,
        )
        self.assertEqual(
            inference.call_args.kwargs["control_image"],
            ["control-image", "depth-control-image"],
        )
        self.assertEqual(
            inference.call_args.kwargs["controlnet_scale"],
            [0.3, 0.3],
        )
        load_pipeline.assert_not_called()
        empty_cache.assert_not_called()

    def test_identity_reuses_dual_pipeline_with_depth_disabled(self):
        shared_pipe = SimpleNamespace(
            controlnet=SimpleNamespace(
                nets=[object(), object()],
            )
        )
        canny_control = Image.new("RGB", (32, 24), color="white")

        control_images, control_scales = (
            _prepare_identity_control_inputs(
                pipe=shared_pipe,
                control_image=canny_control,
                controlnet_scale=0.6,
            )
        )

        self.assertEqual(len(control_images), 2)
        self.assertIs(control_images[0], canny_control)
        self.assertEqual(control_images[1].getbbox(), None)
        self.assertEqual(control_scales, [0.6, 0.0])


if __name__ == "__main__":
    unittest.main()
