import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from adcg.generation.conditioned_diffusion import run_generation
from adcg.preprocessing.product import get_rembg_session


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

    def test_generation_uses_injected_pipeline_without_reloading(self):
        shared_pipe = object()

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
                "placement": {},
            }
            outputs = {
                "image": root / "generated.png",
                "product_mask": root / "mask.png",
            }

            with patch(
                "adcg.generation.conditioned_diffusion."
                "load_product_cutout",
                return_value="product",
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
        load_pipeline.assert_not_called()
        empty_cache.assert_not_called()


if __name__ == "__main__":
    unittest.main()
