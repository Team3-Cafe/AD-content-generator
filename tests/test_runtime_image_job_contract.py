from pathlib import Path
import unittest
from unittest.mock import patch

from adcg.pipelines.image import run_image_pipeline
from adcg.runtime.image_job import (
    prepare_image_job,
    run_prepared_image_job,
)


class RuntimeImageJobContractTests(unittest.TestCase):
    def _run_pipeline(self, module_name, split):
        root = Path("C:/runtime-contract")
        output_dir = root / "output"
        image_path = root / "source.png"
        info_path = root / "info.json"
        prompt_path = output_dir / "02_prompt" / "ad_prompt.json"
        preprocessed = {
            "full_cutout": root / "full.png",
            "trimmed_cutout": root / "trimmed.png",
            "metadata": root / "preprocess.json",
            "original_size": (1600, 1067),
        }
        generated = {
            "image": root / "generated.png",
            "product_mask": root / "mask.png",
            "metadata": root / "generation.json",
        }
        core_image = root / "core.png"
        identity_image = root / "identity.png"
        calls = []

        def record(name, result):
            def fake(**kwargs):
                calls.append((name, kwargs))
                return result

            return fake

        with patch.object(
            Path, "mkdir", return_value=None
        ), patch(
            f"{module_name}.run_preprocess",
            side_effect=record("preprocess", preprocessed),
        ), patch(
            f"{module_name}.run_prompt_generation",
            side_effect=record("prompt", prompt_path),
        ), patch(
            f"{module_name}.run_generation",
            side_effect=record("generation", generated),
        ), patch(
            f"{module_name}.run_core_refinement",
            side_effect=record("core", core_image),
        ), patch(
            f"{module_name}.run_identity_restoration",
            side_effect=record("identity", identity_image),
        ), patch(
            f"{module_name}.remove_pipeline_stage",
        ), patch(
            f"{module_name}.keep_only",
        ):
            kwargs = {
                "image_path": image_path,
                "info_path": info_path,
                "output_dir": output_dir,
                "gpt_model": "test-vlm",
                "product_focus": 0.65,
                "product_scale": 0.55,
                "width": 768,
                "height": 1024,
                "brand_focus": 0.73,
                "layout_mode": "layout",
                "seed": 123,
                "cpu_offload": False,
            }
            if split:
                prepared = prepare_image_job(**kwargs)
                result = run_prepared_image_job(prepared)
            else:
                result = run_image_pipeline(**kwargs)

        return calls, result

    def test_split_runtime_preserves_every_model_call_and_argument(self):
        baseline_calls, baseline_result = self._run_pipeline(
            "adcg.pipelines.image",
            split=False,
        )
        runtime_calls, runtime_result = self._run_pipeline(
            "adcg.runtime.image_job",
            split=True,
        )

        self.assertEqual(runtime_calls, baseline_calls)
        self.assertEqual(runtime_result, baseline_result)

        generation_kwargs = dict(runtime_calls)["generation"]
        identity_kwargs = dict(runtime_calls)["identity"]
        prompt_kwargs = dict(runtime_calls)["prompt"]

        self.assertIsNone(generation_kwargs["pipe"])
        self.assertIsNone(identity_kwargs["pipe"])
        self.assertEqual(
            prompt_kwargs["image_path"],
            Path("C:/runtime-contract/source.png"),
        )


if __name__ == "__main__":
    unittest.main()
