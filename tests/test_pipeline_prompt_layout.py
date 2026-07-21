from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import json
import tempfile
import unittest

from adcg.pipeline import run_pipeline
from adcg.pipelines.copy_layout import run_copy_layout_pipeline
from adcg.pipelines.image import run_image_pipeline


class SplitPipelineTests(unittest.TestCase):
    def test_image_pipeline_stops_after_identity_restoration(self):
        root = Path("C:/pipeline-image-test")
        output_dir = root / "output"
        info_path = root / "product_info.json"
        prompt_path = output_dir / "02_prompt" / "ad_prompt.json"
        generated = {
            "image": root / "generated.png",
            "product_mask": root / "mask.png",
        }
        core_image = root / "core.png"
        identity_image = root / "identity.png"
        calls = []

        with patch.object(
            Path, "mkdir", return_value=None
        ), patch(
            "adcg.pipelines.image.run_preprocess",
            side_effect=lambda **_kwargs: (
                calls.append("preprocess")
                or {
                    "full_cutout": root / "full.png",
                    "trimmed_cutout": root / "trimmed.png",
                }
            ),
        ), patch(
            "adcg.pipelines.image.run_prompt_generation",
            side_effect=lambda **_kwargs: (
                calls.append("prompt") or prompt_path
            ),
        ), patch(
            "adcg.pipelines.image.run_generation",
            side_effect=lambda **_kwargs: (
                calls.append("generation") or generated
            ),
        ), patch(
            "adcg.pipelines.image.run_core_refinement",
            side_effect=lambda **_kwargs: (
                calls.append("core") or core_image
            ),
        ), patch(
            "adcg.pipelines.image.run_identity_restoration",
            side_effect=lambda **_kwargs: (
                calls.append("identity") or identity_image
            ),
        ):
            result = run_image_pipeline(
                image_path=root / "product.png",
                info_path=info_path,
                output_dir=output_dir,
                product_focus=0.65,
                brand_focus=0.73,
            )

        self.assertEqual(
            calls,
            ["preprocess", "prompt", "generation", "core", "identity"],
        )
        self.assertEqual(result.info_path, info_path)
        self.assertEqual(result.prompt_json, prompt_path)
        self.assertEqual(result.identity_restored_image, identity_image)

    def test_copy_layout_pipeline_applies_controls_after_finished_image(self):
        selected_copy = {
            "title": "매일 아침의 따뜻함",
            "subtitle": "동네에서 갓 구운 빵",
            "price": "",
            "cta": "",
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            info_path = root / "product_info.json"
            prompt_path = root / "ad_prompt.json"
            identity_image = root / "identity.png"
            output_dir = root / "output"
            info_path.write_text(
                json.dumps({"product_name": "빵 세트"}),
                encoding="utf-8",
            )
            prompt_path.write_text(
                json.dumps({
                    "generation_prompt": {
                        "background_prompt": "Warm neighborhood bakery",
                    }
                }),
                encoding="utf-8",
            )
            identity_image.write_bytes(b"finished-image")

            layout_result = SimpleNamespace(
                layout_json=output_dir / "07_prompt_layout" / "layout.json",
                final_review_json=(
                    output_dir / "07_prompt_layout" / "final_review.json"
                ),
                rendered_image=(
                    output_dir / "07_prompt_layout" / "final_ad.png"
                ),
            )

            def load_copy(path, copy_index=0):
                document = json.loads(Path(path).read_text(encoding="utf-8"))
                return document["copies"][copy_index]

            with patch(
                "adcg.pipelines.copy_layout.generate_ad_copy",
                return_value=selected_copy,
            ) as copy_generation, patch(
                "adcg.pipelines.copy_layout.load_ad_copy",
                side_effect=load_copy,
            ), patch(
                "adcg.pipelines.copy_layout.generate_prompt_layout",
                return_value=layout_result,
            ) as prompt_layout:
                result = run_copy_layout_pipeline(
                    identity_image=identity_image,
                    info_path=info_path,
                    prompt_json=prompt_path,
                    output_dir=output_dir,
                    copy_tone="따뜻하고 친근한",
                    copy_length="short",
                )

            call = copy_generation.call_args.kwargs
            self.assertEqual(call["copy_tone"], "따뜻하고 친근한")
            self.assertEqual(call["copy_length"], "short")
            self.assertEqual(
                call["background_prompt"],
                "Warm neighborhood bakery",
            )
            prompt_layout.assert_called_once()
            self.assertEqual(
                prompt_layout.call_args.kwargs["image_path"],
                identity_image,
            )
            copy_document = json.loads(
                result.copy_json.read_text(encoding="utf-8")
            )
            self.assertEqual(copy_document["controls"], {
                "copy_tone": "따뜻하고 친근한",
                "copy_length": "short",
            })
            self.assertEqual(result.final_image, layout_result.rendered_image)

    def test_compatibility_pipeline_runs_image_then_copy_layout(self):
        root = Path("C:/pipeline-wrapper-test")
        image_result = SimpleNamespace(
            output_dir=root / "output",
            info_path=root / "info.json",
            prompt_json=root / "prompt.json",
            generated_image=root / "generated.png",
            core_refined_image=root / "core.png",
            identity_restored_image=root / "identity.png",
            eval_json=None,
        )
        copy_result = SimpleNamespace(
            copy_json=root / "copy.json",
            layout_json=root / "layout.json",
            final_review_json=root / "review.json",
            final_image=root / "final.png",
        )
        calls = []

        with patch(
            "adcg.pipeline.run_image_pipeline",
            side_effect=lambda **_kwargs: (
                calls.append("image") or image_result
            ),
        ), patch(
            "adcg.pipeline.run_copy_layout_pipeline",
            side_effect=lambda **_kwargs: (
                calls.append("copy_layout") or copy_result
            ),
        ) as copy_layout:
            result = run_pipeline(
                image_path=root / "product.png",
                info_path=root / "info.json",
                copy_tone="전문적인",
                copy_length="medium",
            )

        self.assertEqual(calls, ["image", "copy_layout"])
        self.assertEqual(
            copy_layout.call_args.kwargs["identity_image"],
            image_result.identity_restored_image,
        )
        self.assertEqual(
            copy_layout.call_args.kwargs["copy_tone"],
            "전문적인",
        )
        self.assertEqual(result.final_image, copy_result.final_image)


if __name__ == "__main__":
    unittest.main()
