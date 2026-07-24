import json
import unittest
from pathlib import Path
from unittest.mock import patch

from adcg.pipelines import CopyLayoutResult, ImagePipelineResult
from adcg.pipelines.cli import main


class PipelinesCliTests(unittest.TestCase):
    def test_image_command_passes_generation_controls(self):
        result = ImagePipelineResult(
            output_dir=Path("outputs/test"),
            info_path=Path("info.json"),
            prompt_json=Path("prompt.json"),
            identity_restored_image=Path("identity.png"),
            eval_json=None,
        )
        with patch(
            "adcg.pipelines.cli.run_image_pipeline",
            return_value=result,
        ) as image_pipeline, patch("builtins.print") as output:
            main([
                "image",
                "--image", "product.png",
                "--info", "info.json",
                "--output-dir", "outputs/test",
                "--product-focus", "0.7",
                "--product-scale", "0.55",
                "--brand-focus", "0.3",
                "--seed", "7",
            ])

        call = image_pipeline.call_args.kwargs
        self.assertEqual(call["product_focus"], 0.7)
        self.assertEqual(call["product_scale"], 0.55)
        self.assertEqual(call["brand_focus"], 0.3)
        self.assertEqual(call["seed"], 7)
        document = json.loads(output.call_args.args[0])
        self.assertEqual(document["identity_restored_image"], "identity.png")

    def test_copy_layout_command_passes_copy_and_layout_controls(self):
        result = CopyLayoutResult(
            output_dir=Path("outputs/test"),
            copy_json=Path("copy.json"),
            layout_json=Path("layout.json"),
            final_image=Path("final.png"),
        )
        with patch(
            "adcg.pipelines.cli.run_copy_layout_pipeline",
            return_value=result,
        ) as copy_pipeline, patch("builtins.print") as output:
            main([
                "copy-layout",
                "--identity-image", "identity.png",
                "--info", "info.json",
                "--prompt-json", "prompt.json",
                "--output-dir", "outputs/test",
                "--copy-tone", "따뜻하고 친근한",
                "--copy-length", "short",
                "--layout-detail", "low",
                "--layout-temperature", "0.2",
                "--font", "font.ttf",
            ])

        call = copy_pipeline.call_args.kwargs
        self.assertEqual(call["copy_tone"], "따뜻하고 친근한")
        self.assertEqual(call["copy_length"], "short")
        self.assertEqual(call["layout_options"], {
            "detail": "low",
            "temperature": 0.2,
            "font_path": "font.ttf",
        })
        document = json.loads(output.call_args.args[0])
        self.assertEqual(document["final_image"], "final.png")


if __name__ == "__main__":
    unittest.main()
