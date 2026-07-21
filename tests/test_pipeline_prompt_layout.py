from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
import json
import sys
import unittest


def _stub_module(name, **attributes):
    module = ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    sys.modules[name] = module


def _unused(*_args, **_kwargs):
    raise AssertionError("Dependency should be patched by the integration test.")


_stub_module("adcg.eval", run_evaluation=_unused)
_stub_module("adcg.generation", run_generation=_unused)
_stub_module("adcg.preprocessing", run_preprocess=_unused)
_stub_module(
    "adcg.prompt_layout",
    generate_prompt_layout=_unused,
    load_ad_copy=_unused,
)
_stub_module(
    "adcg.prompting",
    generate_ad_copy=_unused,
    run_prompt_generation=_unused,
)
_stub_module(
    "adcg.refinement",
    run_core_refinement=_unused,
    run_identity_restoration=_unused,
)

from adcg.pipeline import run_pipeline


class PipelinePromptLayoutTests(unittest.TestCase):
    def test_full_pipeline_returns_prompt_layout_ad_as_final_image(self):
        root = Path("C:/pipeline-integration-test")
        output_dir = root / "output"
        info_path = root / "product_info.json"
        prompt_path = output_dir / "02_prompt" / "ad_prompt.json"
        identity_image = root / "identity.png"
        layout_dir = output_dir / "07_prompt_layout"
        final_ad = layout_dir / "final_ad.png"
        layout_json = layout_dir / "layout.json"
        final_review_json = layout_dir / "final_review.json"
        generated = {
            "image": root / "generated.png",
            "product_mask": root / "mask.png",
        }
        selected_copy = {
            "title": "Warehouse service",
            "price": "10만원부터",
        }
        layout_result = SimpleNamespace(
            layout_json=layout_json,
            final_review_json=final_review_json,
            rendered_image=final_ad,
        )

        def read_text(path, **_kwargs):
            if path == info_path:
                return json.dumps({"product_name": "Forklift"})
            if path == prompt_path:
                return json.dumps({
                    "generation_prompt": {
                        "background_prompt": "Industrial warehouse",
                    },
                })
            raise AssertionError(f"Unexpected read: {path}")

        with patch.object(
            Path, "mkdir", return_value=None
        ), patch.object(
            Path, "read_text", autospec=True, side_effect=read_text
        ), patch.object(
            Path, "write_text", autospec=True, return_value=1
        ), patch(
            "adcg.pipeline.run_preprocess",
            return_value={
                "full_cutout": root / "full.png",
                "trimmed_cutout": root / "trimmed.png",
            },
        ), patch(
            "adcg.pipeline.run_prompt_generation",
            return_value=prompt_path,
        ), patch(
            "adcg.pipeline.generate_ad_copy",
            return_value=selected_copy,
        ), patch(
            "adcg.pipeline.run_generation",
            return_value=generated,
        ), patch(
            "adcg.pipeline.run_core_refinement",
            return_value=root / "core.png",
        ), patch(
            "adcg.pipeline.run_identity_restoration",
            return_value=identity_image,
        ), patch(
            "adcg.pipeline.load_ad_copy",
            return_value=selected_copy,
        ) as copy_loader, patch(
            "adcg.pipeline.generate_prompt_layout",
            return_value=layout_result,
        ) as prompt_layout:
            result = run_pipeline(
                image_path=root / "product.png",
                info_path=info_path,
                output_dir=output_dir,
            )

        copy_loader.assert_called_once_with(
            output_dir / "02_prompt" / "ad_copy.json",
            copy_index=0,
        )
        prompt_layout.assert_called_once()
        call = prompt_layout.call_args.kwargs
        self.assertEqual(call["image_path"], identity_image)
        self.assertEqual(call["output_dir"], layout_dir)
        self.assertEqual(call["ad_copy"], selected_copy)
        self.assertEqual(result.identity_restored_image, identity_image)
        self.assertEqual(result.layout_json, layout_json)
        self.assertEqual(result.final_review_json, final_review_json)
        self.assertEqual(result.final_image, final_ad)


if __name__ == "__main__":
    unittest.main()