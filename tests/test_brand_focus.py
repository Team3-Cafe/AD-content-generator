import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from adcg.brand_focus import (
    EVERYDAY_BACKGROUND_ANCHOR,
    STUDIO_BACKGROUND_ANCHOR,
    anchor_background_prompts,
    blend_prompt_embeddings,
    brand_blend_weight,
    select_background_prompt,
)
from adcg.prompting.generator import (
    build_user_instruction,
    run_prompt_generation,
)
from adcg.prompting.schema import normalize_prompt_json
from adcg.prompting.system_prompt import SYSTEM_PROMPT


class BrandFocusTests(unittest.TestCase):
    def test_continuous_value_is_passed_to_scene_planner(self):
        instruction = build_user_instruction(
            product_info={},
            product_focus=0.6,
            brand_focus=0.37,
        )

        self.assertIn("Brand focus:\n0.37", instruction)
        self.assertIn("63% natural everyday background", instruction)
        self.assertIn("37% premium studio-style background", instruction)
        self.assertIn("Always produce both", instruction)
        self.assertIn("Never assume a fixed product category", instruction)

    def test_brand_focus_has_only_two_input_specific_endpoints(self):
        self.assertIn(
            "unmistakably natural everyday background",
            SYSTEM_PROMPT,
        )
        self.assertIn(
            "unmistakably premium studio-style version",
            SYSTEM_PROMPT,
        )
        self.assertIn(
            "Do not assume or hardcode any product category",
            SYSTEM_PROMPT,
        )
        self.assertNotIn("prop density", SYSTEM_PROMPT)
        self.assertNotIn("surface refinement", SYSTEM_PROMPT)

    def test_anchors_are_domain_neutral_and_preserve_input_scene(self):
        everyday, studio = anchor_background_prompts(
            "input-specific operational scene",
            "input-specific presentation scene",
        )
        self.assertTrue(everyday.startswith(EVERYDAY_BACKGROUND_ANCHOR))
        self.assertTrue(studio.startswith(STUDIO_BACKGROUND_ANCHOR))
        self.assertIn("input-specific operational scene", everyday)
        self.assertIn("input-specific presentation scene", studio)

        anchors = (
            EVERYDAY_BACKGROUND_ANCHOR
            + " "
            + STUDIO_BACKGROUND_ANCHOR
        ).casefold()
        for location in ("warehouse", "kitchen", "bathroom", "office"):
            self.assertNotIn(location, anchors)

    def test_anchor_insertion_is_idempotent(self):
        everyday, studio = anchor_background_prompts(
            EVERYDAY_BACKGROUND_ANCHOR + ", scene",
            STUDIO_BACKGROUND_ANCHOR + ", scene",
        )
        self.assertEqual(everyday.count(EVERYDAY_BACKGROUND_ANCHOR), 1)
        self.assertEqual(studio.count(STUDIO_BACKGROUND_ANCHOR), 1)
    def test_endpoint_weights_and_contrast_curve(self):
        self.assertEqual(brand_blend_weight(0.0), 0.0)
        self.assertEqual(brand_blend_weight(0.5), 0.5)
        self.assertEqual(brand_blend_weight(1.0), 1.0)
        self.assertLess(brand_blend_weight(0.25), 0.25)
        self.assertGreater(brand_blend_weight(0.75), 0.75)

    def test_endpoint_selection_and_continuous_blend(self):
        self.assertEqual(
            select_background_prompt("everyday", "studio", 0.0),
            "everyday",
        )
        self.assertEqual(
            select_background_prompt("everyday", "studio", 1.0),
            "studio",
        )
        self.assertEqual(blend_prompt_embeddings(0.0, 10.0, 0.5), 5.0)

    def test_prompt_schema_supports_endpoints_and_legacy_prompt(self):
        endpoint_data = normalize_prompt_json({
            "product_analysis": {},
            "generation_prompt": {
                "everyday_background_prompt": "real everyday room",
                "studio_background_prompt": "premium studio room",
            },
            "layout": {},
        })["generation_prompt"]
        self.assertEqual(
            endpoint_data["everyday_background_prompt"],
            "real everyday room",
        )
        self.assertEqual(
            endpoint_data["studio_background_prompt"],
            "premium studio room",
        )

        legacy_data = normalize_prompt_json({
            "product_analysis": {},
            "generation_prompt": {
                "background_prompt": "legacy room",
            },
            "layout": {},
        })["generation_prompt"]
        self.assertEqual(
            legacy_data["everyday_background_prompt"],
            "legacy room",
        )
        self.assertEqual(
            legacy_data["studio_background_prompt"],
            "legacy room",
        )
    def test_run_prompt_generation_selects_exact_endpoints(self):
        response_data = {
            "product_analysis": {},
            "generation_prompt": {
                "everyday_background_prompt": "authentic everyday room with ordinary context",
                "studio_background_prompt": "premium studio room with deliberate presentation",
                "negative_prompt": "",
            },
            "layout": {},
        }
        response = SimpleNamespace(
            output_text=json.dumps(response_data),
        )
        client = SimpleNamespace(
            responses=SimpleNamespace(
                create=lambda **_kwargs: response,
            ),
        )

        with patch(
            "adcg.prompting.generator.load_json",
            return_value={},
        ), patch(
            "adcg.prompting.generator.image_to_data_url",
            return_value="data:image/png;base64,aW1hZ2U=",
        ), patch.object(
            Path,
            "exists",
            return_value=True,
        ), patch.object(
            Path,
            "mkdir",
        ), patch.object(
            Path,
            "write_text",
            return_value=1,
        ) as write_text:
            run_prompt_generation(
                image_path="input.png",
                info_path="info.json",
                output_path="everyday.json",
                brand_focus=0.0,
                client=client,
            )
            everyday = json.loads(write_text.call_args.args[0])
            write_text.reset_mock()

            run_prompt_generation(
                image_path="input.png",
                info_path="info.json",
                output_path="studio.json",
                brand_focus=1.0,
                client=client,
            )
            studio = json.loads(write_text.call_args.args[0])

        self.assertTrue(
            everyday["generation_prompt"]["background_prompt"].startswith(
                EVERYDAY_BACKGROUND_ANCHOR
            )
        )
        self.assertIn(
            response_data["generation_prompt"][
                "everyday_background_prompt"
            ],
            everyday["generation_prompt"]["background_prompt"],
        )
        self.assertTrue(
            studio["generation_prompt"]["background_prompt"].startswith(
                STUDIO_BACKGROUND_ANCHOR
            )
        )
        self.assertIn(
            response_data["generation_prompt"][
                "studio_background_prompt"
            ],
            studio["generation_prompt"]["background_prompt"],
        )
        self.assertEqual(
            everyday["controls"]["brand_blend_weight"],
            0.0,
        )
        self.assertEqual(
            studio["controls"]["brand_blend_weight"],
            1.0,
        )
    def test_brand_focus_range_is_validated(self):
        with self.assertRaisesRegex(ValueError, "brand_focus"):
            run_prompt_generation(
                image_path="unused.png",
                info_path="unused.json",
                output_path="unused-output.json",
                brand_focus=1.01,
            )

    def test_vlm_prompt_contains_word_budgets(self):
        self.assertIn(
            "each endpoint prompt between 20 and 35 English words",
            SYSTEM_PROMPT,
        )
        self.assertIn(
            "Set negative_prompt to an empty string",
            SYSTEM_PROMPT,
        )


if __name__ == "__main__":
    unittest.main()
