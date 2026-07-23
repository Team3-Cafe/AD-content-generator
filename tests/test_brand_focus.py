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
    build_brand_environment_instruction,
    build_user_instruction,
    run_prompt_generation,
)
from adcg.prompting.schema import (
    normalize_prompt_json,
    normalize_scene_plan_json,
)
from adcg.prompting.system_prompt import (
    BRAND_TREATMENT_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
)


class BrandFocusTests(unittest.TestCase):
    def test_scene_planner_defers_brand_environment_design(self):
        instruction = build_user_instruction(
            product_info={},
            product_focus=0.6,
            brand_focus=0.37,
        )

        self.assertNotIn("Brand focus:", instruction)
        self.assertIn("later text-only LLM call", instruction)
        self.assertIn("do not lock its location, lighting, or camera", instruction)

    def test_second_call_can_redesign_brand_environment(self):
        self.assertIn(
            "Do not lock the location, lighting design, or camera treatment",
            SYSTEM_PROMPT,
        )
        self.assertIn(
            "supplied everyday_direction and studio_direction",
            BRAND_TREATMENT_SYSTEM_PROMPT,
        )
        self.assertIn(
            "force the same fixed set of design categories",
            BRAND_TREATMENT_SYSTEM_PROMPT,
        )
        self.assertIn(
            "everyday_prompt_parts",
            BRAND_TREATMENT_SYSTEM_PROMPT,
        )

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

    def test_first_response_schema_keeps_neutral_scene_reference(self):
        plan = normalize_scene_plan_json({
            "product_analysis": {"objects": ["forklift"]},
            "generation_prompt": {
                "base_background_prompt": "support surface and open copy area",
            },
            "layout": {},
        })
        self.assertEqual(
            plan["generation_prompt"]["base_background_prompt"],
            "support surface and open copy area",
        )

    def test_run_prompt_generation_selects_exact_endpoints(self):
        scene_data = {
            "product_analysis": {
                "protected_subject_terms": [
                    "forklift",
                    "forklift truck",
                    "fork tines",
                    "marker lights",
                ],
            },
            "generation_prompt": {
                "base_background_prompt": (
                    "support beneath forklift, coherent perspective, "
                    "open copy area"
                ),
            },
            "layout": {},
        }
        brand_data = {
            "everyday_prompt_parts": [
                "active neighborhood service yard",
                "weathered concrete with ordinary wear",
                "eye-level documentary framing",
                "ambient available daylight",
                "forklift side profile left foreground",
                "functional loosely organized workspace",
                "open wall area on left",
                "candid practical daily operation",
            ],
            "studio_prompt_parts": [
                "purpose-built luxury exhibition stage",
                "seamless polished dark platform",
                "low-angle telephoto hero framing",
                "sculpted high-contrast studio illumination",
                "forklift hero placement",
                "precisely controlled symmetrical presentation",
                "architectural negative space above",
                "exclusive premium campaign finish",
            ],
        }
        everyday_raw = ", ".join(
            part
            for part in brand_data["everyday_prompt_parts"]
            if "forklift" not in part
        )
        studio_raw = ", ".join(
            part
            for part in brand_data["studio_prompt_parts"]
            if "forklift" not in part
        )
        calls = []

        def create(**kwargs):
            calls.append(kwargs)
            payload = scene_data if len(calls) % 2 else brand_data
            return SimpleNamespace(output_text=json.dumps(payload))

        client = SimpleNamespace(
            responses=SimpleNamespace(
                create=create,
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
            everyday_raw,
            everyday["generation_prompt"]["background_prompt"],
        )
        self.assertTrue(
            studio["generation_prompt"]["background_prompt"].startswith(
                STUDIO_BACKGROUND_ANCHOR
            )
        )
        self.assertIn(
            studio_raw,
            studio["generation_prompt"]["background_prompt"],
        )
        self.assertNotIn(
            "forklift",
            everyday["generation_prompt"]["background_prompt"].casefold(),
        )
        self.assertNotIn(
            "forklift",
            studio["generation_prompt"]["background_prompt"].casefold(),
        )
        self.assertEqual(
            everyday["generation_prompt"]["removed_subject_prompt_parts"][
                "everyday"
            ],
            ["forklift side profile left foreground"],
        )
        self.assertEqual(
            everyday["generation_prompt"]["removed_subject_prompt_parts"][
                "base"
            ],
            ["support beneath forklift"],
        )
        self.assertEqual(
            everyday["controls"]["brand_blend_weight"],
            0.0,
        )
        self.assertEqual(
            studio["controls"]["brand_blend_weight"],
            1.0,
        )
        self.assertEqual(everyday["prompt_stages"]["llm_calls"], 2)
        self.assertEqual(len(calls), 4)
        self.assertIsInstance(calls[0]["input"], list)
        self.assertIsInstance(calls[1]["input"], str)
        self.assertIn("everyday_direction:", calls[1]["input"])
        self.assertIn("studio_direction:", calls[1]["input"])
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
            "base_background_prompt between 12 and 18 English words",
            SYSTEM_PROMPT,
        )
        self.assertIn(
            "Return two ordered lists",
            BRAND_TREATMENT_SYSTEM_PROMPT,
        )
        self.assertIn(
            "force the same fixed set of design categories",
            BRAND_TREATMENT_SYSTEM_PROMPT,
        )


if __name__ == "__main__":
    unittest.main()
