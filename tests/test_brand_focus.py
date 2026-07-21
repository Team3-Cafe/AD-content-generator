import unittest

from adcg.prompting.generator import (
    build_user_instruction,
    run_prompt_generation,
)
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
        self.assertIn("Do not snap", instruction)
        self.assertIn("never assume a fixed product category", instruction)

    def test_brand_focus_has_only_two_input_specific_endpoints(self):
        self.assertIn(
            "unmistakably natural everyday background",
            SYSTEM_PROMPT,
        )
        self.assertIn(
            "unmistakably premium studio-style background",
            SYSTEM_PROMPT,
        )
        self.assertIn(
            "Do not assume or hardcode any product category",
            SYSTEM_PROMPT,
        )
        self.assertNotIn("prop density", SYSTEM_PROMPT)
        self.assertNotIn("surface refinement", SYSTEM_PROMPT)

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
            "background_prompt between 20 and 35 English words",
            SYSTEM_PROMPT,
        )
        self.assertIn(
            "Set negative_prompt to an empty string",
            SYSTEM_PROMPT,
        )


if __name__ == "__main__":
    unittest.main()
