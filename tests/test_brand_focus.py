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
        self.assertIn("63% authentic everyday environment", instruction)
        self.assertIn("37% premium purpose-built set", instruction)
        self.assertIn("Do not snap", instruction)

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
            "negative_prompt between 30 and 45 English words",
            SYSTEM_PROMPT,
        )


if __name__ == "__main__":
    unittest.main()
