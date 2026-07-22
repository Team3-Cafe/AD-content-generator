import unittest

from adcg.negative_prompts import (
    GENERATION_NEGATIVE_PROMPT,
    IDENTITY_NEGATIVE_PROMPT,
)
from adcg.prompting.schema import normalize_prompt_json


class NegativePromptPolicyTests(unittest.TestCase):
    def test_vlm_negative_is_replaced_with_compact_generation_policy(self):
        normalized = normalize_prompt_json({
            "product_analysis": {},
            "generation_prompt": {
                "background_prompt": "natural everyday kitchen",
                "negative_prompt": "a very long image-specific exclusion list",
            },
            "layout": {},
        })

        self.assertEqual(
            normalized["generation_prompt"]["negative_prompt"],
            GENERATION_NEGATIVE_PROMPT,
        )
        self.assertNotIn(
            "image-specific",
            normalized["generation_prompt"]["negative_prompt"],
        )

    def test_stage_negative_prompts_are_compact_and_distinct(self):
        self.assertLess(len(GENERATION_NEGATIVE_PROMPT.split()), 45)
        self.assertLess(len(IDENTITY_NEGATIVE_PROMPT.split()), 35)
        self.assertNotEqual(GENERATION_NEGATIVE_PROMPT, IDENTITY_NEGATIVE_PROMPT)
        self.assertIn("conflicting perspective", GENERATION_NEGATIVE_PROMPT)
        self.assertIn("jagged edges", IDENTITY_NEGATIVE_PROMPT)

    def test_human_suppression_is_explicit_in_both_stages(self):
        required_terms = (
            "person",
            "human",
            "worker",
            "driver",
            "operator",
            "face",
            "head",
            "torso",
            "arms",
            "legs",
            "human silhouette",
            "human-like figure",
            "seated figure",
            "occupant",
            "mannequin",
        )
        for prompt in (
            GENERATION_NEGATIVE_PROMPT,
            IDENTITY_NEGATIVE_PROMPT,
        ):
            for term in required_terms:
                self.assertIn(term, prompt)


if __name__ == "__main__":
    unittest.main()
