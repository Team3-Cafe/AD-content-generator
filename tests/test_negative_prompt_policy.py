import unittest

from adcg.prompting.schema import (
    REQUIRED_NEGATIVE_TERMS,
    prioritize_required_terms,
)


class NegativePromptPolicyTests(unittest.TestCase):
    def test_required_terms_are_first_and_vlm_duplicates_are_removed(self):
        prompt = prioritize_required_terms(
            "people, low quality clutter, logo, low quality clutter",
            REQUIRED_NEGATIVE_TERMS,
        )
        parts = [part.strip() for part in prompt.split(",")]

        self.assertEqual(
            parts[: len(REQUIRED_NEGATIVE_TERMS)],
            list(REQUIRED_NEGATIVE_TERMS),
        )
        self.assertEqual(parts.count("people"), 1)
        self.assertEqual(parts.count("logo"), 1)
        self.assertEqual(parts.count("low quality clutter"), 1)


if __name__ == "__main__":
    unittest.main()
