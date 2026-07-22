import json
import unittest
from pathlib import Path
from unittest.mock import patch

from adcg.refinement.prompt import load_refinement_prompt


class RefinementPromptTests(unittest.TestCase):
    def test_prefers_effective_generation_prompt(self):
        result_data = {
            "effective_background_prompt": "token-fitted studio background",
        }

        with patch.object(
            Path,
            "exists",
            return_value=True,
        ), patch.object(
            Path,
            "read_text",
            return_value=json.dumps(result_data),
        ):
            prompt = load_refinement_prompt(
                "raw-prompt.json",
                generation_result_json="experiment-result.json",
            )

        self.assertEqual(prompt, "token-fitted studio background")


if __name__ == "__main__":
    unittest.main()
