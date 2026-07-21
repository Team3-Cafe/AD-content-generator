import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from adcg.prompting.copywrite_generate import generate_ad_copy


class CopyControlTests(unittest.TestCase):
    def test_tone_and_length_are_added_only_when_supplied(self):
        captured = {}

        def create(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                output_text=json.dumps({
                    "title": "따뜻한 한 끼",
                    "subtitle": "오늘도 갓 구운 빵",
                    "price": "",
                    "cta": "",
                }, ensure_ascii=False)
            )

        client = SimpleNamespace(
            responses=SimpleNamespace(create=create)
        )
        with patch(
            "adcg.prompting.copywrite_generate.OpenAI",
            return_value=client,
        ):
            result = generate_ad_copy(
                product_info={"product_name": "빵 세트"},
                background_prompt="Warm bakery",
                model="test-model",
                copy_tone="따뜻하고 친근한",
                copy_length="short",
            )

        self.assertIn("따뜻하고 친근한", captured["input"])
        self.assertIn("12 Korean characters", captured["input"])
        self.assertEqual(result["title"], "따뜻한 한 끼")

    def test_invalid_copy_length_is_rejected_before_api_call(self):
        with self.assertRaisesRegex(ValueError, "copy_length"):
            generate_ad_copy(
                product_info={},
                background_prompt="Background",
                model="test-model",
                copy_length="extra-long",
            )


if __name__ == "__main__":
    unittest.main()
