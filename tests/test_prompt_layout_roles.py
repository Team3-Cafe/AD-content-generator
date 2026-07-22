import unittest

from adcg.prompt_layout.generator import _normalize_target_layout_roles


class PromptLayoutRoleTests(unittest.TestCase):
    def test_removes_vlm_invented_cta_when_cta_is_not_rendered(self):
        layout = {
            "elements": [
                {"role": "title"},
                {"role": "subtitle"},
                {"role": "price"},
            ],
        }
        review = {
            "target_layout": {
                "elements": [
                    {"role": "title"},
                    {"role": "subtitle"},
                    {"role": "price"},
                    {"role": "cta"},
                ],
            },
        }

        normalizations = _normalize_target_layout_roles(review, layout)

        self.assertEqual(
            [item["role"] for item in review["target_layout"]["elements"]],
            ["title", "subtitle", "price"],
        )
        self.assertEqual(normalizations[0]["removed_roles"], ["cta"])

    def test_keeps_cta_when_source_copy_rendered_it(self):
        layout = {
            "elements": [{"role": "title"}, {"role": "cta"}],
        }
        review = {
            "target_layout": {
                "elements": [{"role": "title"}, {"role": "cta"}],
            },
        }

        normalizations = _normalize_target_layout_roles(review, layout)

        self.assertEqual(normalizations, [])
        self.assertEqual(
            [item["role"] for item in review["target_layout"]["elements"]],
            ["title", "cta"],
        )


if __name__ == "__main__":
    unittest.main()
