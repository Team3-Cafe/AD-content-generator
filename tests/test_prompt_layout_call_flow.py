import ast
import unittest
from pathlib import Path


GENERATOR_PATH = (
    Path(__file__).resolve().parents[1] / "adcg" / "prompt_layout" / "generator.py"
)
GENERATOR_SOURCE = GENERATOR_PATH.read_text(encoding="utf-8")
GENERATOR_TREE = ast.parse(GENERATOR_SOURCE)
GENERATE_FUNCTION = next(
    node
    for node in GENERATOR_TREE.body
    if isinstance(node, ast.FunctionDef) and node.name == "generate_prompt_layout"
)
GENERATE_SOURCE = ast.get_source_segment(GENERATOR_SOURCE, GENERATE_FUNCTION) or ""


class PromptLayoutCallFlowTests(unittest.TestCase):
    def test_pipeline_uses_exactly_two_vlm_requests(self):
        self.assertEqual(GENERATE_SOURCE.count("_request_json("), 2)
        self.assertNotIn("FINAL_POLISH", GENERATOR_SOURCE)
        self.assertNotIn("build_final_polish", GENERATOR_SOURCE)

    def test_final_review_redesigns_the_rendered_draft_directly(self):
        self.assertIn("image_path=[draft_path, image_path]", GENERATE_SOURCE)
        self.assertIn(
            "_enforce_final_review_revision(final_review, draft_layout)",
            GENERATE_SOURCE,
        )
        self.assertIn(
            "apply_final_review_revision(draft_layout, final_review)",
            GENERATE_SOURCE,
        )
        self.assertNotIn("revised_layout", GENERATE_SOURCE)
        self.assertNotIn("design_revision", GENERATE_SOURCE)
        self.assertNotIn("final_review_input", GENERATE_SOURCE)


if __name__ == "__main__":
    unittest.main()