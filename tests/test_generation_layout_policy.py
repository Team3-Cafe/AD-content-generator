import ast
import unittest
from pathlib import Path


SOURCE_PATH = (
    Path(__file__).resolve().parents[1]
    / "adcg"
    / "generation"
    / "conditioned_diffusion.py"
)


class GenerationLayoutPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = SOURCE_PATH.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)

    def test_product_scale_defaults_to_fixed_point_seven(self):
        assignment = next(
            node
            for node in self.tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "GENERATION_DEFAULTS"
                for target in node.targets
            )
        )
        defaults = ast.literal_eval(assignment.value)
        self.assertEqual(defaults["product_scale"], 0.70)

    def test_product_scale_uses_runtime_config_not_prompt_fallback(self):
        function = next(
            node
            for node in self.tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_resolve_layout"
        )
        function_source = ast.get_source_segment(self.source, function)
        self.assertIn('float(config["product_scale"])', function_source)
        self.assertNotIn('resolve("product_scale"', function_source)


if __name__ == "__main__":
    unittest.main()