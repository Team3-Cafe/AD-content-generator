import ast
import unittest
from pathlib import Path


SOURCE = (
    Path(__file__).resolve().parents[1]
    / "adcg"
    / "pipelines"
    / "image.py"
).read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)
RESOLVER_NODE = next(
    node
    for node in TREE.body
    if isinstance(node, ast.FunctionDef)
    and node.name == "_aspect_aware_generation_size"
)
NAMESPACE = {}
exec(
    compile(
        ast.Module(body=[RESOLVER_NODE], type_ignores=[]),
        "image.py",
        "exec",
    ),
    NAMESPACE,
)
ASPECT_SIZE = NAMESPACE["_aspect_aware_generation_size"]


class GenerationAspectRatioTests(unittest.TestCase):
    def test_landscape_ratio_is_preserved(self):
        width, height = ASPECT_SIZE((1600, 1067))
        self.assertEqual((width, height), (768, 512))
        self.assertAlmostEqual(width / height, 1600 / 1067, places=2)

    def test_portrait_ratio_is_preserved(self):
        width, height = ASPECT_SIZE((736, 1104))
        self.assertEqual((width, height), (512, 768))
        self.assertAlmostEqual(width / height, 736 / 1104, places=2)

    def test_square_input_stays_square(self):
        self.assertEqual(ASPECT_SIZE((900, 900)), (512, 512))

    def test_explicit_resolution_overrides_source_ratio(self):
        self.assertEqual(
            ASPECT_SIZE((1600, 1067), width=1024, height=576),
            (1024, 576),
        )

    def test_explicit_resolution_requires_a_valid_pair(self):
        with self.assertRaises(ValueError):
            ASPECT_SIZE((1600, 1067), width=768)
        with self.assertRaises(ValueError):
            ASPECT_SIZE((1600, 1067), width=767, height=512)

    def test_extreme_ratio_respects_long_side_limit(self):
        width, height = ASPECT_SIZE((4000, 500))
        self.assertLessEqual(width, 1024)
        self.assertEqual(width % 8, 0)
        self.assertEqual(height % 8, 0)


if __name__ == "__main__":
    unittest.main()
