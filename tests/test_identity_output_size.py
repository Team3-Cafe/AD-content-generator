import ast
import unittest
from pathlib import Path


SOURCE = (
    Path(__file__).resolve().parents[1]
    / "adcg"
    / "refinement"
    / "identity.py"
).read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)
RESOLVER_NODE = next(
    node
    for node in TREE.body
    if isinstance(node, ast.FunctionDef)
    and node.name == "_resolve_output_size"
)
NAMESPACE = {}
exec(
    compile(
        ast.Module(body=[RESOLVER_NODE], type_ignores=[]),
        "identity.py",
        "exec",
    ),
    NAMESPACE,
)
RESOLVE_OUTPUT_SIZE = NAMESPACE["_resolve_output_size"]


class IdentityOutputSizeTests(unittest.TestCase):
    def test_defaults_to_input_image_size(self):
        self.assertEqual(
            RESOLVE_OUTPUT_SIZE((512, 512)),
            (512, 512),
        )
        self.assertEqual(
            RESOLVE_OUTPUT_SIZE((1600, 1067)),
            (1600, 1067),
        )

    def test_explicit_size_remains_supported(self):
        self.assertEqual(
            RESOLVE_OUTPUT_SIZE((512, 512), 576, 1024),
            (576, 1024),
        )

    def test_rejects_partial_or_invalid_override(self):
        with self.assertRaises(ValueError):
            RESOLVE_OUTPUT_SIZE((512, 512), width=576)
        with self.assertRaises(ValueError):
            RESOLVE_OUTPUT_SIZE((512, 512), 0, 1024)


if __name__ == "__main__":
    unittest.main()
