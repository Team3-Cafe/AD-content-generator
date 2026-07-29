import ast
import unittest
from pathlib import Path

import cv2
import numpy as np


SOURCE = (
    Path(__file__).resolve().parents[1]
    / "adcg"
    / "refinement"
    / "core.py"
).read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)
RUN_CORE = next(
    node
    for node in TREE.body
    if isinstance(node, ast.FunctionDef)
    and node.name == "run_core_refinement"
)
FUNCTION_NAMES = {
    "_background_blur_radius",
    "_far_blur_radius",
    "_background_proximity_weight",
    "_blur_background_only",
}
FUNCTIONS = [
    node
    for node in TREE.body
    if isinstance(node, ast.FunctionDef)
    and node.name in FUNCTION_NAMES
]
NAMESPACE = {"cv2": cv2, "np": np}
exec(
    compile(ast.Module(body=FUNCTIONS, type_ignores=[]), "core.py", "exec"),
    NAMESPACE,
)
BLUR_RADIUS = NAMESPACE["_background_blur_radius"]
FAR_BLUR_RADIUS = NAMESPACE["_far_blur_radius"]
PROXIMITY_WEIGHT = NAMESPACE["_background_proximity_weight"]
BLUR_BACKGROUND = NAMESPACE["_blur_background_only"]


class ProductFocusBlurTests(unittest.TestCase):
    def test_product_guard_margin_is_fourteen_pixels(self):
        positional = RUN_CORE.args.args
        defaults = RUN_CORE.args.defaults
        first_default = len(positional) - len(defaults)
        default_by_name = {
            positional[index + first_default].arg: ast.literal_eval(default)
            for index, default in enumerate(defaults)
        }
        self.assertEqual(default_by_name["outer_protection"], 14)

    def test_maximum_focus_doubles_previous_radius(self):
        self.assertEqual(BLUR_RADIUS(1.0), 52)

    def test_blur_changes_continuously_across_focus_range(self):
        self.assertEqual(BLUR_RADIUS(0.0), 1)
        self.assertEqual(BLUR_RADIUS(0.5), 26)
        self.assertLess(BLUR_RADIUS(0.25), BLUR_RADIUS(0.75))

    def test_focus_is_clamped(self):
        self.assertEqual(BLUR_RADIUS(-1.0), 1)
        self.assertEqual(BLUR_RADIUS(2.0), 52)

    def test_product_color_does_not_leak_into_blurred_background(self):
        image = np.zeros((41, 41, 3), dtype=np.float32)
        image[..., 2] = 100.0
        image[16:25, 16:25] = (255.0, 0.0, 0.0)

        protected = np.zeros((41, 41), dtype=np.uint8)
        protected[13:28, 13:28] = 255
        blurred = BLUR_BACKGROUND(image, protected, blur_radius=6)

        background = protected == 0
        self.assertLess(float(blurred[..., 0][background].max()), 0.01)
        self.assertTrue(
            np.allclose(blurred[..., 2][background], 100.0, atol=0.01)
        )

    def test_blur_is_stronger_near_product_without_a_guard_gap(self):
        product = np.zeros((101, 101), dtype=np.uint8)
        product[45:56, 45:56] = 255

        proximity, falloff = PROXIMITY_WEIGHT(
            product,
            product_focus=0.7,
            minimum_falloff=14,
        )

        self.assertEqual(float(proximity[50, 50]), 0.0)
        self.assertGreater(float(proximity[44, 50]), 0.95)
        self.assertGreater(
            float(proximity[40, 50]),
            float(proximity[20, 50]),
        )
        self.assertGreaterEqual(falloff, 14)

    def test_near_product_uses_larger_blur_radius(self):
        near_radius = BLUR_RADIUS(0.7)
        self.assertLess(FAR_BLUR_RADIUS(near_radius), near_radius)


if __name__ == "__main__":
    unittest.main()
