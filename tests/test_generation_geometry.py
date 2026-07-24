import importlib.util
import sys
import types
import unittest
from pathlib import Path


class Product:
    def __init__(self, width, height):
        self.width = width
        self.height = height


def load_canvas_module():
    rembg = types.ModuleType("rembg")
    rembg.remove = lambda image, **_kwargs: image
    sys.modules.setdefault("rembg", rembg)

    preprocessing = types.ModuleType("adcg.preprocessing")
    preprocessing.__path__ = []
    product_module = types.ModuleType("adcg.preprocessing.product")
    product_module.get_rembg_session = lambda *_args, **_kwargs: None
    sys.modules.setdefault("adcg.preprocessing", preprocessing)
    sys.modules.setdefault("adcg.preprocessing.product", product_module)

    path = (
        Path(__file__).resolve().parents[1]
        / "adcg"
        / "generation"
        / "canvas.py"
    )
    spec = importlib.util.spec_from_file_location("canvas_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CANVAS = load_canvas_module()


class GenerationGeometryTests(unittest.TestCase):
    def test_manual_scale_uses_short_canvas_axis(self):
        product = Product(400, 200)
        portrait = CANVAS._scaled_product_size(
            product, 576, 1024, 0.5
        )
        landscape = CANVAS._scaled_product_size(
            product, 1024, 576, 0.5
        )
        self.assertEqual(portrait, (288, 144))
        self.assertEqual(landscape, (288, 144))

    def test_none_uses_aspect_aware_automatic_fit(self):
        product = Product(400, 200)
        portrait = CANVAS._scaled_product_size(
            product, 576, 1024, None
        )
        landscape = CANVAS._scaled_product_size(
            product, 1024, 576, None
        )
        self.assertEqual(portrait, (449, 224))
        self.assertEqual(landscape, (798, 399))

    def test_large_manual_scale_is_fitted_inside_canvas(self):
        product = Product(400, 200)
        width, height = CANVAS._scaled_product_size(
            product, 576, 1024, 1.3
        )
        self.assertLessEqual(width, int(576 * 0.94))
        self.assertLessEqual(height, int(1024 * 0.86))


if __name__ == "__main__":
    unittest.main()