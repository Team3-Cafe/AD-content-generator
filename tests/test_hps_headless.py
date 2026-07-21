import sys
import unittest
from unittest import mock

from adcg.eval.eval_HPS_v2 import _headless_hpsv2_import_compatibility


class HeadlessHpsCompatibilityTests(unittest.TestCase):
    def test_uses_temporary_turtle_stub_without_tkinter(self):
        existing_turtle = sys.modules.pop("turtle", None)
        try:
            with mock.patch(
                "adcg.eval.eval_HPS_v2.importlib.import_module",
                side_effect=ModuleNotFoundError(
                    "No module named 'tkinter'"
                ),
            ):
                with _headless_hpsv2_import_compatibility():
                    from turtle import forward

                    self.assertTrue(callable(forward))
                    self.assertIn("turtle", sys.modules)

                self.assertNotIn("turtle", sys.modules)
        finally:
            if existing_turtle is not None:
                sys.modules["turtle"] = existing_turtle


if __name__ == "__main__":
    unittest.main()
