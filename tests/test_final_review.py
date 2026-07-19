from __future__ import annotations

import inspect
import json
import unittest

from PIL import Image, ImageChops, ImageDraw

from adcg.prompt_layout.engine import apply_final_review_revision
from adcg.prompt_layout.generator import (
    _applied_changes,
    _enforce_final_review_revision,
    _layout_state,
    generate_prompt_layout,
)
from adcg.prompt_layout.renderer import _draw_price_line
from adcg.prompt_layout.schemas import FINAL_REVIEW_FEATURES, FINAL_REVIEW_SCHEMA


def _layout() -> dict:
    return {
        "canvas": {"width": 400, "height": 400},
        "elements": [
            {"role": "title", "design_group": "headline", "content": "TITLE",
             "x": 40, "y": 30, "width": 320, "height": 50,
             "font_size": 30, "font_weight": 700, "tracking": 0,
             "text_align": "center", "max_lines": 1, "color": "#FFFFFF"},
            {"role": "subtitle", "design_group": "headline", "content": "SUB",
             "x": 40, "y": 80, "width": 320, "height": 30,
             "font_size": 14, "font_weight": 500, "tracking": 0,
             "text_align": "center", "max_lines": 2, "color": "#FFFFFF"},
            {"role": "price", "design_group": "offer", "content": "10????",
             "x": 80, "y": 280, "width": 240, "height": 35,
             "font_size": 24, "font_weight": 800, "tracking": 0,
             "text_align": "center", "max_lines": 1, "color": "#FFFFFF",
             "number_scale": 1.22, "unit_scale": 0.80},
            {"role": "cta", "design_group": "offer", "content": "????",
             "x": 100, "y": 325, "width": 200, "height": 30,
             "font_size": 20, "font_weight": 700, "tracking": 0,
             "text_align": "center", "max_lines": 1, "color": "#FFFFFF"},
        ],
        "underlays": [
            {"id": "surface-headline", "design_group": "headline", "x": 0,
             "y": 15, "width": 400, "height": 105,
             "background_color": "#111111", "gradient_color": "#111111",
             "opacity": 0.78},
            {"id": "surface-offer", "design_group": "offer", "x": 0,
             "y": 265, "width": 400, "height": 110,
             "background_color": "#222222", "gradient_color": "#222222",
             "opacity": 0.84},
            {"id": "surface-cta", "design_group": "offer", "x": 100,
             "y": 325, "width": 200, "height": 30,
             "background_color": "#FF6600", "gradient_color": "#FF6600",
             "opacity": 0.96},
            {"id": "accent-rule", "design_group": "headline", "x": 170,
             "y": 75, "width": 60, "height": 3,
             "background_color": "#FF6600", "opacity": 1.0},
        ],
        "design_groups": {
            "headline": {"x": 40, "y": 30, "width": 320, "height": 80},
            "offer": {"x": 80, "y": 280, "width": 240, "height": 75},
        },
        "design_tokens": {
            "palette": {"dark": "#111111", "light": "#F5F5F5", "accent": "#FF6600"},
            "headline_alignment": "center", "offer_alignment": "center",
            "offer_arrangement": "vertical", "cta_treatment": "accent_pill",
            "color_direction": {}, "headline_band": {}, "offer_band": {},
        },
    }


def _feature_reviews() -> dict:
    return {
        feature: {
            "verdict": "revise",
            "evidence": f"Visible issue in {feature}.",
            "recommended_change": f"Correct {feature} with the target state.",
            "affected_targets": ["overall_composition"],
        }
        for feature in FINAL_REVIEW_FEATURES
    }


def _target_layout() -> dict:
    return {
        "elements": [
            {"role": "title", "x": 24, "y": 36, "width": 352, "height": 48,
             "font_size": 34, "font_weight": 800, "tracking": 1,
             "text_align": "center", "max_lines": 1, "color": "neutral_light"},
            {"role": "subtitle", "x": 50, "y": 94, "width": 300, "height": 28,
             "font_size": 15, "font_weight": 500, "tracking": 0,
             "text_align": "center", "max_lines": 2, "color": "neutral_light"},
            {"role": "price", "x": 55, "y": 270, "width": 290, "height": 48,
             "font_size": 28, "font_weight": 800, "tracking": 0,
             "text_align": "center", "max_lines": 1, "color": "neutral_dark"},
            {"role": "cta", "x": 80, "y": 330, "width": 240, "height": 28,
             "font_size": 16, "font_weight": 600, "tracking": 1,
             "text_align": "center", "max_lines": 1, "color": "neutral_dark"},
        ],
        "surfaces": [
            {"group": "headline", "x": 0, "y": 20, "width": 400,
             "height": 120, "opacity": 0.88, "background": "palette_dark",
             "gradient": "palette_dark"},
            {"group": "offer", "x": 0, "y": 250, "width": 400,
             "height": 125, "opacity": 0.90, "background": "palette_light",
             "gradient": "palette_light"},
        ],
        "accent_rule": {"present": True, "x": 145, "y": 87,
                         "width": 110, "height": 4, "color": "palette_accent"},
        "price_composition": {"number_scale": 1.15, "unit_scale": 0.92,
                              "number_baseline_shift": 0.0,
                              "unit_baseline_shift": -0.02},
    }


def _review() -> dict:
    return {
        "needs_revision": True,
        "diagnosis": {
            "primary_issue": "hierarchy",
            "feature_reviews": _feature_reviews(),
            "observed_problems": [{
                "category": "hierarchy", "target": "price_number",
                "evidence": "The number is detached from its unit.",
                "required_correction": "Rebalance number and unit sizes.",
                "severity": "high",
            }],
            "correction_summary": "Apply a coherent absolute target layout.",
        },
        "target_layout": _target_layout(),
        "reason": "Resolve the visible hierarchy and spacing defects.",
    }


class FinalReviewTests(unittest.TestCase):
    def test_pipeline_keeps_exactly_three_vlm_calls(self):
        source = inspect.getsource(generate_prompt_layout)
        self.assertEqual(source.count("_request_json("), 3)

    def test_schema_uses_absolute_target_layout(self):
        self.assertIn("target_layout", FINAL_REVIEW_SCHEMA["properties"])
        self.assertNotIn("adjustments", FINAL_REVIEW_SCHEMA["properties"])
        element = FINAL_REVIEW_SCHEMA["properties"]["target_layout"]["properties"]["elements"]["items"]
        self.assertIn("x", element["properties"])
        self.assertIn("font_size", element["properties"])
        json.dumps(FINAL_REVIEW_SCHEMA)

    def test_absolute_targets_replace_values_without_multiplier_stacking(self):
        result = apply_final_review_revision(_layout(), _review())
        items = {item["role"]: item for item in result["elements"]}
        self.assertEqual(items["title"]["font_size"], 34)
        self.assertEqual((items["title"]["x"], items["title"]["width"]), (24, 352))
        self.assertEqual(items["price"]["number_scale"], 1.15)
        self.assertEqual(items["price"]["unit_scale"], 0.92)
        self.assertEqual(items["price"]["unit_baseline_shift"], -0.02)
        underlays = {item["id"]: item for item in result["underlays"]}
        self.assertNotIn("surface-cta", underlays)
        self.assertEqual(underlays["surface-offer"]["background_color"], "#F5F5F5")
        self.assertEqual(underlays["accent-rule"]["width"], 110)
        self.assertEqual(result["design_tokens"]["cta_treatment"], "plain")

    def test_canvas_constraints_are_recorded(self):
        review = _review()
        review["target_layout"]["elements"][0].update(
            {"x": 390, "y": 390, "width": 500, "height": 500}
        )
        result = apply_final_review_revision(_layout(), review)
        title = next(item for item in result["elements"] if item["role"] == "title")
        self.assertEqual((title["x"], title["y"], title["width"], title["height"]),
                         (0, 0, 400, 400))
        self.assertTrue(result["final_review_constraints"])

    def test_review_requires_every_rendered_role_once(self):
        review = _review()
        review["target_layout"]["elements"].pop()
        with self.assertRaisesRegex(ValueError, "exactly match"):
            _enforce_final_review_revision(review, _layout())

    def test_actual_state_and_changes_are_auditable(self):
        before = _layout_state(_layout())
        after = _layout_state(apply_final_review_revision(_layout(), _review()))
        changes = _applied_changes(before, after)
        targets = {item["target"] for item in changes}
        self.assertTrue({"title", "price", "headline_surface", "accent_rule"}.issubset(targets))

    def test_price_baseline_shift_changes_rendered_pixels(self):
        base_item = {"role": "price", "x": 0, "width": 300, "font_size": 24,
                     "font_weight": 800, "text_align": "left",
                     "number_scale": 1.0, "unit_scale": 1.0}
        baseline = Image.new("RGB", (300, 80), "white")
        shifted = Image.new("RGB", (300, 80), "white")
        self.assertTrue(_draw_price_line(ImageDraw.Draw(baseline), "10 USD", 10, 40,
                                         base_item, None, (0, 0, 0)))
        shifted_item = {**base_item, "number_baseline_shift": 0.2}
        self.assertTrue(_draw_price_line(ImageDraw.Draw(shifted), "10 USD", 10, 40,
                                         shifted_item, None, (0, 0, 0)))
        self.assertIsNotNone(ImageChops.difference(baseline, shifted).getbbox())


if __name__ == "__main__":
    unittest.main()
