from __future__ import annotations

import unittest

from PIL import Image, ImageChops, ImageDraw

from adcg.prompt_layout.engine import (
    apply_final_review_revision,
    build_design_layout,
)
from adcg.prompt_layout.generator import _enforce_final_review_revision
from adcg.prompt_layout.renderer import _draw_price_line
from adcg.prompt_layout.schemas import (
    DESIGN_SPEC_SCHEMA,
    FINAL_REVIEW_SCHEMA,
)


def _layout() -> dict:
    return {
        "canvas": {"width": 400, "height": 400},
        "elements": [
            {
                "role": "title",
                "design_group": "headline",
                "x": 40,
                "y": 30,
                "width": 320,
                "height": 50,
                "font_size": 30,
                "font_weight": 700,
                "tracking": 0,
                "text_align": "center",
                "color": "#FFFFFF",
            },
            {
                "role": "subtitle",
                "design_group": "headline",
                "x": 40,
                "y": 80,
                "width": 320,
                "height": 20,
                "font_size": 14,
                "font_weight": 500,
                "tracking": 0,
                "text_align": "center",
                "color": "#FFFFFF",
            },
            {
                "role": "price",
                "design_group": "offer",
                "x": 80,
                "y": 280,
                "width": 240,
                "height": 30,
                "font_size": 24,
                "font_weight": 800,
                "tracking": 0,
                "text_align": "center",
                "color": "#FFFFFF",
                "number_scale": 1.22,
                "unit_scale": 0.80,
            },
            {
                "role": "cta",
                "design_group": "offer",
                "x": 100,
                "y": 320,
                "width": 200,
                "height": 40,
                "font_size": 20,
                "font_weight": 700,
                "tracking": 0,
                "text_align": "center",
                "color": "#FFFFFF",
            },
        ],
        "underlays": [
            {
                "id": "surface-headline",
                "design_group": "headline",
                "x": 0,
                "y": 15,
                "width": 400,
                "height": 100,
                "background_color": "#111111",
                "gradient_color": "#111111",
                "opacity": 0.78,
            },
            {
                "id": "surface-offer",
                "design_group": "offer",
                "x": 0,
                "y": 265,
                "width": 400,
                "height": 110,
                "background_color": "#222222",
                "gradient_color": "#222222",
                "opacity": 0.84,
            },
            {
                "id": "surface-cta",
                "design_group": "offer",
                "x": 100,
                "y": 320,
                "width": 200,
                "height": 40,
                "background_color": "#FF6600",
                "gradient_color": "#FF6600",
                "opacity": 0.96,
            },
            {
                "id": "accent-rule",
                "design_group": "headline",
                "x": 170,
                "y": 75,
                "width": 60,
                "height": 3,
                "background_color": "#FF6600",
                "opacity": 1.0,
            },
        ],
        "design_groups": {
            "headline": {"x": 40, "y": 30},
            "offer": {"x": 100, "y": 320},
        },
        "design_tokens": {
            "palette": {
                "dark": "#111111",
                "light": "#F5F5F5",
                "accent": "#FF6600",
            },
            "headline_alignment": "center",
            "offer_alignment": "center",
            "offer_arrangement": "vertical",
            "cta_treatment": "accent_pill",
            "color_direction": {},
            "headline_band": {},
            "offer_band": {},
        },
    }


def _adjustments() -> dict:
    return {
        "headline_y_shift": 0.01,
        "headline_scale": 1.0,
        "offer_x_shift": 0.0,
        "offer_y_shift": -0.01,
        "offer_scale": 1.0,
        "surface_opacity_delta": 0.05,
        "title_scale": 1.1,
        "subtitle_scale": 1.0,
        "price_scale": 1.0,
        "cta_scale": 0.9,
        "price_number_scale": 0.8,
        "price_unit_scale": 1.2,
        "price_number_baseline_shift": 0.05,
        "price_unit_baseline_shift": -0.05,
        "headline_subtitle_gap_delta": 0.01,
        "price_cta_gap_delta": -0.01,
        "headline_band_height_scale": 0.9,
        "offer_band_height_scale": 0.95,
        "accent_rule_width_scale": 0.5,
        "accent_rule_y_shift": 0.01,
        "headline_weight": "bolder",
        "offer_weight": "lighter",
        "headline_tracking_delta": 2,
        "offer_tracking_delta": 1,
        "offer_alignment": "right",
        "headline_background": "palette_accent",
        "headline_text": "neutral_dark",
        "offer_background": "palette_light",
        "offer_text": "neutral_dark",
        "cta_text": "neutral_light",
    }


class FinalReviewTests(unittest.TestCase):
    def test_final_schema_requires_revision(self):
        needs_revision = FINAL_REVIEW_SCHEMA["properties"]["needs_revision"]
        self.assertEqual(needs_revision["enum"], [True])
        cta_treatment = DESIGN_SPEC_SCHEMA["properties"]["art_direction"][
            "properties"
        ]["cta_treatment"]
        self.assertEqual(cta_treatment["enum"], ["plain"])
        observed = FINAL_REVIEW_SCHEMA["properties"]["diagnosis"][
            "properties"
        ]["observed_problems"]
        self.assertEqual(observed["items"]["type"], "object")
        self.assertEqual(
            set(observed["items"]["required"]),
            {
                "category", "target", "evidence",
                "required_correction", "severity",
            },
        )

    def test_build_layout_never_creates_cta_button(self):
        analysis = {
            "canvas": {"width": 400, "height": 600},
            "palette": {
                "dark": "#111111",
                "light": "#F5F5F5",
                "accent": "#FF6600",
            },
            "overall_luminance": 0.5,
            "horizontal_bands": [],
        }
        design_spec = {
            "scene_analysis": {
                "subject_region": {
                    "x": 0.3,
                    "y": 0.3,
                    "width": 0.4,
                    "height": 0.4,
                }
            },
            "art_direction": {
                "mood": "professional",
                "alignment": "center",
                "spacing_density": "balanced",
                "headline_surface": "full_width_scrim",
                "offer_surface": "full_width_solid",
                "accent_role": "cta",
                "cta_treatment": "accent_pill",
            },
            "color_direction": {
                "headline_background": "palette_dark",
                "headline_text": "neutral_light",
                "offer_background": "palette_dark",
                "offer_text": "neutral_light",
                "cta_background": "palette_accent",
                "cta_text": "neutral_light",
            },
            "composition": {
                "headline_y_ratio": 0.70,
                "offer_y_ratio": 0.08,
                "headline_content_width_ratio": 0.80,
                "offer_content_width_ratio": 0.75,
                "offer_arrangement": "vertical",
                "title_scale": 1.0,
            },
        }
        layout = build_design_layout(
            analysis,
            {"title": "SERVICE", "price": "$10", "cta": "CONTACT"},
            design_spec,
        )
        underlay_ids = {item["id"] for item in layout["underlays"]}
        self.assertNotIn("surface-cta", underlay_ids)
        self.assertEqual(layout["design_tokens"]["cta_treatment"], "plain")

    def test_applies_typography_layout_and_colors(self):
        revision = {
            "needs_revision": True,
            "adjustments": _adjustments(),
            "reason": "Strengthen hierarchy and contrast.",
        }
        result = apply_final_review_revision(_layout(), revision)
        items = {item["role"]: item for item in result["elements"]}
        title = items["title"]
        price = items["price"]
        cta = items["cta"]
        self.assertEqual((title["font_size"], title["font_weight"]), (33, 800))
        self.assertEqual((cta["font_size"], cta["font_weight"]), (18, 600))
        self.assertEqual((title["tracking"], cta["tracking"]), (2, 1))
        self.assertEqual(cta["text_align"], "right")
        self.assertAlmostEqual(price["number_scale"], 0.976)
        self.assertAlmostEqual(price["unit_scale"], 0.96)
        self.assertEqual(price["number_baseline_shift"], 0.05)
        self.assertEqual(price["unit_baseline_shift"], -0.05)
        underlays = {item["id"]: item for item in result["underlays"]}
        self.assertNotIn("surface-cta", underlays)
        self.assertEqual(
            underlays["surface-headline"]["background_color"], "#FF6600"
        )
        self.assertEqual(
            underlays["surface-offer"]["background_color"], "#F5F5F5"
        )
        self.assertEqual(underlays["surface-headline"]["height"], 90)
        self.assertEqual(underlays["surface-offer"]["height"], 104)
        self.assertEqual(underlays["accent-rule"]["width"], 30)

    def test_price_baseline_shift_changes_rendered_pixels(self):
        base_item = {
            "role": "price",
            "x": 0,
            "width": 300,
            "font_size": 24,
            "font_weight": 800,
            "text_align": "left",
            "number_scale": 1.0,
            "unit_scale": 1.0,
        }
        baseline = Image.new("RGB", (300, 80), "white")
        shifted = Image.new("RGB", (300, 80), "white")
        self.assertTrue(
            _draw_price_line(
                ImageDraw.Draw(baseline),
                "10 USD", 10, 40, base_item, None, (0, 0, 0)
            )
        )
        shifted_item = {**base_item, "number_baseline_shift": 0.2}
        self.assertTrue(
            _draw_price_line(
                ImageDraw.Draw(shifted),
                "10 USD", 10, 40, shifted_item, None, (0, 0, 0)
            )
        )
        difference = ImageChops.difference(baseline, shifted)
        self.assertIsNotNone(difference.getbbox())


    def test_neutral_response_gets_visible_fallback(self):
        adjustments = _adjustments()
        adjustments.update(
            {
                "headline_y_shift": 0.0,
                "offer_y_shift": 0.0,
                "surface_opacity_delta": 0.0,
                "title_scale": 1.0,
                "cta_scale": 1.0,
                "price_number_scale": 1.0,
                "price_unit_scale": 1.0,
                "price_number_baseline_shift": 0.0,
                "price_unit_baseline_shift": 0.0,
                "headline_subtitle_gap_delta": 0.0,
                "price_cta_gap_delta": 0.0,
                "headline_band_height_scale": 1.0,
                "offer_band_height_scale": 1.0,
                "accent_rule_width_scale": 1.0,
                "accent_rule_y_shift": 0.0,
                "headline_weight": "keep",
                "offer_weight": "keep",
                "headline_tracking_delta": 0,
                "offer_tracking_delta": 0,
                "offer_alignment": "keep",
                "headline_background": "keep",
                "headline_text": "keep",
                "offer_background": "keep",
                "offer_text": "keep",
                "cta_text": "keep",
            }
        )
        layout = _layout()
        layout["underlays"] = [
            item
            for item in layout["underlays"]
            if item["id"] != "surface-cta"
        ]
        review = _enforce_final_review_revision(
            {
                "needs_revision": False,
                "diagnosis": {
                    "primary_issue": "contrast",
                    "observed_problems": [
                        {
                            "category": "contrast",
                            "target": "headline_band",
                            "evidence": "The headline lacks sufficient contrast.",
                            "required_correction": "Increase surface contrast.",
                            "severity": "high",
                        }
                    ],
                    "correction_summary": "Improve headline contrast.",
                },
                "adjustments": adjustments,
                "reason": "No changes.",
            },
            layout,
        )
        self.assertTrue(review["needs_revision"])
        self.assertTrue(review["revision_enforced"])
        self.assertEqual(review["adjustments"]["surface_opacity_delta"], 0.05)

    def test_price_diagnosis_forces_price_specific_adjustment(self):
        adjustments = _adjustments()
        for field in (
            "price_scale",
            "price_number_scale",
            "price_unit_scale",
        ):
            adjustments[field] = 1.0
        adjustments["price_number_baseline_shift"] = 0.0
        adjustments["price_unit_baseline_shift"] = 0.0
        review = _enforce_final_review_revision(
            {
                "needs_revision": True,
                "diagnosis": {
                    "primary_issue": "hierarchy",
                    "observed_problems": [
                        {
                            "category": "price_composition",
                            "target": "price_number",
                            "evidence": (
                                "The number is disproportionately large relative "
                                "to the surrounding price units."
                            ),
                            "required_correction": (
                                "Reduce the number scale to restore one price line."
                            ),
                            "severity": "high",
                        }
                    ],
                    "correction_summary": "Balance the price typography.",
                },
                "adjustments": adjustments,
                "reason": "Improve hierarchy.",
            },
            _layout(),
        )
        self.assertEqual(review["adjustments"]["price_number_scale"], 0.92)
        self.assertTrue(review["diagnosis_adjustment_enforced"])
        self.assertIn(
            "price_composition", review["enforced_problem_categories"]
        )


if __name__ == "__main__":
    unittest.main()
