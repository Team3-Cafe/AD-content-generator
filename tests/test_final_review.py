from __future__ import annotations

import unittest

from adcg.prompt_layout.engine import apply_final_review_revision
from adcg.prompt_layout.generator import _enforce_final_review_revision
from adcg.prompt_layout.schemas import FINAL_REVIEW_SCHEMA


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
                "y": 20,
                "width": 400,
                "height": 70,
                "background": "#111111",
                "gradient": "#111111",
                "opacity": 0.78,
            },
            {
                "id": "surface-offer",
                "design_group": "offer",
                "x": 0,
                "y": 310,
                "width": 400,
                "height": 70,
                "background": "#222222",
                "gradient": "#222222",
                "opacity": 0.84,
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
        "headline_weight": "bolder",
        "offer_weight": "lighter",
        "headline_tracking_delta": 2,
        "offer_tracking_delta": 1,
        "offer_alignment": "right",
        "headline_background": "palette_accent",
        "headline_text": "neutral_dark",
        "offer_background": "palette_light",
        "offer_text": "neutral_dark",
        "cta_background": "keep",
        "cta_text": "neutral_light",
    }


class FinalReviewTests(unittest.TestCase):
    def test_final_schema_requires_revision(self):
        needs_revision = FINAL_REVIEW_SCHEMA["properties"]["needs_revision"]
        self.assertEqual(needs_revision["enum"], [True])

    def test_applies_typography_layout_and_colors(self):
        revision = {
            "needs_revision": True,
            "adjustments": _adjustments(),
            "reason": "Strengthen hierarchy and contrast.",
        }
        result = apply_final_review_revision(_layout(), revision)
        title, cta = result["elements"]
        self.assertEqual((title["font_size"], title["font_weight"]), (33, 800))
        self.assertEqual((cta["font_size"], cta["font_weight"]), (18, 600))
        self.assertEqual((title["tracking"], cta["tracking"]), (2, 1))
        self.assertEqual(cta["text_align"], "right")
        self.assertEqual(result["underlays"][0]["background"], "#FF6600")
        self.assertEqual(result["underlays"][1]["background"], "#F5F5F5")

    def test_neutral_response_gets_visible_fallback(self):
        adjustments = _adjustments()
        adjustments.update(
            {
                "headline_y_shift": 0.0,
                "offer_y_shift": 0.0,
                "surface_opacity_delta": 0.0,
                "title_scale": 1.0,
                "cta_scale": 1.0,
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
        review = _enforce_final_review_revision(
            {
                "needs_revision": False,
                "adjustments": adjustments,
                "reason": "No changes.",
            },
            _layout(),
        )
        self.assertTrue(review["needs_revision"])
        self.assertTrue(review["revision_enforced"])
        self.assertEqual(review["adjustments"]["surface_opacity_delta"], 0.05)


if __name__ == "__main__":
    unittest.main()
