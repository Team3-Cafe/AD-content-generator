from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image, ImageChops, ImageDraw

from adcg.prompt_layout.analysis import _palette
from adcg.prompt_layout.engine import (
    apply_final_review_revision,
    build_design_layout,
)
from adcg.prompt_layout.generator import (
    _applied_changes,
    _enforce_final_review_revision,
    _layout_state,
    _request_json,
    generate_prompt_layout,
)
from adcg.prompt_layout.prompts import build_final_review_request
from adcg.prompt_layout.renderer import _draw_price_line, _draw_underlays
from adcg.prompt_layout.schemas import FINAL_REVIEW_FEATURES, FINAL_REVIEW_SCHEMA


def _effect(
    colors=("#17324D", "#315B73"),
    *,
    fill_type="linear_gradient",
    opacity=0.82,
    radius=18,
    angle=25.0,
    blur=0,
    blend="normal",
    border=False,
    shadow=False,
):
    return {
        "fill_type": fill_type,
        "fill_colors": list(colors),
        "fill_stops": [
            index / max(1, len(colors) - 1)
            for index in range(len(colors))
        ],
        "gradient_angle": angle, "opacity": opacity,
        "corner_radius": radius, "backdrop_blur": blur,
        "blend_mode": blend, "border_enabled": border,
        "border_color": colors[0], "border_width": 1 if border else 0,
        "border_opacity": 0.5 if border else 0.0,
        "shadow_enabled": shadow, "shadow_color": "#000000",
        "shadow_offset_x": 0, "shadow_offset_y": 5 if shadow else 0,
        "shadow_blur": 12 if shadow else 0,
        "shadow_opacity": 0.22 if shadow else 0.0,
    }


def _render_effect(effect):
    result = dict(effect)
    result["border_radius"] = result.pop("corner_radius")
    return result


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
             **_render_effect(_effect(
                 ("#111111", "#263238"), opacity=0.78, radius=0,
             ))},
            {"id": "surface-offer", "design_group": "offer", "x": 0,
             "y": 265, "width": 400, "height": 110,
             **_render_effect(_effect(
                 ("#222222", "#4A3B2A"), opacity=0.84, radius=0,
             ))},
            {"id": "accent-rule", "design_group": "headline", "x": 170,
             "y": 75, "width": 60, "height": 3,
             **_render_effect(_effect(
                 ("#FF6600", "#FF6600"), fill_type="solid",
                 opacity=1.0, radius=2,
             ))},
        ],
        "design_groups": {
            "headline": {"x": 40, "y": 30, "width": 320, "height": 80},
            "offer": {"x": 80, "y": 280, "width": 240, "height": 75},
        },
        "design_tokens": {
            "palette": {"dark": "#111111", "light": "#F5F5F5", "accent": "#FF6600"},
            "headline_alignment": "center", "offer_alignment": "center",
            "offer_arrangement": "vertical",
            "color_direction": {}, "headline_band": {}, "offer_band": {},
        },
    }


def _feature_reviews() -> dict:
    targets = {
        "typography": ["title_typography"],
        "hierarchy": ["overall_composition"],
        "spacing": ["overall_composition"],
        "price_composition": ["price_composition"],
        "band_proportion": ["headline_surface"],
        "accent_rule": ["accent_rule"],
        "placement": ["overall_composition"],
        "color": ["color_palette"],
        "contrast": ["color_palette"],
        "cta": ["cta_typography"],
        "product_visibility": ["overall_composition"],
    }
    return {
        feature: {
            "verdict": "revise",
            "evidence": f"Visible issue in {feature}.",
            "recommended_change": f"Correct {feature} with the target state.",
            "affected_targets": targets[feature],
        }
        for feature in FINAL_REVIEW_FEATURES
    }


def _target_layout() -> dict:
    elements = [
        {"role": "title", "x": 24, "y": 36, "width": 352, "height": 48,
         "font_size": 34, "font_weight": 800, "tracking": 1,
         "text_align": "left", "max_lines": 1, "color": "#F8F1E5"},
        {"role": "subtitle", "x": 50, "y": 94, "width": 300, "height": 28,
         "font_size": 15, "font_weight": 500, "tracking": 0,
         "text_align": "left", "max_lines": 2, "color": "#F8F1E5"},
        {"role": "price", "x": 55, "y": 270, "width": 290, "height": 48,
         "font_size": 28, "font_weight": 800, "tracking": 0,
         "text_align": "right", "max_lines": 1, "color": "#17324D"},
        {"role": "cta", "x": 80, "y": 330, "width": 240, "height": 28,
         "font_size": 16, "font_weight": 600, "tracking": 1,
         "text_align": "right", "max_lines": 1, "color": "#17324D"},
    ]
    for item in elements:
        item.update({
            "line_height": 1.1, "shadow_offset": 0,
            "shadow_color": "#000000", "stroke_width": 0,
            "stroke_color": "#000000",
        })
    return {
        "elements": elements,
        "surfaces": [
            {
                "group": "headline", "enabled": True,
                "x": 12, "y": 20, "width": 376, "height": 120,
                "effect": _effect(
                    ("#17324D", "#315B73"), angle=32, border=True, shadow=True,
                ),
            },
            {
                "group": "offer", "enabled": False,
                "x": 0, "y": 250, "width": 400, "height": 125,
                "effect": _effect(
                    ("#F8F1E5", "#F8F1E5"), fill_type="solid",
                    opacity=0.0, radius=0,
                ),
            },
        ],
        "accent_rule": {"present": True, "x": 145, "y": 87,
                         "width": 110, "height": 4, "color": "#D9822B"},
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
            "design_observations": [
                {
                    "assessment": (
                        "strength" if index in {1, 8} else "weakness"
                    ),
                    "category": category, "target": target,
                    "evidence": f"Distinct visible observation {index} in {target}.",
                    "design_implication": f"Use observation {index} in the rebuild.",
                    "recommended_action": (
                        "build_on" if index in {1, 8} else "redesign"
                    ),
                    "impact": "high" if index < 3 else "medium",
                }
                for index, (category, target) in enumerate((
                    ("typography", "title"),
                    ("hierarchy", "headline_group"),
                    ("spacing", "offer_group"),
                    ("price_composition", "price_number"),
                    ("band_proportion", "headline_band"),
                    ("accent_rule", "accent_rule"),
                    ("placement", "offer_group"),
                    ("color", "offer_band"),
                    ("contrast", "title"),
                    ("cta", "cta"),
                    ("product_visibility", "product"),
                ))
            ],
            "correction_summary": "Apply a coherent absolute target layout.",
        },
        "redesign_plan": {
            "concept": "Rebuilt industrial editorial layout.",
            "composition_strategy": "Recompose both copy groups.",
            "hierarchy_strategy": "Balance title and price emphasis.",
            "typography_strategy": "Rebuild Korean type scale and rhythm.",
            "surface_strategy": "Resize both editorial bands.",
            "color_strategy": "Use a light offer band with dark type.",
            "product_visibility_strategy": "Keep bands outside the product body.",
        },
        "target_layout": _target_layout(),
        "reason": "Resolve the visible hierarchy and spacing defects.",
    }


class FinalReviewTests(unittest.TestCase):
    def test_keep_feedback_targets_are_normalized_without_aborting(self):
        review = _review()
        review["diagnosis"]["feature_reviews"]["product_visibility"] = {
            "verdict": "keep",
            "evidence": "The product remains unobstructed.",
            "recommended_change": "Preserve product visibility.",
            "affected_targets": ["overall_composition"],
        }

        result = _enforce_final_review_revision(review, _layout())

        feedback = result["diagnosis"]["feature_reviews"][
            "product_visibility"
        ]
        self.assertEqual(feedback["affected_targets"], [])
        self.assertEqual(
            result["feedback_normalizations"][0]["feature"],
            "product_visibility",
        )
        self.assertTrue(result["consistency_validated"])

    def test_feedback_target_mismatch_warns_without_rejecting_redesign(self):
        review = _review()
        original_rule = next(
            item for item in _layout()["underlays"]
            if item["id"] == "accent-rule"
        )
        review["target_layout"]["accent_rule"] = {
            "present": True,
            "x": original_rule["x"],
            "y": original_rule["y"],
            "width": original_rule["width"],
            "height": original_rule["height"],
            "color": "keep",
        }

        result = _enforce_final_review_revision(review, _layout())

        self.assertTrue(result["consistency_validated"])
        self.assertIn(
            "accent_rule claims targets with no material applied change",
            result["material_feedback_warnings"],
        )

    def test_missing_audit_categories_warn_without_rejecting_target(self):
        review = _review()
        review["diagnosis"]["design_observations"] = [
            item
            for item in review["diagnosis"]["design_observations"]
            if item["category"] not in {"accent_rule", "cta"}
        ]

        result = _enforce_final_review_revision(review, _layout())

        self.assertTrue(result["consistency_validated"])
        self.assertTrue(any(
            "accent_rule, cta" in warning
            for warning in result["audit_warnings"]
        ))

    def test_feedback_for_an_absent_cta_is_normalized_to_keep(self):
        layout = _layout()
        layout["elements"] = [
            item for item in layout["elements"] if item["role"] != "cta"
        ]
        review = _review()
        review["target_layout"]["elements"] = [
            item
            for item in review["target_layout"]["elements"]
            if item["role"] != "cta"
        ]
        review["diagnosis"]["feature_reviews"]["cta"] = {
            "verdict": "revise",
            "evidence": "A CTA treatment issue was reported.",
            "recommended_change": "Change the CTA treatment.",
            "affected_targets": ["cta_typography"],
        }

        result = _enforce_final_review_revision(review, layout)

        feedback = result["diagnosis"]["feature_reviews"]["cta"]
        self.assertEqual(feedback["verdict"], "keep")
        self.assertEqual(feedback["affected_targets"], [])
        self.assertTrue(any(
            item["feature"] == "cta"
            and "not rendered" in item["reason"]
            for item in result["feedback_normalizations"]
        ))

    def test_initial_layout_respects_independent_alignment_and_no_surfaces(self):
        analysis = {
            "canvas": {"width": 400, "height": 600},
            "palette": {
                "dark": "#111111", "light": "#F5F5F5",
                "accent": "#FF6600",
            },
            "overall_luminance": 0.5,
            "horizontal_bands": [],
        }
        design_spec = {
            "scene_analysis": {
                "subject_region": {
                    "x": 0.45, "y": 0.35, "width": 0.4, "height": 0.4,
                }
            },
            "art_direction": {
                "mood": "editorial",
                "headline_alignment": "left",
                "offer_alignment": "right",
                "spacing_density": "balanced",
                "headline_surface": "none",
                "offer_surface": "none",
                "accent_role": "rule",
                "headline_effect": _effect(
                    ("#23405A", "#315B73"), angle=45,
                ),
                "offer_effect": _effect(
                    ("#D9B66F", "#F1E7D2"), angle=120,
                ),
            },
            "color_direction": {
                "headline_text": "#F1E7D2",
                "offer_text": "#17324D",
                "cta_text": "#000080",
            },
            "composition": {
                "headline_x_ratio": 0.08, "headline_y_ratio": 0.08,
                "offer_x_ratio": 0.18, "offer_y_ratio": 0.75,
                "headline_content_width_ratio": 0.70,
                "offer_content_width_ratio": 0.62,
                "offer_arrangement": "vertical", "title_scale": 1.0,
            },
        }
        layout = build_design_layout(
            analysis,
            {"title": "TITLE", "subtitle": "SUB", "price": "$10", "cta": ""},
            design_spec,
        )
        items = {item["role"]: item for item in layout["elements"]}
        self.assertEqual(items["title"]["text_align"], "left")
        self.assertEqual(items["price"]["text_align"], "right")
        self.assertNotIn("cta", items)
        surface_ids = {
            item["id"] for item in layout["underlays"]
            if item["id"].startswith("surface-")
        }
        self.assertEqual(surface_ids, set())

        with_cta = build_design_layout(
            analysis,
            {
                "title": "TITLE", "subtitle": "SUB", "price": "$10",
                "cta": "Call 02-123-4567",
            },
            design_spec,
        )
        with_cta_items = {
            item["role"]: item for item in with_cta["elements"]
        }
        self.assertNotEqual(
            with_cta_items["cta"]["color"],
            with_cta_items["price"]["color"],
        )

        price_accent_spec = json.loads(json.dumps(design_spec))
        price_accent_spec["art_direction"]["accent_role"] = "price"
        price_accent_layout = build_design_layout(
            analysis,
            {"title": "TITLE", "subtitle": "SUB", "price": "$10", "cta": ""},
            price_accent_spec,
        )
        self.assertNotIn(
            "accent-rule",
            {item["id"] for item in price_accent_layout["underlays"]},
        )

    def test_final_request_excludes_previous_design_state(self):
        request = build_final_review_request(
            {"title": "TITLE", "price": "10????"},
            {
                "canvas": {"width": 400, "height": 400},
                "palette": {
                    "dark": "#111111", "light": "#F5F5F5",
                    "accent": "#FF6600",
                },
            },
        )
        self.assertIn("Exact copy strings to preserve", request)
        self.assertIn("copy_roles", request)
        self.assertNotIn("font_size", request)
        self.assertNotIn("surface-headline", request)
        self.assertNotIn("Art direction", request)
        self.assertNotIn("design_revision", request)

    def test_final_vlm_request_can_include_completed_and_clean_images(self):
        captured = {}

        class Responses:
            def create(self, **kwargs):
                captured.update(kwargs)
                return SimpleNamespace(output_text="{}")

        client = SimpleNamespace(responses=Responses())
        with patch(
            "adcg.prompt_layout.generator.image_to_data_url",
            side_effect=lambda path: f"data:image/png;base64,{Path(path).stem}",
        ):
            result = _request_json(
                client, model="gpt-4o", instructions="review",
                request_text="rebuild",
                image_path=[Path("completed.png"), Path("clean.png")],
                detail="high", schema_name="test", schema={},
                temperature=0.4,
            )
        self.assertEqual(result, {})
        images = [
            item for item in captured["input"][0]["content"]
            if item["type"] == "input_image"
        ]
        self.assertEqual(len(images), 2)
        self.assertTrue(images[0]["image_url"].endswith("completed"))
        self.assertTrue(images[1]["image_url"].endswith("clean"))

    def test_pipeline_keeps_exactly_three_vlm_calls(self):
        source = inspect.getsource(generate_prompt_layout)
        self.assertEqual(source.count("_request_json("), 3)

    def test_schema_uses_absolute_target_layout(self):
        self.assertIn("target_layout", FINAL_REVIEW_SCHEMA["properties"])
        self.assertIn("redesign_plan", FINAL_REVIEW_SCHEMA["properties"])
        self.assertNotIn("adjustments", FINAL_REVIEW_SCHEMA["properties"])
        observations = FINAL_REVIEW_SCHEMA["properties"]["diagnosis"]["properties"]["design_observations"]
        self.assertEqual(
            (observations["minItems"], observations["maxItems"]),
            (11, 24),
        )
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
        self.assertNotIn("surface-offer", underlays)
        self.assertEqual(
            underlays["surface-headline"]["fill_colors"][0], "#17324D"
        )
        self.assertEqual(items["price"]["text_align"], "right")
        self.assertEqual(underlays["accent-rule"]["width"], 110)

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


    def test_visually_equivalent_target_is_rejected(self):
        layout = _layout()
        review = _review()
        review["target_layout"]["elements"] = [
            {
                "role": item["role"], "x": item["x"], "y": item["y"],
                "width": item["width"], "height": item["height"],
                "font_size": item["font_size"],
                "font_weight": item["font_weight"],
                "tracking": item["tracking"],
                "text_align": item["text_align"],
                "max_lines": item["max_lines"], "color": "keep",
                "line_height": item.get("line_height", 1.2),
                "shadow_offset": item.get("shadow_offset", 0),
                "shadow_color": "keep",
                "stroke_width": item.get("stroke_width", 0),
                "stroke_color": "keep",
            }
            for item in layout["elements"]
        ]
        review["target_layout"]["surfaces"] = [
            {
                "group": item["design_group"], "x": item["x"],
                "y": item["y"], "width": item["width"],
                "height": item["height"], "enabled": True,
                "effect": {
                    **_effect(("keep", "keep"),
                              fill_type=item.get("fill_type", "solid"),
                              opacity=item["opacity"],
                              radius=item.get("border_radius", 0),
                              angle=item.get("gradient_angle", 0.0)),
                    "fill_stops": list(item.get("fill_stops", [0.0, 1.0])),
                    "backdrop_blur": item.get("backdrop_blur", 0),
                    "blend_mode": item.get("blend_mode", "normal"),
                    "border_enabled": item.get("border_enabled", False),
                    "border_width": item.get("border_width", 0),
                    "border_opacity": item.get("border_opacity", 0.0),
                    "shadow_enabled": item.get("shadow_enabled", False),
                    "shadow_offset_x": item.get("shadow_offset_x", 0),
                    "shadow_offset_y": item.get("shadow_offset_y", 0),
                    "shadow_blur": item.get("shadow_blur", 0),
                    "shadow_opacity": item.get("shadow_opacity", 0.0),
                },
            }
            for item in layout["underlays"]
            if item["id"] in {"surface-headline", "surface-offer"}
        ]
        rule = next(item for item in layout["underlays"] if item["id"] == "accent-rule")
        review["target_layout"]["accent_rule"] = {
            "present": True, "x": rule["x"], "y": rule["y"],
            "width": rule["width"], "height": rule["height"],
            "color": "keep",
        }
        review["target_layout"]["price_composition"] = {
            "number_scale": 1.22, "unit_scale": 0.80,
            "number_baseline_shift": 0.0, "unit_baseline_shift": 0.0,
        }
        with self.assertRaisesRegex(ValueError, "material redesign"):
            _enforce_final_review_revision(review, layout)

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

    def test_image_palette_exposes_diverse_surface_swatches(self):
        image = Image.new("RGB", (80, 40), "#14283C")
        draw = ImageDraw.Draw(image)
        draw.rectangle((20, 0, 39, 39), fill="#D88724")
        draw.rectangle((40, 0, 59, 39), fill="#E8E0D0")
        draw.rectangle((60, 0, 79, 39), fill="#466B5A")
        palette = _palette(image)
        self.assertGreaterEqual(len(palette["swatches"]), 4)
        self.assertIn(palette["accent"], palette["swatches"] or [palette["accent"]])

    def test_composable_surface_effects_change_rendered_pixels(self):
        background = Image.new("RGB", (220, 160), "#B8C4CC")
        ImageDraw.Draw(background).rectangle(
            (0, 70, 219, 90), fill="#334455"
        )
        plain = {
            "id": "surface-test", "x": 30, "y": 30,
            "width": 150, "height": 90, "z_index": 0,
            **_render_effect(_effect(
                ("#336699", "#CC8844"), angle=0, radius=16,
            )),
        }
        rich = {
            **plain,
            **_render_effect(_effect(
                ("#163A5F", "#D58A32", "#F2E4C8"),
                fill_type="radial_gradient", angle=135, radius=24,
                blur=7, blend="multiply", border=True, shadow=True,
            )),
        }
        plain_image = _draw_underlays(background, [plain])
        rich_image = _draw_underlays(background, [rich])
        self.assertIsNotNone(
            ImageChops.difference(
                plain_image.convert("RGB"), rich_image.convert("RGB")
            ).getbbox()
        )
        self.assertNotEqual(
            plain_image.getpixel((35, 35)), rich_image.getpixel((35, 35))
        )
        variants = []
        for mode, angle in (("normal", 0), ("normal", 90),
                            ("screen", 45), ("overlay", 135)):
            effect = _render_effect(_effect(
                ("#163A5F", "#D58A32", "#F2E4C8"),
                angle=angle, blend=mode, border=True, shadow=True,
            ))
            variants.append(_draw_underlays(background, [{
                **plain, **effect,
            }]).convert("RGB").tobytes())
        self.assertEqual(len(set(variants)), len(variants))

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
