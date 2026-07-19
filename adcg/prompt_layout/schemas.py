from __future__ import annotations
from copy import deepcopy


COPY_ROLES = ("title", "subtitle", "price", "cta")


COLOR_TOKEN_SCHEMA = {
    "type": "string",
    "enum": [
        "palette_dark",
        "palette_light",
        "palette_accent",
        "neutral_dark",
        "neutral_light",
    ],
}


NORMALIZED_BOX_SCHEMA = {
    "type": "object",
    "properties": {
        "x": {"type": "number", "minimum": 0, "maximum": 1},
        "y": {"type": "number", "minimum": 0, "maximum": 1},
        "width": {"type": "number", "minimum": 0.05, "maximum": 1},
        "height": {"type": "number", "minimum": 0.05, "maximum": 1},
    },
    "required": ["x", "y", "width", "height"],
    "additionalProperties": False,
}


DESIGN_SPEC_SCHEMA = {
    "type": "object",
    "properties": {
        "scene_analysis": {
            "type": "object",
            "properties": {
                "composition_summary": {"type": "string"},
                "safe_space_description": {"type": "string"},
                "subject_region": NORMALIZED_BOX_SCHEMA,
                "visual_flow": {"type": "string"},
            },
            "required": [
                "composition_summary",
                "safe_space_description",
                "subject_region",
                "visual_flow",
            ],
            "additionalProperties": False,
        },
        "art_direction": {
            "type": "object",
            "properties": {
                "mood": {
                    "type": "string",
                    "enum": [
                        "professional",
                        "bold",
                        "premium",
                        "friendly",
                        "editorial",
                    ],
                },
                "alignment": {
                    "type": "string",
                    "enum": ["left", "center", "right"],
                },
                "spacing_density": {
                    "type": "string",
                    "enum": ["compact", "balanced", "airy"],
                },
                "headline_surface": {
                    "type": "string",
                    "enum": [
                        "full_width_solid",
                        "full_width_gradient",
                        "full_width_scrim",
                    ],
                },
                "offer_surface": {
                    "type": "string",
                    "enum": [
                        "full_width_solid",
                        "full_width_gradient",
                        "accent_band",
                    ],
                },
                "accent_role": {
                    "type": "string",
                    "enum": ["rule", "price", "cta", "price_and_cta"],
                },
                "cta_treatment": {
                    "type": "string",
                    "enum": ["plain"],
                },
            },
            "required": [
                "mood",
                "alignment",
                "spacing_density",
                "headline_surface",
                "offer_surface",
                "accent_role",
                "cta_treatment",
            ],
            "additionalProperties": False,
        },
        "color_direction": {
            "type": "object",
            "properties": {
                "headline_background": COLOR_TOKEN_SCHEMA,
                "headline_text": COLOR_TOKEN_SCHEMA,
                "offer_background": COLOR_TOKEN_SCHEMA,
                "offer_text": COLOR_TOKEN_SCHEMA,
                "cta_background": COLOR_TOKEN_SCHEMA,
                "cta_text": COLOR_TOKEN_SCHEMA,
            },
            "required": [
                "headline_background",
                "headline_text",
                "offer_background",
                "offer_text",
                "cta_background",
                "cta_text",
            ],
            "additionalProperties": False,
        },
        "composition": {
            "type": "object",
            "properties": {
                "headline_y_ratio": {
                    "type": "number",
                    "minimum": 0.03,
                    "maximum": 0.72,
                },
                "offer_y_ratio": {
                    "type": "number",
                    "minimum": 0.18,
                    "maximum": 0.90,
                },
                "headline_content_width_ratio": {
                    "type": "number",
                    "minimum": 0.56,
                    "maximum": 0.92,
                },
                "offer_content_width_ratio": {
                    "type": "number",
                    "minimum": 0.52,
                    "maximum": 0.92,
                },
                "offer_arrangement": {
                    "type": "string",
                    "enum": ["horizontal", "vertical"],
                },
                "title_scale": {
                    "type": "number",
                    "minimum": 0.80,
                    "maximum": 1.20,
                },
            },
            "required": [
                "headline_y_ratio",
                "offer_y_ratio",
                "headline_content_width_ratio",
                "offer_content_width_ratio",
                "offer_arrangement",
                "title_scale",
            ],
            "additionalProperties": False,
        },
        "rationale": {"type": "string"},
    },
    "required": [
        "scene_analysis",
        "art_direction",
        "color_direction",
        "composition",
        "rationale",
    ],
    "additionalProperties": False,
}


DESIGN_REVISION_SCHEMA = {
    "type": "object",
    "properties": {
        "needs_revision": {"type": "boolean"},
        "adjustments": {
            "type": "object",
            "properties": {
                "headline_y_shift": {
                    "type": "number",
                    "minimum": -0.08,
                    "maximum": 0.08,
                },
                "headline_scale": {
                    "type": "number",
                    "minimum": 0.85,
                    "maximum": 1.15,
                },
                "offer_x_shift": {
                    "type": "number",
                    "minimum": -0.08,
                    "maximum": 0.08,
                },
                "offer_y_shift": {
                    "type": "number",
                    "minimum": -0.08,
                    "maximum": 0.08,
                },
                "offer_scale": {
                    "type": "number",
                    "minimum": 0.85,
                    "maximum": 1.15,
                },
                "surface_opacity_delta": {
                    "type": "number",
                    "minimum": -0.20,
                    "maximum": 0.20,
                },
            },
            "required": [
                "headline_y_shift",
                "headline_scale",
                "offer_x_shift",
                "offer_y_shift",
                "offer_scale",
                "surface_opacity_delta",
            ],
            "additionalProperties": False,
        },
        "reason": {"type": "string"},
    },
    "required": ["needs_revision", "adjustments", "reason"],
    "additionalProperties": False,
}

FINAL_COLOR_TOKENS = [
    "keep",
    "palette_dark",
    "palette_light",
    "palette_accent",
    "neutral_dark",
    "neutral_light",
]


FINAL_REVIEW_SCHEMA = deepcopy(DESIGN_REVISION_SCHEMA)
FINAL_REVIEW_SCHEMA["properties"]["needs_revision"] = {
    "type": "boolean",
    "enum": [True],
}
FINAL_REVIEW_SCHEMA["properties"]["diagnosis"] = {
    "type": "object",
    "properties": {
        "primary_issue": {
            "type": "string",
            "enum": [
                "typography",
                "hierarchy",
                "spacing",
                "price_composition",
                "band_proportion",
                "accent_rule",
                "placement",
                "color",
                "contrast",
                "cta",
                "product_visibility",
            ],
        },
        "observed_problems": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 4,
        },
        "correction_summary": {"type": "string"},
    },
    "required": [
        "primary_issue",
        "observed_problems",
        "correction_summary",
    ],
    "additionalProperties": False,
}
FINAL_REVIEW_SCHEMA["required"].insert(1, "diagnosis")

_final_adjustments = FINAL_REVIEW_SCHEMA["properties"]["adjustments"]
_final_adjustments["properties"].update(
    {
        "title_scale": {"type": "number", "minimum": 0.80, "maximum": 1.20},
        "subtitle_scale": {"type": "number", "minimum": 0.80, "maximum": 1.20},
        "price_scale": {"type": "number", "minimum": 0.80, "maximum": 1.20},
        "cta_scale": {"type": "number", "minimum": 0.80, "maximum": 1.20},
        "price_number_scale": {"type": "number", "minimum": 0.70, "maximum": 1.20},
        "price_unit_scale": {"type": "number", "minimum": 0.80, "maximum": 1.30},
        "price_number_baseline_shift": {
            "type": "number", "minimum": -0.30, "maximum": 0.30,
        },
        "price_unit_baseline_shift": {
            "type": "number", "minimum": -0.30, "maximum": 0.30,
        },
        "headline_subtitle_gap_delta": {
            "type": "number", "minimum": -0.04, "maximum": 0.08,
        },
        "price_cta_gap_delta": {
            "type": "number", "minimum": -0.04, "maximum": 0.08,
        },
        "headline_band_height_scale": {
            "type": "number", "minimum": 0.80, "maximum": 1.20,
        },
        "offer_band_height_scale": {
            "type": "number", "minimum": 0.80, "maximum": 1.20,
        },
        "accent_rule_width_scale": {
            "type": "number", "minimum": 0.50, "maximum": 1.50,
        },
        "accent_rule_y_shift": {
            "type": "number", "minimum": -0.04, "maximum": 0.04,
        },
        "headline_weight": {
            "type": "string",
            "enum": ["keep", "lighter", "bolder"],
        },
        "offer_weight": {
            "type": "string",
            "enum": ["keep", "lighter", "bolder"],
        },
        "headline_tracking_delta": {
            "type": "integer",
            "minimum": -2,
            "maximum": 4,
        },
        "offer_tracking_delta": {
            "type": "integer",
            "minimum": -2,
            "maximum": 4,
        },
        "offer_alignment": {
            "type": "string",
            "enum": ["keep", "left", "center", "right"],
        },
        **{
            name: {"type": "string", "enum": FINAL_COLOR_TOKENS}
            for name in (
                "headline_background",
                "headline_text",
                "offer_background",
                "offer_text",
                "cta_text",
            )
        },
    }
)
_final_adjustments["required"].extend(
    [
        "title_scale",
        "subtitle_scale",
        "price_scale",
        "cta_scale",
        "price_number_scale",
        "price_unit_scale",
        "price_number_baseline_shift",
        "price_unit_baseline_shift",
        "headline_subtitle_gap_delta",
        "price_cta_gap_delta",
        "headline_band_height_scale",
        "offer_band_height_scale",
        "accent_rule_width_scale",
        "accent_rule_y_shift",
        "headline_weight",
        "offer_weight",
        "headline_tracking_delta",
        "offer_tracking_delta",
        "offer_alignment",
        "headline_background",
        "headline_text",
        "offer_background",
        "offer_text",
        "cta_text",
    ]
)



__all__ = [
    "COPY_ROLES",
    "DESIGN_REVISION_SCHEMA",
    "FINAL_REVIEW_SCHEMA",
    "DESIGN_SPEC_SCHEMA",
]
