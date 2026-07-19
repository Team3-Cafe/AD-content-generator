from __future__ import annotations

COPY_ROLES = ("title", "subtitle", "price", "cta")


COLOR_VALUE_PATTERN = (
    r"^(?:palette_dark|palette_light|palette_accent|#[0-9A-Fa-f]{6})$"
)
COLOR_TOKEN_SCHEMA = {
    "type": "string",
    "pattern": COLOR_VALUE_PATTERN,
    "description": (
        "A supplied palette token or an exact #RRGGBB color chosen for the image."
    ),
}
FINAL_COLOR_SCHEMA = {
    "type": "string",
    "pattern": (
        r"^(?:keep|palette_dark|palette_light|palette_accent|"
        r"#[0-9A-Fa-f]{6})$"
    ),
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
                "headline_alignment": {
                    "type": "string",
                    "enum": ["left", "center", "right"],
                },
                "offer_alignment": {
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
                        "content_solid",
                        "content_gradient",
                        "none",
                    ],
                },
                "offer_surface": {
                    "type": "string",
                    "enum": [
                        "full_width_solid",
                        "full_width_gradient",
                        "accent_band",
                        "content_solid",
                        "content_gradient",
                        "none",
                    ],
                },
                "accent_role": {
                    "type": "string",
                    "enum": ["rule", "price"],
                },
            },
            "required": [
                "mood",
                "headline_alignment",
                "offer_alignment",
                "spacing_density",
                "headline_surface",
                "offer_surface",
                "accent_role",
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
                "cta_text": COLOR_TOKEN_SCHEMA,
            },
            "required": [
                "headline_background",
                "headline_text",
                "offer_background",
                "offer_text",
                "cta_text",
            ],
            "additionalProperties": False,
        },
        "composition": {
            "type": "object",
            "properties": {
                "headline_x_ratio": {
                    "type": "number", "minimum": 0.0, "maximum": 0.95,
                },
                "headline_y_ratio": {
                    "type": "number",
                    "minimum": 0.03,
                    "maximum": 0.72,
                },
                "offer_x_ratio": {
                    "type": "number", "minimum": 0.0, "maximum": 0.95,
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
                "headline_x_ratio",
                "headline_y_ratio",
                "offer_x_ratio",
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


FINAL_REVIEW_FEATURES = (
    "typography", "hierarchy", "spacing", "price_composition",
    "band_proportion", "accent_rule", "placement", "color",
    "contrast", "cta", "product_visibility",
)

FINAL_REVIEW_FEATURE_TARGETS = {
    "typography": [
        "title_typography", "subtitle_typography",
        "price_typography", "cta_typography", "price_composition",
    ],
    "hierarchy": [
        "title_geometry", "title_typography", "subtitle_geometry",
        "subtitle_typography", "price_geometry", "price_typography",
        "cta_geometry", "cta_typography", "overall_composition",
    ],
    "spacing": [
        "title_geometry", "subtitle_geometry", "price_geometry",
        "cta_geometry", "headline_surface", "offer_surface",
        "overall_composition",
    ],
    "price_composition": [
        "price_geometry", "price_typography", "price_composition",
    ],
    "band_proportion": ["headline_surface", "offer_surface"],
    "accent_rule": ["accent_rule"],
    "placement": [
        "title_geometry", "subtitle_geometry", "price_geometry",
        "cta_geometry", "overall_composition",
    ],
    "color": ["color_palette"],
    "contrast": ["color_palette", "headline_surface", "offer_surface"],
    "cta": ["cta_geometry", "cta_typography", "offer_surface"],
    "product_visibility": [
        "title_geometry", "subtitle_geometry", "price_geometry",
        "cta_geometry", "headline_surface", "offer_surface",
        "overall_composition",
    ],
}



def _feature_feedback_schema(feature: str) -> dict:
    return {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": ["keep", "revise"]},
            "evidence": {"type": "string"},
            "recommended_change": {"type": "string"},
            "affected_targets": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": FINAL_REVIEW_FEATURE_TARGETS[feature],
                },
                "maxItems": 8,
            },
        },
        "required": [
            "verdict", "evidence", "recommended_change",
            "affected_targets",
        ],
        "additionalProperties": False,
    }


_ABSOLUTE_ELEMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "role": {"type": "string", "enum": list(COPY_ROLES)},
        "x": {"type": "integer", "minimum": 0, "maximum": 4096},
        "y": {"type": "integer", "minimum": 0, "maximum": 4096},
        "width": {"type": "integer", "minimum": 1, "maximum": 4096},
        "height": {"type": "integer", "minimum": 1, "maximum": 4096},
        "font_size": {"type": "integer", "minimum": 8, "maximum": 256},
        "font_weight": {
            "type": "integer",
            "enum": [300, 400, 500, 600, 700, 800, 900],
        },
        "tracking": {"type": "integer", "minimum": 0, "maximum": 12},
        "text_align": {
            "type": "string", "enum": ["left", "center", "right"],
        },
        "max_lines": {"type": "integer", "minimum": 1, "maximum": 3},
        "color": FINAL_COLOR_SCHEMA,
        "line_height": {"type": "number", "minimum": 0.8, "maximum": 1.8},
        "shadow_offset": {"type": "integer", "minimum": 0, "maximum": 8},
        "shadow_color": FINAL_COLOR_SCHEMA,
        "stroke_width": {"type": "integer", "minimum": 0, "maximum": 6},
        "stroke_color": FINAL_COLOR_SCHEMA,
    },
    "required": [
        "role", "x", "y", "width", "height", "font_size",
        "font_weight", "tracking", "text_align", "max_lines", "color",
        "line_height", "shadow_offset", "shadow_color",
        "stroke_width", "stroke_color",
    ],
    "additionalProperties": False,
}

_ABSOLUTE_SURFACE_SCHEMA = {
    "type": "object",
    "properties": {
        "group": {"type": "string", "enum": ["headline", "offer"]},
        "enabled": {"type": "boolean"},
        "style": {
            "type": "string",
            "enum": ["none", "solid", "gradient", "scrim"],
        },
        "x": {"type": "integer", "minimum": 0, "maximum": 4096},
        "y": {"type": "integer", "minimum": 0, "maximum": 4096},
        "width": {"type": "integer", "minimum": 1, "maximum": 4096},
        "height": {"type": "integer", "minimum": 1, "maximum": 4096},
        "opacity": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "background_color": FINAL_COLOR_SCHEMA,
        "gradient_color": FINAL_COLOR_SCHEMA,
        "corner_radius": {"type": "integer", "minimum": 0, "maximum": 256},
    },
    "required": [
        "group", "enabled", "style", "x", "y", "width", "height",
        "opacity", "background_color", "gradient_color", "corner_radius",
    ],
    "additionalProperties": False,
}


FINAL_REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "needs_revision": {"type": "boolean", "enum": [True]},
        "diagnosis": {
            "type": "object",
            "properties": {
                "primary_issue": {
                    "type": "string", "enum": list(FINAL_REVIEW_FEATURES),
                },
                "feature_reviews": {
                    "type": "object",
                    "properties": {
                        feature: _feature_feedback_schema(feature)
                        for feature in FINAL_REVIEW_FEATURES
                    },
                    "required": list(FINAL_REVIEW_FEATURES),
                    "additionalProperties": False,
                },
                "design_observations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "assessment": {
                                "type": "string",
                                "enum": ["strength", "weakness"],
                            },
                            "category": {
                                "type": "string",
                                "enum": list(FINAL_REVIEW_FEATURES),
                            },
                            "target": {
                                "type": "string",
                                "enum": [
                                    "title", "subtitle", "price_line",
                                    "price_number", "price_unit", "cta",
                                    "headline_group", "offer_group",
                                    "headline_band", "offer_band",
                                    "accent_rule", "product", "background",
                                    "overall",
                                ],
                            },
                            "evidence": {"type": "string"},
                            "design_implication": {"type": "string"},
                            "recommended_action": {
                                "type": "string",
                                "enum": ["preserve", "build_on", "redesign"],
                            },
                            "impact": {
                                "type": "string",
                                "enum": ["low", "medium", "high"],
                            },
                        },
                        "required": [
                            "assessment", "category", "target", "evidence",
                            "design_implication", "recommended_action", "impact",
                        ],
                        "additionalProperties": False,
                    },
                    "minItems": 11,
                    "maxItems": 24,
                },
                "correction_summary": {"type": "string"},
            },
            "required": [
                "primary_issue", "feature_reviews", "design_observations",
                "correction_summary",
            ],
            "additionalProperties": False,
        },
        "redesign_plan": {
            "type": "object",
            "properties": {
                "concept": {"type": "string"},
                "composition_strategy": {"type": "string"},
                "hierarchy_strategy": {"type": "string"},
                "typography_strategy": {"type": "string"},
                "surface_strategy": {"type": "string"},
                "color_strategy": {"type": "string"},
                "product_visibility_strategy": {"type": "string"},
            },
            "required": [
                "concept", "composition_strategy", "hierarchy_strategy",
                "typography_strategy", "surface_strategy",
                "color_strategy", "product_visibility_strategy",
            ],
            "additionalProperties": False,
        },
        "target_layout": {
            "type": "object",
            "properties": {
                "elements": {
                    "type": "array", "items": _ABSOLUTE_ELEMENT_SCHEMA,
                    "minItems": 1, "maxItems": 4,
                },
                "surfaces": {
                    "type": "array", "items": _ABSOLUTE_SURFACE_SCHEMA,
                    "minItems": 0, "maxItems": 2,
                },
                "accent_rule": {
                    "type": "object",
                    "properties": {
                        "present": {"type": "boolean"},
                        "x": {"type": "integer", "minimum": 0, "maximum": 4096},
                        "y": {"type": "integer", "minimum": 0, "maximum": 4096},
                        "width": {"type": "integer", "minimum": 1, "maximum": 4096},
                        "height": {"type": "integer", "minimum": 1, "maximum": 64},
                        "color": FINAL_COLOR_SCHEMA,
                    },
                    "required": [
                        "present", "x", "y", "width", "height", "color",
                    ],
                    "additionalProperties": False,
                },
                "price_composition": {
                    "type": "object",
                    "properties": {
                        "number_scale": {
                            "type": "number", "minimum": 0.7, "maximum": 1.8,
                        },
                        "unit_scale": {
                            "type": "number", "minimum": 0.7, "maximum": 1.4,
                        },
                        "number_baseline_shift": {
                            "type": "number", "minimum": -0.3, "maximum": 0.3,
                        },
                        "unit_baseline_shift": {
                            "type": "number", "minimum": -0.3, "maximum": 0.3,
                        },
                    },
                    "required": [
                        "number_scale", "unit_scale",
                        "number_baseline_shift", "unit_baseline_shift",
                    ],
                    "additionalProperties": False,
                },
            },
            "required": [
                "elements", "surfaces", "accent_rule", "price_composition",
            ],
            "additionalProperties": False,
        },
        "reason": {"type": "string"},
    },
    "required": [
        "needs_revision", "diagnosis", "redesign_plan",
        "target_layout", "reason",
    ],
    "additionalProperties": False,
}


__all__ = [
    "COPY_ROLES",
    "DESIGN_REVISION_SCHEMA",
    "FINAL_REVIEW_SCHEMA",
    "FINAL_REVIEW_FEATURES",
    "FINAL_REVIEW_FEATURE_TARGETS",
    "DESIGN_SPEC_SCHEMA",
]
