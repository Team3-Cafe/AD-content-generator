from __future__ import annotations


COPY_ROLES = ("title", "subtitle", "price", "cta")


NORMALIZED_POINT_SCHEMA = {
    "type": "object",
    "properties": {
        "x": {"type": "number", "minimum": 0.04, "maximum": 0.90},
        "y": {"type": "number", "minimum": 0.04, "maximum": 0.90},
    },
    "required": ["x", "y"],
    "additionalProperties": False,
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
                "contrast_mode": {
                    "type": "string",
                    "enum": ["auto", "light", "dark"],
                },
                "headline_surface": {
                    "type": "string",
                    "enum": ["none", "gradient_scrim", "soft_panel"],
                },
                "offer_surface": {
                    "type": "string",
                    "enum": ["none", "gradient_scrim", "solid_lockup"],
                },
                "accent_role": {
                    "type": "string",
                    "enum": ["rule", "price", "cta", "price_and_cta"],
                },
            },
            "required": [
                "mood",
                "alignment",
                "spacing_density",
                "contrast_mode",
                "headline_surface",
                "offer_surface",
                "accent_role",
            ],
            "additionalProperties": False,
        },
        "composition": {
            "type": "object",
            "properties": {
                "headline_anchor": NORMALIZED_POINT_SCHEMA,
                "offer_anchor": NORMALIZED_POINT_SCHEMA,
                "headline_width_ratio": {
                    "type": "number",
                    "minimum": 0.30,
                    "maximum": 0.70,
                },
                "offer_width_ratio": {
                    "type": "number",
                    "minimum": 0.24,
                    "maximum": 0.62,
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
                "headline_anchor",
                "offer_anchor",
                "headline_width_ratio",
                "offer_width_ratio",
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
                "headline_x_shift": {
                    "type": "number",
                    "minimum": -0.08,
                    "maximum": 0.08,
                },
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
                "headline_x_shift",
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


__all__ = [
    "COPY_ROLES",
    "DESIGN_REVISION_SCHEMA",
    "DESIGN_SPEC_SCHEMA",
]
