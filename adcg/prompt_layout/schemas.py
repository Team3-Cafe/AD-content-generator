from __future__ import annotations

from copy import deepcopy


COPY_ROLES = ("title", "subtitle", "price", "cta")


def _bbox_properties() -> dict:
    return {
        "x": {"type": "integer", "minimum": 0},
        "y": {"type": "integer", "minimum": 0},
        "width": {"type": "integer", "minimum": 1},
        "height": {"type": "integer", "minimum": 1},
    }


BBOX_SCHEMA = {
    "type": "object",
    "properties": _bbox_properties(),
    "required": ["x", "y", "width", "height"],
    "additionalProperties": False,
}

REGION_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string"},
        "importance": {
            "type": "string",
            "enum": ["critical", "supporting", "background"],
        },
        "bbox": deepcopy(BBOX_SCHEMA),
        "reason": {"type": "string"},
    },
    "required": ["label", "importance", "bbox", "reason"],
    "additionalProperties": False,
}

PLAN_ELEMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "role": {"type": "string", "enum": list(COPY_ROLES)},
        "content": {"type": "string"},
        "priority": {"type": "integer", "minimum": 1, "maximum": 4},
        "preferred_zone": {
            "type": "string",
            "enum": [
                "top_left",
                "top_center",
                "top_right",
                "middle_left",
                "middle_center",
                "middle_right",
                "bottom_left",
                "bottom_center",
                "bottom_right",
            ],
        },
        "alignment": {
            "type": "string",
            "enum": ["left", "center", "right"],
        },
        "relationship": {"type": "string"},
        "rationale": {"type": "string"},
    },
    "required": [
        "role",
        "content",
        "priority",
        "preferred_zone",
        "alignment",
        "relationship",
        "rationale",
    ],
    "additionalProperties": False,
}

PLACEMENT_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "scene_summary": {"type": "string"},
        "protected_regions": {
            "type": "array",
            "items": deepcopy(REGION_SCHEMA),
        },
        "safe_regions": {
            "type": "array",
            "items": deepcopy(REGION_SCHEMA),
        },
        "placement_plan": {"type": "string"},
        "elements": {
            "type": "array",
            "items": deepcopy(PLAN_ELEMENT_SCHEMA),
        },
        "visual_strategy": {
            "type": "object",
            "properties": {
                "text_color": {"type": "string"},
                "accent_color": {"type": "string"},
                "underlay_recommended": {"type": "boolean"},
                "underlay_color": {"type": "string"},
                "underlay_opacity": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                },
            },
            "required": [
                "text_color",
                "accent_color",
                "underlay_recommended",
                "underlay_color",
                "underlay_opacity",
            ],
            "additionalProperties": False,
        },
        "risk_notes": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": [
        "scene_summary",
        "protected_regions",
        "safe_regions",
        "placement_plan",
        "elements",
        "visual_strategy",
        "risk_notes",
    ],
    "additionalProperties": False,
}

LAYOUT_ELEMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "role": {"type": "string", "enum": list(COPY_ROLES)},
        "content": {"type": "string"},
        **_bbox_properties(),
        "z_index": {"type": "integer", "minimum": 1},
        "text_align": {
            "type": "string",
            "enum": ["left", "center", "right"],
        },
        "vertical_align": {
            "type": "string",
            "enum": ["top", "center", "bottom"],
        },
        "font_size": {"type": "integer", "minimum": 8},
        "font_weight": {"type": "integer", "minimum": 100, "maximum": 900},
        "line_height": {"type": "number", "minimum": 0.8, "maximum": 2.0},
        "color": {"type": "string"},
        "max_lines": {"type": "integer", "minimum": 1, "maximum": 4},
    },
    "required": [
        "id",
        "role",
        "content",
        "x",
        "y",
        "width",
        "height",
        "z_index",
        "text_align",
        "vertical_align",
        "font_size",
        "font_weight",
        "line_height",
        "color",
        "max_lines",
    ],
    "additionalProperties": False,
}

UNDERLAY_SCHEMA = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "target_ids": {
            "type": "array",
            "items": {"type": "string"},
        },
        **_bbox_properties(),
        "z_index": {"type": "integer", "minimum": 0},
        "background_color": {"type": "string"},
        "opacity": {"type": "number", "minimum": 0, "maximum": 1},
        "border_radius": {"type": "integer", "minimum": 0},
    },
    "required": [
        "id",
        "target_ids",
        "x",
        "y",
        "width",
        "height",
        "z_index",
        "background_color",
        "opacity",
        "border_radius",
    ],
    "additionalProperties": False,
}

LAYOUT_SCHEMA = {
    "type": "object",
    "properties": {
        "canvas": {
            "type": "object",
            "properties": {
                "width": {"type": "integer", "minimum": 1},
                "height": {"type": "integer", "minimum": 1},
            },
            "required": ["width", "height"],
            "additionalProperties": False,
        },
        "elements": {
            "type": "array",
            "items": deepcopy(LAYOUT_ELEMENT_SCHEMA),
        },
        "underlays": {
            "type": "array",
            "items": deepcopy(UNDERLAY_SCHEMA),
        },
        "rationale": {"type": "string"},
        "warnings": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["canvas", "elements", "underlays", "rationale", "warnings"],
    "additionalProperties": False,
}
