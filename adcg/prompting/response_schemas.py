STRING_ARRAY_SCHEMA = {
    "type": "array",
    "items": {"type": "string"},
}

SCENE_PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "product_analysis": {
            "type": "object",
            "properties": {
                "objects": STRING_ARRAY_SCHEMA,
                "protected_subject_terms": STRING_ARRAY_SCHEMA,
                "colors": STRING_ARRAY_SCHEMA,
                "camera_angle": {"type": "string"},
                "visual_features": STRING_ARRAY_SCHEMA,
            },
            "required": [
                "objects",
                "protected_subject_terms",
                "colors",
                "camera_angle",
                "visual_features",
            ],
            "additionalProperties": False,
        },
        "generation_prompt": {
            "type": "object",
            "properties": {
                "base_background_prompt": {"type": "string"},
            },
            "required": ["base_background_prompt"],
            "additionalProperties": False,
        },
        "layout": {
            "type": "object",
            "properties": {
                "product_position": {"type": "string"},
                "product_x": {"type": "number"},
                "product_y": {"type": "number"},
                "product_scale": {"type": "number"},
                "headline_position": {"type": "string"},
            },
            "required": [
                "product_position",
                "product_x",
                "product_y",
                "product_scale",
                "headline_position",
            ],
            "additionalProperties": False,
        },
    },
    "required": ["product_analysis", "generation_prompt", "layout"],
    "additionalProperties": False,
}

BRAND_PROMPT_SCHEMA = {
    "type": "object",
    "properties": {
        "everyday_prompt_parts": STRING_ARRAY_SCHEMA,
        "studio_prompt_parts": STRING_ARRAY_SCHEMA,
    },
    "required": ["everyday_prompt_parts", "studio_prompt_parts"],
    "additionalProperties": False,
}


def strict_json_format(name, schema):
    return {
        "format": {
            "type": "json_schema",
            "name": name,
            "strict": True,
            "schema": schema,
        }
    }


__all__ = [
    "BRAND_PROMPT_SCHEMA",
    "SCENE_PLAN_SCHEMA",
    "strict_json_format",
]
