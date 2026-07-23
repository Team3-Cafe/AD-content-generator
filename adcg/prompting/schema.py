from adcg.negative_prompts import GENERATION_NEGATIVE_PROMPT


PROMPT_SCHEMA_KEYS = {
    "product_analysis",
    "generation_prompt",
    "layout",
}

DEFAULT_POSITIVE = (
    "commercial advertising composition, realistic environment, "
    "physically believable supporting surface, natural lighting, "
    "matching perspective, subtle depth, clean copy space"
)


def clamp_float(value, minimum, maximum, default):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default

    return max(minimum, min(maximum, number))


def normalize_string_list(value):
    if not isinstance(value, list):
        return []

    return [
        str(item).strip()
        for item in value
        if str(item).strip()
    ]


def _normalize_analysis(raw_analysis):
    raw_analysis = raw_analysis if isinstance(raw_analysis, dict) else {}
    return {
        "objects": normalize_string_list(raw_analysis.get("objects")),
        "protected_subject_terms": normalize_string_list(
            raw_analysis.get("protected_subject_terms")
        ),
        "colors": normalize_string_list(raw_analysis.get("colors")),
        "camera_angle": str(raw_analysis.get("camera_angle") or "").strip(),
        "visual_features": normalize_string_list(
            raw_analysis.get("visual_features")
        ),
    }


def _normalize_layout(raw_layout):
    raw_layout = raw_layout if isinstance(raw_layout, dict) else {}
    return {
        "product_position": str(
            raw_layout.get("product_position") or "lower_center"
        ).strip(),
        "product_x": clamp_float(
            raw_layout.get("product_x"), 0.10, 0.90, 0.50
        ),
        "product_y": clamp_float(
            raw_layout.get("product_y"), 0.20, 0.92, 0.70
        ),
        "product_scale": clamp_float(
            raw_layout.get("product_scale"), 0.20, 0.75, 0.45
        ),
        "headline_position": str(
            raw_layout.get("headline_position") or "top_center"
        ).strip(),
    }


def normalize_scene_plan_json(data):
    """Normalize the first, image-aware LLM response."""
    if not isinstance(data, dict):
        raise ValueError("GPT 응답의 최상위 값이 JSON 객체가 아닙니다.")
    for key in PROMPT_SCHEMA_KEYS:
        if key not in data:
            raise ValueError(f"프롬프트 JSON 필수 키가 없습니다: {key}")

    raw_generation = data.get("generation_prompt")
    raw_generation = raw_generation if isinstance(raw_generation, dict) else {}
    base_prompt = str(
        raw_generation.get("base_background_prompt")
        or raw_generation.get("background_prompt")
        or DEFAULT_POSITIVE
    ).strip()
    return {
        "product_analysis": _normalize_analysis(data.get("product_analysis")),
        "generation_prompt": {"base_background_prompt": base_prompt},
        "layout": _normalize_layout(data.get("layout")),
    }


def normalize_prompt_json(data):
    if not isinstance(data, dict):
        raise ValueError(
            "GPT 응답의 최상위 값이 JSON 객체가 아닙니다."
        )

    for key in PROMPT_SCHEMA_KEYS:
        if key not in data:
            raise ValueError(
                f"프롬프트 JSON 필수 키가 없습니다: {key}"
            )

    raw_generation = data.get("generation_prompt")

    if not isinstance(raw_generation, dict):
        raw_generation = {}

    legacy_background_prompt = str(
        raw_generation.get("background_prompt") or ""
    ).strip()
    base_background_prompt = str(
        raw_generation.get("base_background_prompt") or ""
    ).strip()
    everyday_background_prompt = str(
        raw_generation.get("everyday_background_prompt")
        or legacy_background_prompt
        or base_background_prompt
        or DEFAULT_POSITIVE
    ).strip()
    studio_background_prompt = str(
        raw_generation.get("studio_background_prompt")
        or legacy_background_prompt
        or base_background_prompt
        or DEFAULT_POSITIVE
    ).strip()

    negative_prompt = GENERATION_NEGATIVE_PROMPT

    return {
        "product_analysis": _normalize_analysis(data.get("product_analysis")),
        "generation_prompt": {
            "background_prompt": (
                legacy_background_prompt
                or everyday_background_prompt
            ),
            "everyday_background_prompt": everyday_background_prompt,
            "studio_background_prompt": studio_background_prompt,
            "base_background_prompt": base_background_prompt,
            "negative_prompt": negative_prompt,
        },
        "layout": _normalize_layout(data.get("layout")),
    }
