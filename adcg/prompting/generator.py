import json
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from adcg.brand_focus import (
    anchor_background_prompts,
    brand_blend_weight,
    select_background_prompt,
)

from .encoding import extract_json, image_to_data_url
from .schema import normalize_prompt_json, normalize_scene_plan_json
from .response_schemas import (
    BRAND_PROMPT_SCHEMA,
    SCENE_PLAN_SCHEMA,
    strict_json_format,
)
from .system_prompt import BRAND_TREATMENT_SYSTEM_PROMPT, SYSTEM_PROMPT


ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")

EVERYDAY_DIRECTION = (
    "maximally ordinary, unstaged, familiar daily-life background"
)
STUDIO_DIRECTION = (
    "maximally luxurious, purpose-built, premium advertising-studio background"
)
GENERIC_SUBJECT_WORDS = {
    "product",
    "object",
    "item",
    "set",
    "machine",
    "vehicle",
    "equipment",
    "foreground",
}


def load_json(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"JSON 파일을 찾을 수 없습니다: {path}"
        )

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def load_preprocess_context(metadata_path):
    if metadata_path is None:
        return {}

    metadata = load_json(metadata_path)

    return {
        "original_size": metadata.get("original_size", {}),
        "product_bbox": metadata.get("product_bbox", {}),
        "trimmed_size": metadata.get("trimmed_size", {}),
        "truncation": metadata.get(
            "truncation",
            {
                "is_truncated": False,
                "touching_edges": [],
            },
        ),
    }


def build_truncation_instruction(preprocess_context):
    truncation = preprocess_context.get("truncation", {})

    if not truncation.get("is_truncated"):
        return (
            "The preprocessing stage did not detect meaningful "
            "foreground contact with the source image boundary."
        )

    touching_edges = truncation.get(
        "touching_edges",
        [],
    )
    edge_text = ", ".join(touching_edges) or "unknown"

    return (
        "The visible foreground touches these source image edges: "
        f"{edge_text}. The source may contain truncated product regions. "
        "Do not invent missing parts. Plan a composition that makes the "
        "visible crop physically plausible and does not expose the missing "
        "region in open background space."
    )


def build_user_instruction(
    product_info,
    product_focus=1.0,
    brand_focus=0.5,
    preprocess_context=None,
):
    preprocess_context = preprocess_context or {}
    product_percent = int(round(product_focus * 100))

    return f"""
Create a scene plan for the supplied foreground product image.

Product focus:
{product_percent}%

Interpret this value as how strongly the final scene should emphasize the product.
Higher values should make the product more visually dominant and the surrounding
background simpler and softer. Lower values may allow a more atmospheric or
blurrier background while preserving product recognition.

Brand-focus endpoint design is performed by a later text-only LLM call. Produce
only a minimal scene reference and do not lock its location, lighting, or camera
treatment. The later stage may redesign those properties while preserving the
visible foreground product evidence.
Product and store metadata:
{json.dumps(product_info, ensure_ascii=False, indent=2)}

Preprocessing context:
{json.dumps(preprocess_context, ensure_ascii=False, indent=2)}

Boundary guidance:
{build_truncation_instruction(preprocess_context)}

Treat all visible foreground objects as one protected commercial product set
when they form a single arrangement. Use the metadata as reference, but rely
on visible image evidence for object identity, count, shape, and color.

Return only the JSON object required by the system instructions.
""".strip()


def build_brand_environment_instruction(scene_plan, product_info):
    return f"""
Create two final background designs from the same product evidence and base
constraints. Incorporate the following directions into concrete, visually
specific prompt parts. The directions are design intents, not literal phrases
to copy into every list item.

everyday_direction:
{EVERYDAY_DIRECTION}

studio_direction:
{STUDIO_DIRECTION}

Make the two results strongly distinguishable while remaining appropriate for
this specific product. Preserve product identity, count, shape, colors, and
physical support requirements. Do not treat the first stage's scene reference
as a fixed location.

First-stage scene plan:
{json.dumps(scene_plan, ensure_ascii=False, indent=2)}

Product and store metadata:
{json.dumps(product_info, ensure_ascii=False, indent=2)}

Return only the JSON object required by the system instructions.
""".strip()


def _normalize_prompt_parts(parts, endpoint_name):
    if not isinstance(parts, list):
        raise ValueError(
            f"2차 브랜드 프롬프트 응답이 목록이 아닙니다: {endpoint_name}"
        )
    normalized = [
        str(part or "").strip().strip(",")
        for part in parts
        if str(part or "").strip().strip(",")
    ]
    if not normalized:
        raise ValueError(
            f"2차 브랜드 프롬프트 응답이 비어 있습니다: {endpoint_name}"
        )
    return normalized


def _subject_match_terms(protected_subject_terms):
    phrases = set()
    tokens = set()
    for raw_term in protected_subject_terms:
        phrase = " ".join(str(raw_term or "").casefold().split())
        if not phrase or phrase in GENERIC_SUBJECT_WORDS:
            continue
        phrases.add(phrase)
        for token in re.findall(r"[\w-]+", phrase, flags=re.UNICODE):
            if len(token) >= 3 and token not in GENERIC_SUBJECT_WORDS:
                tokens.add(token)
    return phrases, tokens


def _filter_subject_prompt_parts(parts, protected_subject_terms):
    phrases, tokens = _subject_match_terms(protected_subject_terms)
    if not phrases and not tokens:
        return list(parts), []

    kept = []
    removed = []
    for part in parts:
        normalized = " ".join(str(part).casefold().split())
        part_tokens = set(
            re.findall(r"[\w-]+", normalized, flags=re.UNICODE)
        )
        contains_subject = any(
            phrase in normalized for phrase in phrases
        ) or bool(part_tokens & tokens)
        (removed if contains_subject else kept).append(part)
    return kept, removed


def _parts_to_prompt(parts):
    return ", ".join(parts)


def run_prompt_generation(
    image_path,
    info_path,
    output_path,
    model="gpt-5.4-nano",
    product_focus=1.0,
    brand_focus=0.5,
    detail="low",
    preprocess_metadata_path=None,
    client=None,
):
    image_path = Path(image_path)
    info_path = Path(info_path)
    output_path = Path(output_path)


    if detail not in {"low", "high"}:
        raise ValueError(
            "detail은 'low' 또는 'high'여야 합니다."
        )

    if not 0.0 <= product_focus <= 1.0:
        raise ValueError(
            "product_focus는 0.0부터 1.0 사이의 값이어야 합니다."
        )

    if not 0.0 <= brand_focus <= 1.0:
        raise ValueError(
            "brand_focus must be between 0.0 and 1.0."
        )

    if not image_path.exists():
        raise FileNotFoundError(
            f"상품 이미지를 찾을 수 없습니다: {image_path}"
        )

    product_info = load_json(info_path)
    preprocess_context = load_preprocess_context(
        preprocess_metadata_path
    )

    if client is None:
        client = OpenAI()

    print("[Prompt LLM 1/2] Product evidence and scene constraints")
    scene_response = client.responses.create(
        model=model,
        instructions=SYSTEM_PROMPT,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": build_user_instruction(
                            product_info=product_info,
                            product_focus=product_focus,
                            brand_focus=brand_focus,
                            preprocess_context=preprocess_context,
                        ),
                    },
                    {
                        "type": "input_image",
                        "image_url": image_to_data_url(
                            image_path
                        ),
                        "detail": detail,
                    },
                ],
            }
        ],
        text=strict_json_format("product_scene_plan", SCENE_PLAN_SCHEMA),
    )

    scene_plan = normalize_scene_plan_json(
        extract_json(scene_response.output_text)
    )
    protected_subject_terms = scene_plan["product_analysis"].get(
        "protected_subject_terms",
        [],
    )
    base_prompt = scene_plan["generation_prompt"][
        "base_background_prompt"
    ]
    safe_base_parts, removed_base_parts = _filter_subject_prompt_parts(
        [
            part.strip()
            for part in base_prompt.split(",")
            if part.strip()
        ],
        protected_subject_terms,
    )
    scene_plan["generation_prompt"]["base_background_prompt"] = (
        ", ".join(safe_base_parts)
        or "physical support, coherent perspective, usable copy space"
    )
    print("[Prompt LLM 2/2] Direction-based background prompt design")
    brand_response = client.responses.create(
        model=model,
        instructions=BRAND_TREATMENT_SYSTEM_PROMPT,
        input=build_brand_environment_instruction(
            scene_plan=scene_plan,
            product_info=product_info,
        ),
        text=strict_json_format(
            "brand_background_prompts",
            BRAND_PROMPT_SCHEMA,
        ),
    )
    brand_prompt_design = extract_json(brand_response.output_text)
    if not isinstance(brand_prompt_design, dict):
        raise ValueError("2차 브랜드 프롬프트 응답이 JSON 객체가 아닙니다.")
    everyday_prompt_parts = _normalize_prompt_parts(
        brand_prompt_design.get("everyday_prompt_parts"),
        "everyday_prompt_parts",
    )
    studio_prompt_parts = _normalize_prompt_parts(
        brand_prompt_design.get("studio_prompt_parts"),
        "studio_prompt_parts",
    )
    everyday_prompt_parts, removed_everyday_parts = (
        _filter_subject_prompt_parts(
            everyday_prompt_parts,
            protected_subject_terms,
        )
    )
    studio_prompt_parts, removed_studio_parts = (
        _filter_subject_prompt_parts(
            studio_prompt_parts,
            protected_subject_terms,
        )
    )
    if not everyday_prompt_parts or not studio_prompt_parts:
        raise ValueError(
            "상품 관련 표현 제거 후 2차 배경 프롬프트가 비었습니다."
        )
    everyday_raw_prompt = _parts_to_prompt(everyday_prompt_parts)
    studio_raw_prompt = _parts_to_prompt(studio_prompt_parts)

    result = normalize_prompt_json({
        "product_analysis": scene_plan["product_analysis"],
        "generation_prompt": {
            "base_background_prompt": scene_plan["generation_prompt"][
                "base_background_prompt"
            ],
            "everyday_background_prompt": everyday_raw_prompt,
            "studio_background_prompt": studio_raw_prompt,
        },
        "layout": scene_plan["layout"],
    })
    generation_prompt = result["generation_prompt"]
    everyday_prompt = generation_prompt[
        "everyday_background_prompt"
    ]
    studio_prompt = generation_prompt[
        "studio_background_prompt"
    ]
    everyday_prompt, studio_prompt = anchor_background_prompts(
        everyday_prompt,
        studio_prompt,
    )
    generation_prompt[
        "everyday_background_prompt"
    ] = everyday_prompt
    generation_prompt[
        "studio_background_prompt"
    ] = studio_prompt
    generation_prompt["everyday_prompt_parts"] = everyday_prompt_parts
    generation_prompt["studio_prompt_parts"] = studio_prompt_parts
    generation_prompt["removed_subject_prompt_parts"] = {
        "base": removed_base_parts,
        "everyday": removed_everyday_parts,
        "studio": removed_studio_parts,
    }
    blend_weight = brand_blend_weight(brand_focus)
    generation_prompt["background_prompt"] = (
        select_background_prompt(
            everyday_prompt,
            studio_prompt,
            brand_focus,
        )
    )
    result["controls"] = {
        "product_focus": product_focus,
        "brand_focus": brand_focus,
        "brand_blend_weight": blend_weight,
    }
    result["prompt_stages"] = {
        "llm_calls": 2,
        "scene_planner_model": model,
        "brand_prompt_model": model,
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    output_path.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[DONE] prompt json saved: {output_path}")

    return output_path
