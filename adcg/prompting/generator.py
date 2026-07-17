import json
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from .encoding import extract_json, image_to_data_url
from .schema import normalize_prompt_json
from .system_prompt import SYSTEM_PROMPT


ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")

DIRECTIONS = {
    "product_focus",
    "brand_focus",
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
    direction,
    preprocess_context=None,
):
    preprocess_context = preprocess_context or {}

    return f"""
Create a scene plan for the supplied foreground product image.

Advertising direction:
{direction}

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


def run_prompt_generation(
    image_path,
    info_path,
    output_path,
    model="gpt-5.4-nano",
    direction="product_focus",
    detail="low",
    preprocess_metadata_path=None,
    client=None,
):
    image_path = Path(image_path)
    info_path = Path(info_path)
    output_path = Path(output_path)

    if direction not in DIRECTIONS:
        raise ValueError(
            f"지원하지 않는 광고 방향입니다: {direction}. "
            f"사용 가능 값: {sorted(DIRECTIONS)}"
        )

    if detail not in {"low", "high"}:
        raise ValueError(
            "detail은 'low' 또는 'high'여야 합니다."
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

    response = client.responses.create(
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
                            direction=direction,
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
    )

    result = extract_json(response.output_text)
    result = normalize_prompt_json(result)

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