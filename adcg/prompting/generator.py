import json
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from .encoding import extract_json, image_to_data_url
from .schema import normalize_prompt_json
from .system_prompt import SYSTEM_PROMPT


ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")



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
    brand_percent = int(round(brand_focus * 100))
    everyday_percent = 100 - brand_percent

    return f"""
Create a scene plan for the supplied foreground product image.

Product focus:
{product_percent}%

Interpret this value as how strongly the final scene should emphasize the product.
Higher values should make the product more visually dominant and the surrounding
background simpler and softer. Lower values may allow a more atmospheric or
blurrier background while preserving product recognition.

Brand focus:
{brand_focus:.2f} ({everyday_percent}% authentic everyday environment / {brand_percent}% premium purpose-built set)

Treat brand focus as a continuous background-art-direction value. Do not snap it
to low, medium, or high presets. Use it only to balance environmental authenticity,
staging, contextual prop density, surface refinement, realistic imperfections, and
deliberate negative space. Preserve any explicit desired scene. Do not use brand
focus to prescribe lighting intensity, exposure, brightness, contrast, saturation,
white balance, shadows, highlights, blur, or foreground product appearance.
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
    )

    result = extract_json(response.output_text)
    result = normalize_prompt_json(result)
    result["controls"] = {
        "product_focus": product_focus,
        "brand_focus": brand_focus,
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