import argparse
import base64
import json
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")

PROMPT_SCHEMA_KEYS = {
    "product_analysis",
    "generation_prompt",
    "layout",
}

SYSTEM_PROMPT = SYSTEM_PROMPT = """
You are the scene-planning component of a product-preserving commercialadvertisement generation pipeline.

The attached PNG contains the original foreground subject. The foregroundpixels are protected by an inpainting mask and must remain unchanged.Your output controls only the newly generated surrounding background.
Your task is to:
Identify the visible foreground subject.
Select a commercially plausible environment for that specific subject.
Write a concise Korean caption describing the visible subject.
Write a concise English diffusion prompt describing only the background.
The input may contain any kind of product, equipment, food, object, package,furniture, appliance, artwork, or service-related subject. Treat the subjectcategory as open-ended. Do not select a scene from a fixed list of businesscategories.

Evidence priority:
Use clear visual evidence from the PNG to determine subject identity,physical scale, orientation, camera angle, and likely support surface.
Use product name and product description to disambiguate uncertain visual details.
Use seller and store information to choose a compatible commercial context.
Use mood and additional requests to control atmosphere, materials, color,lighting, and composition.
Use focus ratios only as relative visual priorities.

When information conflicts:
Never contradict clear visual evidence.
Use metadata to resolve ambiguity, not to replace the visible subject.
Do not force the subject into a business environment that is physically orsemantically unrelated to it.
When the category remains uncertain, choose a restrained, realistic,category-compatible commercial setting with minimal secondary elements.

Before answering, internally determine:
what the visible subject is
its approximate real-world scale and function
camera height, viewing angle, horizon, and perspective
the surface or ground that would realistically support it
a commercially relevant indoor or outdoor environment
visible lighting direction, softness, intensity, and color temperature
suitable scene complexity
the safest low-detail region for later advertising copy
Do not reveal this analysis.

Background construction rules:
Generate only the environment outside the protected foreground.
Never regenerate, replace, reshape, move, resize, crop, or duplicate the subject.
Never introduce another item that could be mistaken for the foreground subject.
Do not name or describe the foreground subject in background_prompt.
Choose one coherent environment rather than combining multiple scene concepts.
Match the input camera angle, horizon, perspective, scale, and viewing distance.
Provide a believable continuous ground, floor, tabletop, platform, wall,or other physically appropriate supporting structure.
Use subtle natural grounding and lighting consistent with the visible subject.
Keep the foreground boundary and immediate surrounding area visually simple.
Keep strong edges, props, structural lines, and high-contrast details awayfrom the subject silhouette.
Secondary environmental elements may appear only when they clarify the settingand remain subordinate to the foreground.
Prefer realistic commercial photography over cinematic fantasy or decorative excess.

Focus interpretation:
Product focus increases subject-background separation, simpler surroundings,clearer illumination, and lower local detail.
Brand focus increases atmosphere, material identity, color direction,environmental character, and premium styling.
Sales focus increases polished advertising composition and creates a usefullow-detail area for later copy placement.
Blend these priorities according to their relative values.
Never mention ratios or percentages in the output.

Copy-space rules:
Reserve one natural low-detail region in the upper or side area when composition allows.
Place it away from the foreground silhouette and major perspective lines.
Keep it visually integrated with the environment.
It must not appear as a blank rectangle, signboard, poster, or artificial white area.
Do not generate text inside the copy-space region.

Forbidden scene content:
people, hands, faces, or body parts
duplicates or close substitutes for the foreground subject
floating or physically unsupported elements
conflicting scale, horizon, perspective, lighting, or shadows
clutter or strong edges touching the foreground boundary
text, letters, numbers, logos, labels, signs, prices, or watermarks
cartoon, illustration, CGI, or obvious 3D-render styling unless explicitly requested

new_caption output rules:
Korean only.
Exactly one concise sentence.
Describe only the clearly visible foreground subject.
Prefer observable appearance and category over promotional language.
Do not mention the background.
Do not infer unsupported specifications, quality, origin, or performance.
background_prompt output rules:
English only.
Describe only positive background attributes.
Use one line of concise comma-separated phrases.
Use 20 to 35 words.
Keep the result safely below 70 CLIP tokens.
Use this semantic order:composition, environment, supporting surface, lighting, depth,commercial mood, copy-space location.
Put the most important scene conditions first.
Use one coherent lighting setup and one coherent visual style.
Do not repeat adjectives or scene concepts.
Do not use complete explanatory sentences.
Do not use negative expressions such as "no", "without", or "avoid".
Do not include the foreground subject, output labels, explanations, or prefixes.
Return only the fields required by the provided output schema.

[출력 JSON 형식]
{
  "product_analysis": {
    "objects": [],
    "colors": [],
    "camera_angle": "",
    "visual_features": []
  },
  "generation_prompt": {
    "background_prompt": "",
    "negative_prompt": ""
  },
  "layout": {
    "product_position":"",
    "product_x":,
    "product_y":,
    "product_scale":,
    "headline_position": ""
  }
}
""".strip()

DEFAULT_POSITIVE = (
    "unoccupied commercial product advertising scene, "
    "realistic supporting surface directly beneath the conditioned product, "
    "matching camera perspective and natural lighting"
)

DEFAULT_NEGATIVE = (
    "person, people, woman, man, child, human, face, portrait, head, body, "
    "hands, mannequin, character, floating product, pasted cutout, harsh outline, "
    "black edge, duplicate main product, extra main product, distorted product, "
    "merged objects, text, logo, watermark"
)




def image_to_data_url(image_path):
    image_path = Path(image_path)

    extension = image_path.suffix.lower().lstrip(".")
    mime_types = {
        "jpg": "jpeg",
        "jpeg": "jpeg",
        "png": "png",
        "webp": "webp",
        "gif": "gif",
    }

    if extension not in mime_types:
        raise ValueError(f"지원하지 않는 이미지 형식입니다: {extension}")

    encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:image/{mime_types[extension]};base64,{encoded}"


def extract_json(text):
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")

        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"응답에서 JSON을 찾지 못했습니다:\n{text}")

        return json.loads(text[start:end + 1])


def build_user_instruction(product_info, direction):
    return f"""
다음 상품 이미지와 상품/매장 정보를 분석하여 광고 생성용 JSON을 작성해라.

광고 방향: {direction}

상품/매장 정보:
{json.dumps(product_info, ensure_ascii=False, indent=2)}

입력 이미지에 여러 상품이 하나의 세트로 촬영되어 있다면,
각 상품을 분리하지 말고 전체 세트를 하나의 광고 상품 구성으로 유지해라.

배경은 상품이 자연스럽게 놓일 실제 지지면과 상업 광고에 적합한 환경을
제공해야 하며 인물, 얼굴, 손, 신체는 절대 포함하지 마라.
""".strip()


def append_prompt(original, required):
    original = str(original or "").strip().strip(",")
    required = required.strip().strip(",")

    if not original:
        return required

    return f"{original}, {required}"


def clamp_float(value, minimum, maximum, default):
    try:
        return max(minimum, min(maximum, float(value)))
    except (TypeError, ValueError):
        return default


def normalize_string_list(value):
    if not isinstance(value, list):
        return []

    return [
        str(item).strip()
        for item in value
        if str(item).strip()
    ]


def normalize_prompt_json(data):
    if not isinstance(data, dict):
        raise ValueError("GPT 응답의 최상위 값이 JSON 객체가 아닙니다.")

    for key in PROMPT_SCHEMA_KEYS:
        if key not in data:
            raise ValueError(f"프롬프트 JSON에 필수 키가 없습니다: {key}")

    product_analysis = data.setdefault("product_analysis", {})
    product_analysis["objects"] = normalize_string_list(
        product_analysis.get("objects")
    )
    product_analysis["colors"] = normalize_string_list(
        product_analysis.get("colors")
    )
    product_analysis["visual_features"] = normalize_string_list(
        product_analysis.get("visual_features")
    )
    product_analysis["camera_angle"] = str(
        product_analysis.get("camera_angle", "")
    ).strip()

    generation_prompt = data.setdefault("generation_prompt", {})

    background_prompt = generation_prompt.get("background_prompt", "")
    negative_prompt = generation_prompt.get("negative_prompt", "")

    generation_prompt["background_prompt"] = append_prompt(
        background_prompt,
        DEFAULT_POSITIVE,
    )
    generation_prompt["negative_prompt"] = append_prompt(
        negative_prompt,
        DEFAULT_NEGATIVE,
    )

    layout = data.setdefault("layout", {})
    layout["product_position"] = str(
        layout.get("product_position", "lower_center")
    )
    layout["product_x"] = clamp_float(
        layout.get("product_x"),
        0.15,
        0.85,
        0.50,
    )
    layout["product_y"] = clamp_float(
        layout.get("product_y"),
        0.35,
        0.90,
        0.70,
    )
    layout["product_scale"] = clamp_float(
        layout.get("product_scale"),
        0.25,
        0.60,
        0.44,
    )
    layout["headline_position"] = str(
        layout.get("headline_position", "top_center")
    )

    return data


def run_prompt_generation(
    image_path,
    info_path,
    output_path,
    model="gpt-5.4-nano",
    direction="product_focus",
    detail="low",
):
    image_path = Path(image_path)
    info_path = Path(info_path)
    output_path = Path(output_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"상품 이미지를 찾을 수 없습니다: {image_path}"
        )

    if not info_path.exists():
        raise FileNotFoundError(
            f"상품 정보 JSON을 찾을 수 없습니다: {info_path}"
        )

    product_info = json.loads(
        info_path.read_text(encoding="utf-8")
    )
    image_url = image_to_data_url(image_path)

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
                        ),
                    },
                    {
                        "type": "input_image",
                        "image_url": image_url,
                        "detail": detail,
                    },
                ],
            }
        ],
    )

    result = extract_json(response.output_text)
    result = normalize_prompt_json(result)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[DONE] prompt json saved: {output_path}")
    return output_path
