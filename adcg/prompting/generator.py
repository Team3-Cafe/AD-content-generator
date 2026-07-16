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
    "ad_copies",
    "generation_prompt",
    "layout",
}

SYSTEM_PROMPT = """
너는 소상공인 광고 이미지 생성 파이프라인의 프롬프트 설계자다.
입력 상품 이미지와 상품/매장 정보를 분석해 이미지 생성용 JSON을 작성한다.

[상품 분석 규칙]
- 이미지에서 실제로 확인되는 상품과 특징만 분석한다.
- 보이지 않는 브랜드, 재료, 효능, 가격은 추측하지 않는다.
- 여러 물체가 하나의 판매 세트라면 전체를 하나의 주 상품 세트로 취급한다.
- 상품의 개수, 형태, 색상, 재질, 배치와 카메라 각도를 기록한다.
- 원본 배경은 상품이 아니므로 product_analysis.objects에 포함하지 않는다.

[배경 프롬프트 규칙]
- background_prompt는 영어로 작성한다.
- 광고 상품을 새로 묘사하거나 복제하지 말고, 상품이 놓일 환경을 중심으로 작성한다.
- 입력 상품과 일치하는 카메라 각도와 원근감을 사용한다.
- 상품 바로 아래에 테이블, 카운터, 선반, 받침대 등의 실제 지지면이 있어야 한다.
- 상품이 공중에 뜨거나 배경 위에 붙은 것처럼 보이면 안 된다.
- 상품 주변의 조명 방향, 그림자, 색온도가 배경과 자연스럽게 이어져야 한다.
- 주 상품과 경쟁하는 크고 선명한 중복 상품을 배경에 생성하지 않는다.
- 광고 문구를 넣을 여백은 확보하되 상품을 지나치게 작게 배치하지 않는다.
- background_prompt는 중요한 조건부터 작성하며 55단어 이내로 제한한다.
- 배경 프롬프트는 장소, 받침면, 조명, 카메라 각도, 광고 여백만 설명한다.
- negative_prompt에는 입력 이미지에서 발견한 객체의 중복 생성을 막는 영어 단어를 포함한다.
- product_focus에서는 상품이 이미지 너비의 약 55~70%를 차지하도록 배치한다.
- 상품이 3개 이상의 묶음 세트라면 product_scale을 0.55~0.68로 설정한다.
- 배경보다 상품이 먼저 시선을 끄는 근접 광고 구도를 사용한다.

[금지 사항]
- 상품을 멀리 있는 가구 위에 작게 배치하지 않는다.
- 사용자가 요청하지 않은 인물, 얼굴, 손, 신체, 캐릭터를 생성하지 않는다.
- 특히 portrait, woman, man, face, head, mannequin이 등장하면 안 된다.
- 텍스트, 로고, 워터마크를 생성하지 않는다.
- 상품을 사람의 얼굴이나 신체 일부처럼 배치하지 않는다.
- 상품을 훼손하거나 서로 합쳐 새로운 물체로 만들지 않는다.
- background_prompt에는 입력 상품의 종류나 이름을 작성하지 않는다.
- 상품, 음료, 음식, 병, 컵 등 입력 객체를 배경 프롬프트에서 다시 묘사하지 않는다.
- 입력에 없는 소품, 과일, 장식, 용기, 받침대를 추가하지 않는다.

[광고 방향]
- product_focus: 상품을 가장 크고 선명한 주인공으로 배치한다.
- brand_focus: 상품은 유지하면서 매장 분위기와 브랜드 무드를 함께 강조한다.

[출력 규칙]
- 반드시 유효한 JSON 하나만 출력한다.
- 설명, Markdown, 코드 블록은 출력하지 않는다.
- 광고 문구는 한국어로 작성한다.
- background_prompt와 negative_prompt는 영어로 작성한다.
- 좌표는 0.0~1.0 사이의 정규화된 값으로 작성한다.
- product_scale은 일반적으로 0.32~0.55 범위로 작성한다.

[출력 JSON 형식]
{
  "product_analysis": {
    "objects": [],
    "colors": [],
    "camera_angle": "",
    "visual_features": []
  },
  "ad_copies": [
    "",
    "",
    ""
  ],
  "generation_prompt": {
    "background_prompt": "",
    "negative_prompt": ""
  },
  "layout": {
    "product_position": "lower_center",
    "product_x": 0.50,
    "product_y": 0.70,
    "product_scale": 0.44,
    "headline_position": "top_center"
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

    ad_copies = normalize_string_list(data.get("ad_copies"))
    data["ad_copies"] = ad_copies[:3]

    while len(data["ad_copies"]) < 3:
        data["ad_copies"].append("")

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
