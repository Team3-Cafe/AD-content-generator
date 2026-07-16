from __future__ import annotations

import json
from time import perf_counter

from openai import OpenAI

COPY_MODEL = "gpt-5.4-mini"
COPY_INPUT_FIELDS = (
    "product_name",
    "store_name",
    "store_type",
    "product_category",
    "product_description",
    "features",
    "target_customer",
    "promotion",
    "price",
    "tone",
)


def generate_ad_copy(
    product_info: dict,
    background_prompt: str,
    model: str = COPY_MODEL,
) -> dict:
    """단일 OpenAI 모델로 한국어 광고 문구 한 세트를 생성한다."""
    request_data = {
        field: product_info.get(field)
        for field in COPY_INPUT_FIELDS
    }
    request_data["background_prompt"] = background_prompt
    schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "subtitle": {"type": "string"},
            "price": {"type": "string"},
            "cta": {"type": "string"},
        },
        "required": ["title", "subtitle", "price", "cta"],
        "additionalProperties": False,
    }
    prompt = (
        "아래 정보를 바탕으로 자연스럽고 간결한 한국어 광고 문구를 작성하세요. "
        "확인할 수 없는 가격, 할인율, 효능은 만들지 마세요. "
        "가격 정보가 없으면 price는 빈 문자열로 반환하세요.\n\n"
        + json.dumps(request_data, ensure_ascii=False, indent=2)
    )

    started_at = perf_counter()
    client = OpenAI()
    response = client.responses.create(
        model=model,
        instructions=(
            "당신은 상품 광고 카피라이터입니다. title, subtitle, price, cta를 "
            "각각 짧고 명확하게 작성하세요."
        ),
        input=prompt,
        text={
            "format": {
                "type": "json_schema",
                "name": "advertising_copy",
                "strict": True,
                "schema": schema,
            }
        },
    )
    copy = json.loads(response.output_text)
    copy["model"] = model
    copy["latency_sec"] = round(perf_counter() - started_at, 2)
    return copy


# TODO: 파이프라인에 연결할 때 아래 경로를 실제 경로로 교체한다.
# INFO_JSON_PATH = "상품/매장 정보 JSON 경로"
# PROMPT_JSON_PATH = "ad_prompt.json 경로"
# with open(INFO_JSON_PATH, "r", encoding="utf-8") as file:
#     product_info = json.load(file)
# with open(PROMPT_JSON_PATH, "r", encoding="utf-8") as file:
#     prompt_data = json.load(file)
#
# 문구 생성에 필요한 입력:
# - product_info: product_name, store_name, store_type, product_category,
#   product_description, features, target_customer, promotion, price, tone
# - background_prompt: 생성 이미지의 배경 프롬프트
#
# result = generate_ad_copy(
#     product_info=product_info,
#     background_prompt=prompt_data["generation_prompt"]["background_prompt"],
# )
