from __future__ import annotations

import json
from time import perf_counter

from openai import OpenAI

COPY_INPUT_FIELDS = (
    "product_name",
    "store_name",
    "store_type",
    "product_category",
    "product_description",
    "price",
    "tone",
)


def generate_ad_copy(
    product_info: dict,
    background_prompt: str,
    model: str,
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