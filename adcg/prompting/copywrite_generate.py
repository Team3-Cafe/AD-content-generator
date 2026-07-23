from __future__ import annotations

import json
from time import perf_counter

from openai import OpenAI

COPY_INPUT_FIELDS = (
    "product_name",
    "product_description",
    "seller_description",
    "store_info",
    "reviews",
    "focus",
    "additional_request",
    "store_name",
    "store_type",
    "product_category",
    "features",
    "target_customer",
    "promotion",
    "price",
    "tone",
)

COPY_LENGTH_GUIDANCE = {
    "short": (
        "Keep title within 12 Korean characters and subtitle within "
        "20 Korean characters."
    ),
    "medium": (
        "Keep title within 18 Korean characters and subtitle within "
        "35 Korean characters."
    ),
    "long": (
        "Keep title within 24 Korean characters and subtitle within "
        "50 Korean characters."
    ),
}


def generate_ad_copy(
    product_info: dict,
    background_prompt: str,
    model: str,
    *,
    copy_tone: str | None = None,
    copy_length: str | None = None,
) -> dict:
    """Generate one concise, evidence-grounded Korean advertising copy set."""
    if copy_tone is not None:
        copy_tone = str(copy_tone).strip()
        if not copy_tone:
            raise ValueError("copy_tone은 비어 있을 수 없습니다.")

    if copy_length is not None:
        copy_length = str(copy_length).strip().lower()
        if copy_length not in COPY_LENGTH_GUIDANCE:
            raise ValueError(
                "copy_length는 short, medium, long 중 하나여야 합니다."
            )

    request_data = {
        field: product_info.get(field)
        for field in COPY_INPUT_FIELDS
    }
    request_data["background_prompt"] = background_prompt
    if copy_tone is not None:
        request_data["copy_tone"] = copy_tone
    if copy_length is not None:
        request_data["copy_length"] = copy_length
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
        "Write concise Korean advertising copy using only the supplied facts. "
        "Use seller_description, store_info, reviews, focus, and "
        "additional_request when present. Treat reviews as supporting context, "
        "not as verified claims or verbatim testimonials. Interpret focus as "
        "product, brand, or sales emphasis without inventing facts. "
        "Do not invent prices, discounts, benefits, or contact channels. Return "
        "an empty price when price is absent. The accepted input has no contact "
        "destination, so return an empty CTA and never invent a generic contact "
        "invitation.\n\n"
        + json.dumps(request_data, ensure_ascii=False, indent=2)
    )

    if copy_tone is not None:
        prompt += (
            "\n\nUse this explicit copywriting tone in preference to any "
            f"metadata tone: {copy_tone}."
        )
    if copy_length is not None:
        prompt += "\n" + COPY_LENGTH_GUIDANCE[copy_length]


    started_at = perf_counter()
    client = OpenAI()
    response = client.responses.create(
        model=model,
        instructions=(
            "You are a Korean advertising copywriter. Write title, subtitle, "
            "price, and CTA briefly and clearly using supplied facts only."
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
    copy["cta"] = ""
    copy["model"] = model
    copy["latency_sec"] = round(perf_counter() - started_at, 2)
    return copy
