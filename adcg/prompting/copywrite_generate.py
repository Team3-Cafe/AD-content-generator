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
    """Generate one concise, evidence-grounded Korean advertising copy set."""
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
        "Write concise Korean advertising copy using only the supplied facts. "
        "Do not invent prices, discounts, benefits, or contact channels. Return "
        "an empty price when price is absent. The accepted input has no contact "
        "destination, so return an empty CTA and never invent a generic contact "
        "invitation.\n\n"
        + json.dumps(request_data, ensure_ascii=False, indent=2)
    )


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
