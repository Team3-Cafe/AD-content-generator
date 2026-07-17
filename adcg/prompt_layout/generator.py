from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

from .html import render_layout_html
from .io import image_to_data_url
from .prompts import (
    LAYOUT_SYSTEM_PROMPT,
    PLAN_SYSTEM_PROMPT,
    build_layout_request,
    build_plan_request,
)
from .renderer import render_layout_image
from .schemas import LAYOUT_SCHEMA, PLACEMENT_PLAN_SCHEMA
from .validation import normalize_layout


ROOT_DIR = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class LayoutGenerationResult:
    output_dir: Path
    plan_json: Path
    layout_json: Path
    preview_html: Path
    rendered_image: Path


def _request_json(
    client,
    *,
    model: str,
    instructions: str,
    request_text: str,
    image_url: str,
    detail: str,
    schema_name: str,
    schema: dict,
    temperature: float,
) -> dict:
    response = client.responses.create(
        model=model,
        instructions=instructions,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": request_text,
                    },
                    {
                        "type": "input_image",
                        "image_url": image_url,
                        "detail": detail,
                    },
                ],
            }
        ],
        temperature=temperature,
        top_p=1.0,
        text={
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "strict": True,
                "schema": schema,
            }
        },
    )
    output_text = str(response.output_text or "").strip()
    if not output_text:
        raise RuntimeError(
            f"GPT-4o returned no output for {schema_name}."
        )
    try:
        result = json.loads(output_text)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"GPT-4o returned invalid JSON for {schema_name}: "
            f"{output_text[:500]}"
        ) from error
    if not isinstance(result, dict):
        raise ValueError(
            f"GPT-4o returned a non-object for {schema_name}."
        )
    return result


def generate_prompt_layout(
    image_path: str | Path,
    ad_copy: dict[str, str],
    output_dir: str | Path,
    *,
    model: str = "gpt-4o",
    detail: str = "high",
    temperature: float = 0.7,
    font_path: str | Path | None = None,
    client=None,
) -> LayoutGenerationResult:
    """Generate a plan, pixel layout, and HTML preview without running the pipeline."""
    image_path = Path(image_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if detail not in {"low", "high", "auto"}:
        raise ValueError("detail must be one of: low, high, auto")
    if not 0 <= temperature <= 2:
        raise ValueError("temperature must be between 0 and 2.")
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not ad_copy:
        raise ValueError("ad_copy must contain at least one non-empty field.")

    output_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path) as image:
        width, height = image.size

    if client is None:
        load_dotenv(ROOT_DIR / ".env")
        client = OpenAI()

    image_url = image_to_data_url(image_path)
    started_at = perf_counter()

    plan = _request_json(
        client,
        model=model,
        instructions=PLAN_SYSTEM_PROMPT,
        request_text=build_plan_request(ad_copy, width, height),
        image_url=image_url,
        detail=detail,
        schema_name="ad_copy_placement_plan",
        schema=PLACEMENT_PLAN_SCHEMA,
        temperature=temperature,
    )
    plan_path = output_dir / "placement_plan.json"
    plan_path.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    raw_layout = _request_json(
        client,
        model=model,
        instructions=LAYOUT_SYSTEM_PROMPT,
        request_text=build_layout_request(
            ad_copy,
            plan,
            width,
            height,
        ),
        image_url=image_url,
        detail=detail,
        schema_name="ad_copy_pixel_layout",
        schema=LAYOUT_SCHEMA,
        temperature=temperature,
    )
    layout = normalize_layout(
        raw_layout,
        copy=ad_copy,
        width=width,
        height=height,
    )
    layout_document = {
        "model": model,
        "source_image": str(image_path),
        "latency_sec": round(perf_counter() - started_at, 2),
        "copy": ad_copy,
        **layout,
    }
    layout_path = output_dir / "layout.json"
    layout_path.write_text(
        json.dumps(layout_document, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    preview_path = render_layout_html(
        image_path=image_path,
        layout=layout,
        output_path=output_dir / "layout_preview.html",
    )
    rendered_path = render_layout_image(
        image_path=image_path,
        layout=layout,
        output_path=output_dir / "final_ad.png",
        font_path=font_path,
    )
    return LayoutGenerationResult(
        output_dir=output_dir,
        plan_json=plan_path,
        layout_json=layout_path,
        preview_html=preview_path,
        rendered_image=rendered_path,
    )
