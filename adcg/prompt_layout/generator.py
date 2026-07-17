from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import shutil
from time import perf_counter

from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image

from .constraints import (
    layout_geometry_signature,
    layout_geometry_violations,
)
from .few_shot import EXAMPLE_SPECS, build_few_shot_content
from .html import render_layout_html
from .io import image_to_data_url
from .prompts import (
    LAYOUT_VARIANTS_SYSTEM_PROMPT,
    PLAN_SYSTEM_PROMPT,
    build_layout_request,
    build_plan_request,
)
from .renderer import (
    ensure_layout_contrast,
    fit_layout_typography,
    render_layout_image,
)
from .schemas import (
    LAYOUT_STRATEGIES,
    LAYOUT_VARIANTS_SCHEMA,
    PLACEMENT_PLAN_SCHEMA,
)
from .styles import (
    STYLE_NAMES,
    apply_style,
    score_layout_design,
)
from .validation import normalize_layout, normalize_plan_grid


ROOT_DIR = Path(__file__).resolve().parents[2]


STYLE_SELECTION_PROMPT = """
You are a senior advertising creative director. Select the strongest finished
advertisement from five candidates with varied content-aware geometry and
visual styling.

Judge the rendered pixels, not the preset name. Prioritize:
1. immediate headline hierarchy and semantic role clarity;
2. readable title, subtitle, price, and CTA;
3. balanced negative space without covering the product;
4. coherent palette, restrained panels, spacing, and alignment;
5. professional commercial finish without template-like clutter.

Reject a candidate when text blends into the background, panels feel
excessive, the CTA looks like body text, or the style competes with the
product. Return one selected candidate id and a concise rationale.
""".strip()


@dataclass(frozen=True)
class LayoutGenerationResult:
    output_dir: Path
    plan_json: Path
    layout_json: Path
    preview_html: Path
    rendered_image: Path
    style_selection_json: Path
    layout_variants_json: Path
    candidate_images: tuple[Path, ...]
    selected_style: str
    selected_strategy: str


def _selection_schema(candidate_ids: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "selected_candidate_id": {
                "type": "string",
                "enum": candidate_ids,
            },
            "rationale": {"type": "string"},
        },
        "required": ["selected_candidate_id", "rationale"],
        "additionalProperties": False,
    }


def _parse_response_json(response, schema_name: str) -> dict:
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
    few_shot_stage: str,
    canvas_width: int,
    canvas_height: int,
) -> dict:
    content = build_few_shot_content(
        few_shot_stage,
        canvas_width,
        canvas_height,
    )
    content.extend(
        [
            {
                "type": "input_text",
                "text": (
                    "Test sample input. Apply the demonstrated design "
                    "principles to this image and return only the required "
                    "structured output:\n" + request_text
                ),
            },
            {
                "type": "input_image",
                "image_url": image_url,
                "detail": detail,
            },
        ]
    )
    response = client.responses.create(
        model=model,
        instructions=instructions,
        input=[
            {
                "role": "user",
                "content": content,
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
    return _parse_response_json(response, schema_name)


def _select_candidate(
    client,
    *,
    model: str,
    detail: str,
    ad_copy: dict[str, str],
    candidates: list[dict],
) -> dict:
    content = [
        {
            "type": "input_text",
            "text": (
                "Compare all five finished candidates. The deterministic "
                "scores are supporting diagnostics only; visual judgment "
                "takes priority. Copy:\n"
                + json.dumps(ad_copy, ensure_ascii=False, indent=2)
            ),
        }
    ]
    for candidate in candidates:
        content.extend(
            [
                {
                    "type": "input_text",
                    "text": (
                        f"Candidate id: {candidate['id']}\n"
                        f"Geometry: {candidate['strategy']}\n"
                        f"Style: {candidate['style']}\n"
                        + json.dumps(
                            candidate["score"],
                            ensure_ascii=False,
                            indent=2,
                        )
                    ),
                },
                {
                    "type": "input_image",
                    "image_url": image_to_data_url(candidate["path"]),
                    "detail": detail,
                },
            ]
        )
    response = client.responses.create(
        model=model,
        instructions=STYLE_SELECTION_PROMPT,
        input=[{"role": "user", "content": content}],
        temperature=0,
        top_p=1.0,
        text={
            "format": {
                "type": "json_schema",
                "name": "ad_design_style_selection",
                "strict": True,
                "schema": _selection_schema(
                    [item["id"] for item in candidates]
                ),
            }
        },
    )
    return _parse_response_json(
        response,
        "ad_design_style_selection",
    )


def _top_candidate_specs(options: list[dict]) -> list[dict]:
    valid = [
        item
        for item in options
        if not item["violations"]
    ]
    chosen = []
    for strategy in LAYOUT_STRATEGIES:
        strategy_options = [
            item
            for item in valid
            if item["strategy"] == strategy
        ]
        if not strategy_options:
            continue
        chosen.append(
            max(
                strategy_options,
                key=lambda item: item["score"]["total"],
            )
        )
    if len(chosen) < 3:
        diagnostics = {
            item["id"]: item["violations"]
            for item in options
            if item["violations"]
        }
        raise RuntimeError(
            "Fewer than three distinct layout strategies survived hard "
            "validation: "
            + json.dumps(diagnostics, ensure_ascii=False)
        )

    chosen_ids = {item["id"] for item in chosen}
    remaining = sorted(
        (
            item
            for item in valid
            if item["id"] not in chosen_ids
        ),
        key=lambda item: item["score"]["total"],
        reverse=True,
    )
    for item in remaining:
        if len(chosen) >= 5:
            break
        chosen.append(item)
    if len(chosen) != 5:
        raise RuntimeError(
            "Fewer than five valid geometry/style candidates survived."
        )
    return chosen


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
    """Generate four layouts, five finalists, and one selected ad."""
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
        few_shot_stage="plan",
        canvas_width=width,
        canvas_height=height,
    )
    plan = normalize_plan_grid(plan, width, height)
    plan_path = output_dir / "placement_plan.json"
    plan_path.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    raw_variants = _request_json(
        client,
        model=model,
        instructions=LAYOUT_VARIANTS_SYSTEM_PROMPT,
        request_text=build_layout_request(
            ad_copy,
            plan,
            width,
            height,
        ),
        image_url=image_url,
        detail=detail,
        schema_name="ad_copy_pixel_layout_variants",
        schema=LAYOUT_VARIANTS_SCHEMA,
        temperature=temperature,
        few_shot_stage="layout",
        canvas_width=width,
        canvas_height=height,
    )
    variants = []
    seen_strategies = set()
    signatures = set()
    for raw_variant in raw_variants["variants"]:
        strategy = raw_variant["strategy"]
        if strategy in seen_strategies:
            raise ValueError(
                f"Duplicate layout strategy returned: {strategy}"
            )
        seen_strategies.add(strategy)
        layout = normalize_layout(
            raw_variant["layout"],
            copy=ad_copy,
            width=width,
            height=height,
            plan=plan,
        )
        layout = fit_layout_typography(
            layout,
            font_path=font_path,
        )
        signature = layout_geometry_signature(layout)
        if signature in signatures:
            raise ValueError(
                "GPT-4o returned duplicate layout geometry for "
                f"{strategy}."
            )
        signatures.add(signature)
        variants.append(
            {
                "strategy": strategy,
                "layout": layout,
            }
        )
    if seen_strategies != set(LAYOUT_STRATEGIES):
        raise ValueError(
            "GPT-4o did not return every required layout strategy."
        )

    variants_document = {
        "model": model,
        "few_shot_count": len(EXAMPLE_SPECS),
        "source_image": str(image_path),
        "copy": ad_copy,
        "variants": variants,
    }
    variants_path = output_dir / "layout_variants.json"
    variants_path.write_text(
        json.dumps(
            variants_document,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    options = []
    for variant in variants:
        for style_name in STYLE_NAMES:
            styled_layout, metadata = apply_style(
                variant["layout"],
                image_path,
                style_name,
            )
            styled_layout = fit_layout_typography(
                styled_layout,
                font_path=font_path,
            )
            styled_layout = ensure_layout_contrast(
                image_path,
                styled_layout,
            )
            candidate_id = f"{variant['strategy']}__{style_name}"
            options.append(
                {
                    "id": candidate_id,
                    "strategy": variant["strategy"],
                    "style": style_name,
                    "layout": styled_layout,
                    "score": score_layout_design(styled_layout),
                    "palette": metadata["palette"],
                    "decorative_panel_count": metadata[
                        "decorative_panel_count"
                    ],
                    "violations": layout_geometry_violations(
                        styled_layout,
                        plan=plan,
                    ),
                }
            )

    selected_specs = _top_candidate_specs(options)
    candidate_dir = output_dir / "design_candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    candidates = []
    for item in selected_specs:
        candidate_path = render_layout_image(
            image_path=image_path,
            layout=item["layout"],
            output_path=candidate_dir / f"{item['id']}.png",
            font_path=font_path,
        )
        candidates.append(
            {
                **item,
                "path": candidate_path,
            }
        )

    selection = _select_candidate(
        client,
        model=model,
        detail=detail,
        ad_copy=ad_copy,
        candidates=candidates,
    )
    selected_candidate_id = selection["selected_candidate_id"]
    selected_candidate = next(
        item
        for item in candidates
        if item["id"] == selected_candidate_id
    )
    selected_style = selected_candidate["style"]
    selected_strategy = selected_candidate["strategy"]
    selected_layout = selected_candidate["layout"]
    rendered_path = output_dir / "final_ad.png"
    shutil.copyfile(selected_candidate["path"], rendered_path)

    layout_document = {
        "model": model,
        "few_shot_count": len(EXAMPLE_SPECS),
        "source_image": str(image_path),
        "latency_sec": round(perf_counter() - started_at, 2),
        "copy": ad_copy,
        "selected_strategy": selected_strategy,
        "selected_style": selected_style,
        **selected_layout,
    }
    layout_path = output_dir / "layout.json"
    layout_path.write_text(
        json.dumps(layout_document, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    preview_path = render_layout_html(
        image_path=image_path,
        layout=selected_layout,
        output_path=output_dir / "layout_preview.html",
    )

    selection_document = {
        "model": model,
        "selected_candidate_id": selected_candidate_id,
        "selected_strategy": selected_strategy,
        "selected_style": selected_style,
        "rationale": selection["rationale"],
        "candidates": [
            {
                "id": item["id"],
                "strategy": item["strategy"],
                "style": item["style"],
                "path": str(item["path"]),
                "score": item["score"],
                "palette": item["palette"],
                "decorative_panel_count": item[
                    "decorative_panel_count"
                ],
            }
            for item in candidates
        ],
    }
    selection_path = output_dir / "style_selection.json"
    selection_path.write_text(
        json.dumps(
            selection_document,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return LayoutGenerationResult(
        output_dir=output_dir,
        plan_json=plan_path,
        layout_json=layout_path,
        preview_html=preview_path,
        rendered_image=rendered_path,
        style_selection_json=selection_path,
        layout_variants_json=variants_path,
        candidate_images=tuple(
            item["path"]
            for item in candidates
        ),
        selected_style=selected_style,
        selected_strategy=selected_strategy,
    )
