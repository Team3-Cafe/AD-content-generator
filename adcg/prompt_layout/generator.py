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
from .optimizer import (
    FINAL_WEIGHTS,
    PRESELECTION_WEIGHTS,
    VLM_DIMENSION_WEIGHTS,
    final_score,
    normalize_scores,
    preselection_score,
    vlm_design_score,
)
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
You are a senior advertising creative director. Independently score all five
finished advertisement candidates with varied content-aware geometry and
visual styling. Use an integer from 0 to 100 for every dimension.

Judge the rendered pixels, not the preset name. Prioritize:
1. immediate headline hierarchy and semantic role clarity;
2. readable title, subtitle, price, and CTA;
3. balanced negative space without covering the product;
4. coherent palette, restrained panels, spacing, and alignment;
5. professional commercial finish without template-like clutter.

Penalize a candidate when text blends into the background, panels feel
excessive, the CTA looks like body text, or the style competes with the
product. Judge every candidate; do not select a winner. The local optimizer
will combine these ratings with LAION aesthetic and structural scores.
""".strip()


@dataclass(frozen=True)
class LayoutGenerationResult:
    output_dir: Path
    plan_json: Path
    layout_json: Path
    preview_html: Path
    rendered_image: Path
    style_selection_json: Path
    candidate_scores_json: Path
    layout_variants_json: Path
    candidate_images: tuple[Path, ...]
    selected_style: str
    selected_strategy: str


def _review_schema(candidate_ids: list[str]) -> dict:
    score = {"type": "integer", "minimum": 0, "maximum": 100}
    return {
        "type": "object",
        "properties": {
            "ratings": {
                "type": "array",
                "minItems": len(candidate_ids),
                "maxItems": len(candidate_ids),
                "items": {
                    "type": "object",
                    "properties": {
                        "candidate_id": {
                            "type": "string",
                            "enum": candidate_ids,
                        },
                        "hierarchy": score,
                        "readability": score,
                        "balance": score,
                        "commercial_finish": score,
                        "rationale": {"type": "string"},
                    },
                    "required": [
                        "candidate_id",
                        "hierarchy",
                        "readability",
                        "balance",
                        "commercial_finish",
                        "rationale",
                    ],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["ratings"],
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


def _review_candidates(
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
                "Score all five finished candidates. Copy:\n"
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
                            {
                                "structure": candidate["score"],
                                "laion_aesthetic_raw": candidate[
                                    "aesthetic_raw"
                                ],
                                "laion_aesthetic_normalized": candidate[
                                    "aesthetic_normalized"
                                ],
                            },
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
                "name": "ad_design_candidate_review",
                "strict": True,
                "schema": _review_schema(
                    [item["id"] for item in candidates]
                ),
            }
        },
    )
    return _parse_response_json(
        response,
        "ad_design_candidate_review",
    )


def _validated_ratings(review: dict, candidates: list[dict]) -> dict:
    ratings = review.get("ratings", [])
    by_id = {item["candidate_id"]: item for item in ratings}
    expected = {item["id"] for item in candidates}
    if len(ratings) != len(by_id) or set(by_id) != expected:
        raise ValueError(
            "GPT-4o candidate review must rate every finalist exactly once."
        )
    return by_id


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
                key=lambda item: item["preselection_score"],
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
        key=lambda item: item["preselection_score"],
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

    candidate_dir = output_dir / "design_candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    candidates = []
    valid_options = [item for item in options if not item["violations"]]
    if len(valid_options) < 5:
        raise RuntimeError(
            "Fewer than five candidates survived hard layout validation."
        )
    for item in valid_options:
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

    from adcg.eval.eval_LAION_aesthetic_score import (
        score_aesthetic_images,
    )

    aesthetic_rows = score_aesthetic_images(
        [item["path"] for item in candidates]
    )
    if len(aesthetic_rows) != len(candidates):
        raise RuntimeError("LAION returned an unexpected candidate count.")
    raw_aesthetic_scores = [float(score) for _, score in aesthetic_rows]
    normalized_aesthetic_scores = normalize_scores(raw_aesthetic_scores)
    for candidate, raw_score, normalized_score in zip(
        candidates,
        raw_aesthetic_scores,
        normalized_aesthetic_scores,
    ):
        candidate["aesthetic_raw"] = round(raw_score, 6)
        candidate["aesthetic_normalized"] = normalized_score
        candidate["preselection_score"] = preselection_score(
            aesthetic=normalized_score,
            structure=float(candidate["score"]["total"]),
        )

    finalists = _top_candidate_specs(candidates)
    review = _review_candidates(
        client,
        model=model,
        detail=detail,
        ad_copy=ad_copy,
        candidates=finalists,
    )
    ratings = _validated_ratings(review, finalists)
    for candidate in finalists:
        rating = ratings[candidate["id"]]
        candidate["vlm_rating"] = rating
        candidate["vlm_design_score"] = vlm_design_score(rating)
        candidate["final_score"] = final_score(
            vlm_design=candidate["vlm_design_score"],
            aesthetic=candidate["aesthetic_normalized"],
            structure=float(candidate["score"]["total"]),
        )

    selected_candidate = max(
        finalists,
        key=lambda item: item["final_score"],
    )
    selected_candidate_id = selected_candidate["id"]
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

    finalist_ids = {item["id"] for item in finalists}
    candidate_scores_document = {
        "optimizer": "hps_free_multi_objective",
        "selected_candidate_id": selected_candidate_id,
        "preselection_weights": PRESELECTION_WEIGHTS,
        "final_weights": FINAL_WEIGHTS,
        "vlm_dimension_weights": VLM_DIMENSION_WEIGHTS,
        "candidates": [
            {
                "id": item["id"],
                "strategy": item["strategy"],
                "style": item["style"],
                "path": str(item["path"]),
                "structure": item["score"],
                "aesthetic_raw": item["aesthetic_raw"],
                "aesthetic_normalized": item[
                    "aesthetic_normalized"
                ],
                "preselection_score": item["preselection_score"],
                "finalist": item["id"] in finalist_ids,
                "vlm_rating": item.get("vlm_rating"),
                "vlm_design_score": item.get("vlm_design_score"),
                "final_score": item.get("final_score"),
            }
            for item in candidates
        ],
        "rejected_candidates": [
            {
                "id": item["id"],
                "violations": item["violations"],
            }
            for item in options
            if item["violations"]
        ],
    }
    candidate_scores_path = output_dir / "candidate_scores.json"
    candidate_scores_path.write_text(
        json.dumps(
            candidate_scores_document,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    selection_document = {
        "model": model,
        "optimizer": "hps_free_multi_objective",
        "selected_candidate_id": selected_candidate_id,
        "selected_strategy": selected_strategy,
        "selected_style": selected_style,
        "final_score": selected_candidate["final_score"],
        "rationale": selected_candidate["vlm_rating"]["rationale"],
        "weights": FINAL_WEIGHTS,
        "candidates": [
            {
                "id": item["id"],
                "strategy": item["strategy"],
                "style": item["style"],
                "path": str(item["path"]),
                "structure": item["score"],
                "aesthetic_raw": item["aesthetic_raw"],
                "aesthetic_normalized": item[
                    "aesthetic_normalized"
                ],
                "vlm_rating": item["vlm_rating"],
                "vlm_design_score": item["vlm_design_score"],
                "final_score": item["final_score"],
                "palette": item["palette"],
                "decorative_panel_count": item[
                    "decorative_panel_count"
                ],
            }
            for item in finalists
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
        candidate_scores_json=candidate_scores_path,
        layout_variants_json=variants_path,
        candidate_images=tuple(
            item["path"]
            for item in candidates
        ),
        selected_style=selected_style,
        selected_strategy=selected_strategy,
    )
