from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from dotenv import load_dotenv
from openai import OpenAI

from .analysis import analyze_image_space
from .engine import apply_design_revision, build_design_layout
from .io import image_to_data_url
from .prompts import (
    DESIGN_SYSTEM_PROMPT,
    FINAL_REVIEW_SYSTEM_PROMPT,
    REVISION_SYSTEM_PROMPT,
    build_design_request,
    build_final_review_request,
    build_revision_request,
)
from .renderer import (
    ensure_layout_contrast,
    fit_layout_typography,
    render_layout_image,
)
from .schemas import DESIGN_REVISION_SCHEMA, DESIGN_SPEC_SCHEMA


ROOT_DIR = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class LayoutGenerationResult:
    output_dir: Path
    design_analysis_json: Path
    design_spec_json: Path
    design_revision_json: Path
    final_review_json: Path
    layout_json: Path
    draft_image: Path
    final_review_input_image: Path
    rendered_image: Path


def _parse_response_json(response, schema_name: str) -> dict:
    output_text = str(response.output_text or "").strip()
    if not output_text:
        raise RuntimeError(f"GPT-4o returned no output for {schema_name}.")
    try:
        result = json.loads(output_text)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"GPT-4o returned invalid JSON for {schema_name}: "
            f"{output_text[:500]}"
        ) from error
    if not isinstance(result, dict):
        raise ValueError(f"GPT-4o returned a non-object for {schema_name}.")
    return result


def _request_json(
    client,
    *,
    model: str,
    instructions: str,
    request_text: str,
    image_path: Path,
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
                    {"type": "input_text", "text": request_text},
                    {
                        "type": "input_image",
                        "image_url": image_to_data_url(image_path),
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
    return _parse_response_json(response, schema_name)


def _write_json(path: Path, document: dict) -> Path:
    path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def generate_prompt_layout(
    image_path: str | Path,
    ad_copy: dict[str, str],
    output_dir: str | Path,
    *,
    model: str = "gpt-4o",
    detail: str = "high",
    temperature: float = 0.4,
    font_path: str | Path | None = None,
    client=None,
) -> LayoutGenerationResult:
    """Create and refine one content-aware advertisement design."""
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
    if not ad_copy.get("title"):
        raise ValueError("ad_copy.title is required for headline hierarchy.")

    output_dir.mkdir(parents=True, exist_ok=True)
    if client is None:
        load_dotenv(ROOT_DIR / ".env")
        client = OpenAI()

    started_at = perf_counter()
    computed_analysis = analyze_image_space(image_path)
    design_spec = _request_json(
        client,
        model=model,
        instructions=DESIGN_SYSTEM_PROMPT,
        request_text=build_design_request(ad_copy, computed_analysis),
        image_path=image_path,
        detail=detail,
        schema_name="single_ad_design_spec",
        schema=DESIGN_SPEC_SCHEMA,
        temperature=temperature,
    )

    analysis_document = {
        "source_image": str(image_path),
        "computed_image_space": computed_analysis,
        "vlm_scene_analysis": design_spec["scene_analysis"],
    }
    analysis_path = _write_json(
        output_dir / "design_analysis.json",
        analysis_document,
    )
    spec_path = _write_json(
        output_dir / "design_spec.json",
        {
            "model": model,
            "copy": ad_copy,
            **design_spec,
        },
    )

    draft_layout = build_design_layout(
        computed_analysis,
        ad_copy,
        design_spec,
    )
    draft_layout = fit_layout_typography(
        draft_layout,
        font_path=font_path,
    )
    draft_layout = ensure_layout_contrast(image_path, draft_layout)
    draft_path = render_layout_image(
        image_path=image_path,
        layout=draft_layout,
        output_path=output_dir / "design_draft.png",
        font_path=font_path,
    )

    revision = _request_json(
        client,
        model=model,
        instructions=REVISION_SYSTEM_PROMPT,
        request_text=build_revision_request(
            ad_copy,
            design_spec,
            draft_layout,
        ),
        image_path=draft_path,
        detail=detail,
        schema_name="single_ad_design_revision",
        schema=DESIGN_REVISION_SCHEMA,
        temperature=0,
    )
    revision_path = _write_json(
        output_dir / "design_revision.json",
        {"model": model, **revision},
    )

    revised_layout = apply_design_revision(draft_layout, revision)
    revised_layout = fit_layout_typography(
        revised_layout,
        font_path=font_path,
    )
    revised_layout = ensure_layout_contrast(image_path, revised_layout)
    final_review_input_path = render_layout_image(
        image_path=image_path,
        layout=revised_layout,
        output_path=output_dir / "final_review_input.png",
        font_path=font_path,
    )

    final_review = _request_json(
        client,
        model=model,
        instructions=FINAL_REVIEW_SYSTEM_PROMPT,
        request_text=build_final_review_request(
            ad_copy,
            design_spec,
            revised_layout,
        ),
        image_path=final_review_input_path,
        detail=detail,
        schema_name="completed_ad_final_layout_review",
        schema=DESIGN_REVISION_SCHEMA,
        temperature=0,
    )
    final_review_path = _write_json(
        output_dir / "final_review.json",
        {"model": model, **final_review},
    )

    final_layout = apply_design_revision(revised_layout, final_review)
    final_layout = fit_layout_typography(
        final_layout,
        font_path=font_path,
    )
    final_layout = ensure_layout_contrast(image_path, final_layout)
    layout_document = {
        "model": model,
        "source_image": str(image_path),
        "latency_sec": round(perf_counter() - started_at, 2),
        "copy": ad_copy,
        "design_rationale": design_spec["rationale"],
        "revision_reason": revision["reason"],
        "final_review_reason": final_review["reason"],
        **final_layout,
    }
    layout_path = _write_json(output_dir / "layout.json", layout_document)
    rendered_path = render_layout_image(
        image_path=image_path,
        layout=final_layout,
        output_path=output_dir / "final_ad.png",
        font_path=font_path,
    )

    return LayoutGenerationResult(
        output_dir=output_dir,
        design_analysis_json=analysis_path,
        design_spec_json=spec_path,
        design_revision_json=revision_path,
        final_review_json=final_review_path,
        layout_json=layout_path,
        draft_image=draft_path,
        final_review_input_image=final_review_input_path,
        rendered_image=rendered_path,
    )
