from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from dotenv import load_dotenv
from openai import OpenAI

from .analysis import analyze_image_space
from .engine import (
    apply_design_revision,
    apply_final_review_revision,
    build_design_layout,
)
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
from .schemas import (
    DESIGN_REVISION_SCHEMA,
    DESIGN_SPEC_SCHEMA,
    FINAL_REVIEW_FEATURE_CONTROLS,
    FINAL_REVIEW_FEATURES,
    FINAL_REVIEW_SCHEMA,
)


ROOT_DIR = Path(__file__).resolve().parents[2]
MIN_FINAL_REVISED_FEATURES = 6
MIN_FINAL_NON_NEUTRAL_CONTROLS = 10


_SCALE_ADJUSTMENTS = {
    "headline_scale", "offer_scale", "title_scale", "subtitle_scale",
    "price_scale", "cta_scale", "price_number_scale", "price_unit_scale",
    "headline_band_height_scale", "offer_band_height_scale",
    "accent_rule_width_scale",
}


def _is_non_neutral_adjustment(name: str, value) -> bool:
    if name in _SCALE_ADJUSTMENTS:
        return abs(float(value) - 1.0) >= 1e-9
    if isinstance(value, str):
        return value != "keep"
    return abs(float(value)) >= 1e-9


def _reconcile_feature_controls(review: dict) -> list[dict]:
    """Link VLM-selected adjustments to compatible feature feedback."""
    adjustments = review["adjustments"]
    feature_reviews = review["diagnosis"]["feature_reviews"]
    reconciled = []

    for feature in FINAL_REVIEW_FEATURES:
        feedback = feature_reviews[feature]
        original_controls = list(feedback["controls"])
        active_controls = [
            control
            for control in original_controls
            if _is_non_neutral_adjustment(
                control, adjustments[control]
            )
        ]
        if active_controls != original_controls:
            feedback["controls"] = active_controls
            reconciled.append(
                {
                    "feature": feature,
                    "removed_neutral_controls": [
                        control
                        for control in original_controls
                        if control not in active_controls
                    ],
                }
            )
        if active_controls and feedback["verdict"] == "keep":
            feedback["verdict"] = "revise"
            reconciled.append(
                {"feature": feature, "verdict_changed_to": "revise"}
            )
        elif not active_controls and feedback["verdict"] == "revise":
            feedback["verdict"] = "keep"
            reconciled.append(
                {"feature": feature, "verdict_changed_to": "keep"}
            )

    referenced = {
        control
        for feedback in feature_reviews.values()
        for control in feedback["controls"]
    }
    for control, value in adjustments.items():
        if not _is_non_neutral_adjustment(control, value):
            continue
        if control in referenced:
            continue
        compatible = [
            feature
            for feature in FINAL_REVIEW_FEATURES
            if control in FINAL_REVIEW_FEATURE_CONTROLS[feature]
        ]
        revised = [
            feature
            for feature in compatible
            if feature_reviews[feature]["verdict"] == "revise"
        ]
        if not compatible:
            continue
        feature = (revised or compatible)[0]
        feedback = feature_reviews[feature]
        feedback["verdict"] = "revise"
        feedback["controls"].append(control)
        referenced.add(control)
        reconciled.append(
            {"feature": feature, "linked_control": control}
        )
    return reconciled

def _review_consistency_issues(review: dict) -> list[str]:
    """Check that feature feedback and model-selected controls agree."""
    adjustments = review["adjustments"]
    feature_reviews = review.get("diagnosis", {}).get(
        "feature_reviews", {}
    )
    issues = []
    referenced_controls = set()
    for feature, feedback in feature_reviews.items():
        verdict = feedback.get("verdict")
        controls = feedback.get("controls", [])
        if verdict == "revise" and not controls:
            issues.append(f"{feature} needs revision but selects no controls")
        if verdict == "keep" and controls:
            issues.append(f"{feature} is keep but selects controls")
        if verdict != "revise":
            continue
        for control in controls:
            referenced_controls.add(control)
            if not _is_non_neutral_adjustment(
                control, adjustments[control]
            ):
                issues.append(
                    f"{feature} selects neutral control {control}"
                )

    for control, value in adjustments.items():
        if (
            _is_non_neutral_adjustment(control, value)
            and control not in referenced_controls
        ):
            issues.append(
                f"non-neutral control {control} has no feature feedback"
            )
    return issues


def _review_scope_issues(review: dict) -> list[str]:
    """Require a substantial, model-selected final design revision."""
    feature_reviews = review["diagnosis"]["feature_reviews"]
    revised_features = [
        feature
        for feature, feedback in feature_reviews.items()
        if feedback["verdict"] == "revise" and feedback["controls"]
    ]
    active_controls = {
        control
        for feature in revised_features
        for control in feature_reviews[feature]["controls"]
        if _is_non_neutral_adjustment(
            control, review["adjustments"][control]
        )
    }
    issues = []
    if len(revised_features) < MIN_FINAL_REVISED_FEATURES:
        issues.append(
            "broad revision requires at least "
            f"{MIN_FINAL_REVISED_FEATURES} revised features; "
            f"received {len(revised_features)}"
        )
    if len(active_controls) < MIN_FINAL_NON_NEUTRAL_CONTROLS:
        issues.append(
            "broad revision requires at least "
            f"{MIN_FINAL_NON_NEUTRAL_CONTROLS} non-neutral controls; "
            f"received {len(active_controls)}"
        )
    return issues


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

def _enforce_final_review_revision(review: dict, layout: dict) -> dict:
    """Validate and apply only the VLM's feature-backed corrections."""
    review["needs_revision"] = True
    reconciled = _reconcile_feature_controls(review)
    if reconciled:
        review["feature_feedback_reconciled"] = True
        review["feature_feedback_reconciliations"] = reconciled
    consistency_issues = _review_consistency_issues(review)
    if consistency_issues:
        raise ValueError(
            "Final review feedback is inconsistent with its adjustments: "
            + "; ".join(consistency_issues)
        )

    scope_issues = _review_scope_issues(review)
    review["scope_target_met"] = not scope_issues
    if scope_issues:
        review["scope_issues"] = scope_issues

    candidate = apply_final_review_revision(layout, review)
    visible_change = (
        candidate["elements"] != layout["elements"]
        or candidate.get("underlays", []) != layout.get("underlays", [])
    )
    if not visible_change:
        raise ValueError(
            "Final review selected no effective design revision."
        )
    review["consistency_validated"] = True
    review["revision_enforced"] = False
    return review


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
        schema=FINAL_REVIEW_SCHEMA,
        temperature=0,
    )
    final_review = _enforce_final_review_revision(
        final_review, revised_layout
    )
    final_review["review_attempts"] = 1
    final_review_path = _write_json(
        output_dir / "final_review.json",
        {"model": model, **final_review},
    )

    final_layout = apply_final_review_revision(revised_layout, final_review)
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
