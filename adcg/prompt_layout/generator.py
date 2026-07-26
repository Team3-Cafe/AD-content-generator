from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from ..artifacts import keep_only

from dotenv import load_dotenv
from openai import OpenAI

from .analysis import analyze_image_space, build_design_candidate_pool
from .engine import apply_final_review_revision, build_design_layout
from .io import image_to_data_url
from .prompts import (
    DESIGN_SYSTEM_PROMPT,
    FINAL_REVIEW_SYSTEM_PROMPT,
    build_design_request,
    build_final_review_request,
)
from .renderer import (
    ensure_layout_contrast,
    fit_layout_typography,
    render_layout_image,
)
from .schemas import (
    DESIGN_SPEC_SCHEMA,
    FINAL_REVIEW_FEATURES,
    FINAL_REVIEW_SCHEMA,
)


ROOT_DIR = Path(__file__).resolve().parents[2]


def _normalize_feature_review_metadata(
    review: dict,
    layout: dict,
) -> list[dict]:
    """Repair harmless verdict/target contradictions from the VLM response."""
    normalizations = []
    rendered_roles = {str(item["role"]) for item in layout["elements"]}
    role_targets = {
        role: {f"{role}_geometry", f"{role}_typography"}
        for role in ("title", "subtitle", "price", "cta")
    }
    role_targets["price"].add("price_composition")
    inactive_targets = set().union(*(
        targets
        for role, targets in role_targets.items()
        if role not in rendered_roles
    ))

    feature_reviews = review["diagnosis"]["feature_reviews"]
    for feature, feedback in feature_reviews.items():
        original_verdict = str(feedback.get("verdict"))
        original_targets = list(feedback.get("affected_targets", []))
        targets = [
            target for target in original_targets
            if target not in inactive_targets
        ]
        verdict = original_verdict
        if verdict == "keep":
            targets = []
        elif not targets:
            verdict = "keep"

        if verdict != original_verdict or targets != original_targets:
            feedback["verdict"] = verdict
            feedback["affected_targets"] = targets
            normalizations.append({
                "feature": feature,
                "before": {
                    "verdict": original_verdict,
                    "affected_targets": original_targets,
                },
                "after": {
                    "verdict": verdict,
                    "affected_targets": targets,
                },
                "reason": (
                    "feedback referenced a copy role that is not rendered"
                    if any(
                        target in inactive_targets
                        for target in original_targets
                    )
                    else "keep verdict cannot claim redesigned targets"
                ),
            })
    return normalizations


def _normalize_target_layout_roles(review: dict, layout: dict) -> list[dict]:
    """Remove VLM-invented copy roles while preserving missing-role errors."""
    expected_roles = {
        str(item["role"]) for item in layout["elements"]
    }
    elements = review["target_layout"]["elements"]
    kept = [
        item
        for item in elements
        if str(item["role"]) in expected_roles
    ]
    removed_roles = [
        str(item["role"])
        for item in elements
        if str(item["role"]) not in expected_roles
    ]
    review["target_layout"]["elements"] = kept
    if not removed_roles:
        return []
    return [{
        "feature": "target_layout.elements",
        "before": {"roles": [str(item["role"]) for item in elements]},
        "after": {"roles": [str(item["role"]) for item in kept]},
        "reason": (
            "removed copy roles invented by the VLM because their source "
            "copy is empty"
        ),
        "removed_roles": removed_roles,
    }]


def _review_audit_warnings(review: dict) -> list[str]:
    """Report diagnosis coverage issues without rejecting a valid target."""
    warnings = []
    observations = review["diagnosis"]["design_observations"]
    if len(observations) < len(FINAL_REVIEW_FEATURES):
        warnings.append(
            "completed-ad audit has fewer observations than design categories"
        )
    observation_keys = {
        (
            item["assessment"], item["category"], item["target"],
            item["evidence"].strip(),
        )
        for item in observations
    }
    if len(observation_keys) != len(observations):
        warnings.append("completed-ad audit contains duplicate observations")
    categories = {item["category"] for item in observations}
    missing_categories = set(FINAL_REVIEW_FEATURES) - categories
    if missing_categories:
        warnings.append(
            "completed-ad audit is missing design categories: "
            + ", ".join(sorted(missing_categories))
        )
    return warnings


def _review_consistency_issues(review: dict, layout: dict) -> list[str]:
    """Validate render-critical target completeness and accountability."""
    issues = []
    feature_reviews = review["diagnosis"]["feature_reviews"]
    for feature in FINAL_REVIEW_FEATURES:
        feedback = feature_reviews[feature]
        targets = feedback["affected_targets"]
        if feedback["verdict"] == "revise" and not targets:
            issues.append(f"{feature} needs revision but names no affected targets")

    expected_roles = {str(item["role"]) for item in layout["elements"]}
    target_roles = [str(item["role"]) for item in review["target_layout"]["elements"]]
    if len(target_roles) != len(set(target_roles)):
        issues.append("target_layout contains duplicate element roles")
    if set(target_roles) != expected_roles:
        issues.append(
            "target_layout element roles must exactly match the rendered roles: "
            f"expected {sorted(expected_roles)}, received {sorted(set(target_roles))}"
        )

    surface_groups = [
        str(item["group"]) for item in review["target_layout"]["surfaces"]
    ]
    if len(surface_groups) != len(set(surface_groups)):
        issues.append("target_layout contains duplicate surface groups")
    if not set(surface_groups).issubset({"headline", "offer"}):
        issues.append("target_layout contains an unsupported surface group")
    return issues


def _design_deliberation_warnings(review: dict) -> list[str]:
    """Audit whether the structured feature deliberation supports its selection."""
    warnings = []
    exploration = review["design_exploration"]
    feature_reviews = review["diagnosis"]["feature_reviews"]
    for feature in FINAL_REVIEW_FEATURES:
        decision = exploration[feature]
        directions = [
            str(item["direction"]).strip()
            for item in decision["candidate_evaluations"]
        ]
        if len({direction.casefold() for direction in directions}) != len(directions):
            warnings.append(f"{feature} compares duplicate candidate directions")
        selected = str(decision["selected_direction"]).strip().casefold()
        if selected not in {direction.casefold() for direction in directions}:
            warnings.append(
                f"{feature} selected_direction does not match an evaluated candidate"
            )
        if not any(
            dependency != feature for dependency in decision["interacts_with"]
        ):
            warnings.append(f"{feature} names no cross-feature interaction")
        if len(set(decision["interacts_with"])) != len(decision["interacts_with"]):
            warnings.append(f"{feature} repeats cross-feature interactions")
        commitments = decision["target_layout_commitments"]
        if len(set(commitments)) != len(commitments):
            warnings.append(f"{feature} repeats target-layout commitments")
        verdict = feature_reviews[feature]["verdict"]
        if verdict == "revise" and not commitments:
            warnings.append(f"{feature} revision has no target-layout commitment")
        if verdict == "keep" and commitments:
            warnings.append(f"{feature} keep decision claims target-layout commitments")
    for index, relationship in enumerate(
        review["coherence_review"]["cross_feature_decisions"]
    ):
        features = relationship["features"]
        if len(set(features)) != len(features):
            warnings.append(
                f"coherence relationship {index} repeats design features"
            )
    return warnings


def _layout_state(layout: dict) -> dict:
    """Serialize a canonical rendered state, including visual defaults."""
    elements = []
    for item in layout["elements"]:
        role = str(item["role"])
        state = {
            "role": role,
            "x": int(item["x"]), "y": int(item["y"]),
            "width": int(item["width"]), "height": int(item["height"]),
            "font_size": int(item["font_size"]),
            "font_weight": int(item.get("font_weight", 600)),
            "tracking": int(item.get("tracking", 0)),
            "wrap_mode": str(item.get("wrap_mode", "character")),
            "min_font_size": int(item.get("min_font_size", 8)),
            "optical_align": bool(item.get("optical_align", True)),
            "text_align": str(item.get("text_align", "left")),
            "max_lines": int(item.get("max_lines", 1 if role == "title" else 2)),
            "line_height": float(item.get("line_height", 1.2)),
            "color": str(item.get("color", "#FFFFFF")),
            "shadow_offset": int(item.get("shadow_offset", 0)),
            "shadow_color": str(item.get("shadow_color", "#000000")),
            "stroke_width": int(item.get("stroke_width", 0)),
            "stroke_color": str(item.get("stroke_color", "#000000")),
        }
        if role == "price":
            state.update({
                "number_scale": float(item.get("number_scale", 1.22)),
                "unit_scale": float(item.get("unit_scale", 0.80)),
                "number_baseline_shift": float(
                    item.get("number_baseline_shift", 0.0)
                ),
                "unit_baseline_shift": float(
                    item.get("unit_baseline_shift", 0.0)
                ),
                "baseline_mode": str(item.get("baseline_mode", "shared")),
            })
        elements.append(state)
    return {
        "elements": elements,
        "surfaces": [
            {
                "group": str(item["id"]).removeprefix("surface-"),
                "x": int(item["x"]), "y": int(item["y"]),
                "width": int(item["width"]), "height": int(item["height"]),
                "fill_type": str(item.get("fill_type", "solid")),
                "fill_colors": list(item.get("fill_colors", ["#000000"] * 2)),
                "fill_stops": [
                    float(value) for value in item.get("fill_stops", [0.0, 1.0])
                ],
                "gradient_angle": float(item.get("gradient_angle", 0.0)),
                "shape": str(item.get("shape", "rounded_rect")),
                "overlay_color": str(item.get("overlay_color", "#000000")),
                "overlay_opacity": float(item.get("overlay_opacity", 0.0)),
                "opacity": float(item.get("opacity", 1.0)),
                "corner_radius": int(item.get("border_radius", 0)),
                "backdrop_blur": int(item.get("backdrop_blur", 0)),
                "blend_mode": str(item.get("blend_mode", "normal")),
                "border_enabled": bool(item.get("border_enabled", False)),
                "border_color": str(item.get("border_color", "#000000")),
                "border_width": int(item.get("border_width", 0)),
                "border_opacity": float(item.get("border_opacity", 0.0)),
                "shadow_enabled": bool(item.get("shadow_enabled", False)),
                "shadow_color": str(item.get("shadow_color", "#000000")),
                "shadow_offset_x": int(item.get("shadow_offset_x", 0)),
                "shadow_offset_y": int(item.get("shadow_offset_y", 0)),
                "shadow_blur": int(item.get("shadow_blur", 0)),
                "shadow_opacity": float(item.get("shadow_opacity", 0.0)),
                "shadow_layers": list(item.get("shadow_layers", [])),
            }
            for item in layout.get("underlays", [])
            if str(item.get("id", "")).startswith("surface-")
        ],
        "accent_rule": next(
            (
                {
                    "present": True,
                    "x": int(item["x"]), "y": int(item["y"]),
                    "width": int(item["width"]), "height": int(item["height"]),
                    "background_color": str(
                        (item.get("fill_colors") or ["#000000"])[0]
                    ),
                }
                for item in layout.get("underlays", [])
                if item.get("id") == "accent-rule"
            ),
            {"present": False},
        ),
    }


def _material_revision_summary(
    before: dict,
    after: dict,
    canvas: dict,
) -> dict:
    """Measure meaningful state changes without comparing rendered pixels."""
    x_threshold = max(4, round(int(canvas["width"]) * 0.015))
    y_threshold = max(4, round(int(canvas["height"]) * 0.015))
    systems: dict[str, set[str]] = {
        name: set()
        for name in ("composition", "typography", "surface", "color", "price", "accent")
    }
    changed_targets = set()
    properties = set()

    def record(system: str, target: str, property_name: str) -> None:
        systems[system].add(target)
        changed_targets.add(target)
        properties.add((target, property_name))

    before_elements = {item["role"]: item for item in before["elements"]}
    for item in after["elements"]:
        role = item["role"]
        previous = before_elements[role]
        for key in ("x", "width"):
            if abs(int(item[key]) - int(previous[key])) >= x_threshold:
                record("composition", f"{role}_geometry", key)
                changed_targets.add("overall_composition")
        for key in ("y", "height"):
            if abs(int(item[key]) - int(previous[key])) >= y_threshold:
                record("composition", f"{role}_geometry", key)
                changed_targets.add("overall_composition")
        if abs(int(item["font_size"]) - int(previous["font_size"])) >= 2:
            record("typography", f"{role}_typography", "font_size")
        for key in (
            "font_weight", "tracking", "wrap_mode", "min_font_size",
            "optical_align", "text_align", "max_lines",
            "shadow_offset", "stroke_width",
        ):
            if item[key] != previous[key]:
                record("typography", f"{role}_typography", key)
        if abs(float(item["line_height"]) - float(previous["line_height"])) >= 0.1:
            record("typography", f"{role}_typography", "line_height")
        for key in ("color", "shadow_color", "stroke_color"):
            if item[key] != previous[key]:
                record("color", "color_palette", f"{role}.{key}")
        if role == "price":
            for key, threshold in (
                ("number_scale", 0.06), ("unit_scale", 0.06),
                ("number_baseline_shift", 0.025),
                ("unit_baseline_shift", 0.025),
            ):
                if abs(float(item[key]) - float(previous[key])) >= threshold:
                    record("price", "price_composition", key)
            if item["baseline_mode"] != previous["baseline_mode"]:
                record("price", "price_composition", "baseline_mode")

    before_surfaces = {item["group"]: item for item in before["surfaces"]}
    after_surfaces = {item["group"]: item for item in after["surfaces"]}
    for group in sorted(set(before_surfaces) | set(after_surfaces)):
        previous = before_surfaces.get(group)
        item = after_surfaces.get(group)
        if previous is None or item is None:
            record("surface", f"{group}_surface", "presence")
            changed_targets.add("overall_composition")
            continue
        for key in ("x", "width"):
            if abs(int(item[key]) - int(previous[key])) >= x_threshold:
                record("surface", f"{group}_surface", key)
                changed_targets.add("overall_composition")
        for key in ("y", "height"):
            if abs(int(item[key]) - int(previous[key])) >= y_threshold:
                record("surface", f"{group}_surface", key)
                changed_targets.add("overall_composition")
        for key, threshold in (
            ("opacity", 0.04), ("gradient_angle", 3.0),
            ("border_opacity", 0.04), ("shadow_opacity", 0.04),
            ("overlay_opacity", 0.04),
        ):
            if abs(float(item[key]) - float(previous[key])) >= threshold:
                record("surface", f"{group}_surface", key)
        for key in (
            "fill_type", "fill_stops", "shape", "corner_radius",
            "backdrop_blur", "blend_mode", "border_enabled", "border_width",
            "shadow_enabled", "shadow_offset_x", "shadow_offset_y",
            "shadow_blur", "shadow_layers",
        ):
            if item[key] != previous[key]:
                record("surface", f"{group}_surface", key)
        for key in (
            "fill_colors", "overlay_color", "border_color", "shadow_color",
        ):
            if item[key] != previous[key]:
                record("color", "color_palette", f"{group}.{key}")
                changed_targets.add(f"{group}_surface")


    before_rule = before["accent_rule"]
    after_rule = after["accent_rule"]
    if before_rule.get("present") != after_rule.get("present"):
        record("accent", "accent_rule", "present")
    elif after_rule.get("present"):
        for key in ("x", "width"):
            if abs(int(after_rule[key]) - int(before_rule[key])) >= x_threshold:
                record("accent", "accent_rule", key)
        for key in ("y", "height"):
            if abs(int(after_rule[key]) - int(before_rule[key])) >= y_threshold:
                record("accent", "accent_rule", key)
        if after_rule["background_color"] != before_rule["background_color"]:
            record("accent", "accent_rule", "color")
            changed_targets.add("color_palette")

    active_systems = [name for name, targets in systems.items() if targets]
    return {
        "active_systems": active_systems,
        "systems": {name: sorted(targets) for name, targets in systems.items()},
        "changed_targets": sorted(changed_targets),
        "material_property_count": len(properties),
        "thresholds": {
            "horizontal_geometry_px": x_threshold,
            "vertical_geometry_px": y_threshold,
            "font_size_px": 2,
            "surface_opacity": 0.04,
            "price_scale": 0.06,
            "price_baseline": 0.025,
        },
    }


def _material_revision_issues(summary: dict) -> list[str]:
    """Require broad visible changes in the rebuilt target."""
    issues = []
    composition_targets = summary["systems"]["composition"]
    if len(composition_targets) < 2:
        issues.append(
            "rebuilt design must materially recompose at least two copy roles; "
            f"changed {len(composition_targets)}"
        )
    if len(summary["active_systems"]) < 3:
        issues.append(
            "rebuilt design must change at least three design systems; "
            f"changed {len(summary['active_systems'])}"
        )
    return issues


def _material_feedback_warnings(review: dict, summary: dict) -> list[str]:
    """Audit diagnosis-to-target claims without blocking a valid redesign."""
    warnings = []
    feature_reviews = review["diagnosis"]["feature_reviews"]
    revised_features = [
        feature for feature, feedback in feature_reviews.items()
        if feedback["verdict"] == "revise"
    ]
    changed_targets = set(summary["changed_targets"])
    claimed_targets = {
        target
        for feature in revised_features
        for target in feature_reviews[feature]["affected_targets"]
    }
    for feature in revised_features:
        claimed = set(feature_reviews[feature]["affected_targets"])
        if not claimed.intersection(changed_targets):
            warnings.append(
                f"{feature} claims targets with no material applied change"
            )
    for system in summary["active_systems"]:
        system_targets = set(summary["systems"][system])
        if system == "composition":
            system_targets.add("overall_composition")
        if not system_targets.intersection(claimed_targets):
            warnings.append(
                f"materially changed {system} system has no revise feedback"
            )
    return warnings


def _applied_changes(before: dict, after: dict) -> list[dict]:
    changes = []
    before_elements = {item["role"]: item for item in before["elements"]}
    for item in after["elements"]:
        previous = before_elements.get(item["role"], {})
        changed = {
            key: {"before": previous.get(key), "after": value}
            for key, value in item.items()
            if key != "role" and previous.get(key) != value
        }
        if changed:
            changes.append({"target": item["role"], "changes": changed})
    before_surfaces = {item["group"]: item for item in before["surfaces"]}
    after_surfaces = {item["group"]: item for item in after["surfaces"]}
    for group in sorted(set(before_surfaces) | set(after_surfaces)):
        previous = before_surfaces.get(group)
        item = after_surfaces.get(group)
        if previous is None or item is None:
            changes.append({
                "target": f"{group}_surface",
                "changes": {"state": {"before": previous, "after": item}},
            })
            continue
        changed = {
            key: {"before": previous.get(key), "after": value}
            for key, value in item.items()
            if key != "group" and previous.get(key) != value
        }
        if changed:
            changes.append({
                "target": f"{group}_surface", "changes": changed,
            })

    if before["accent_rule"] != after["accent_rule"]:
        changes.append({
            "target": "accent_rule",
            "changes": {
                "state": {
                    "before": before["accent_rule"],
                    "after": after["accent_rule"],
                }
            },
        })
    return changes


@dataclass(frozen=True)
class LayoutGenerationResult:
    output_dir: Path
    layout_json: Path
    first_vlm_image: Path
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
    image_path: Path | list[Path] | tuple[Path, ...],
    detail: str,
    schema_name: str,
    schema: dict,
    temperature: float,
) -> dict:
    image_paths = (
        list(image_path)
        if isinstance(image_path, (list, tuple))
        else [image_path]
    )
    content = [{"type": "input_text", "text": request_text}]
    content.extend(
        {
            "type": "input_image",
            "image_url": image_to_data_url(path),
            "detail": detail,
        }
        for path in image_paths
    )
    response = client.responses.create(
        model=model,
        instructions=instructions,
        input=[{"role": "user", "content": content}],
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
    """Validate the VLM's complete absolute target before applying it."""
    review["needs_revision"] = True
    review["feedback_normalizations"] = (
        _normalize_target_layout_roles(review, layout)
        + _normalize_feature_review_metadata(review, layout)
    )
    review["audit_warnings"] = _review_audit_warnings(review)
    review["deliberation_warnings"] = _design_deliberation_warnings(review)
    consistency_issues = _review_consistency_issues(review, layout)
    if consistency_issues:
        raise ValueError(
            "Final review target is inconsistent: "
            + "; ".join(consistency_issues)
        )

    candidate = apply_final_review_revision(layout, review)
    requested_summary = _material_revision_summary(
        _layout_state(layout), _layout_state(candidate), layout["canvas"]
    )
    material_issues = _material_revision_issues(requested_summary)
    if material_issues:
        raise ValueError(
            "Final review did not produce a material redesign: "
            + "; ".join(material_issues)
        )
    review["material_feedback_warnings"] = _material_feedback_warnings(
        review, requested_summary
    )
    review["consistency_validated"] = True
    review["deliberation_validated"] = not review["deliberation_warnings"]
    review["revision_mode"] = "completed_ad_rebuild"
    review["requested_material_changes"] = requested_summary
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
    """Create a draft and independently redesign one advertisement."""
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
        output_path=output_dir / "vlm_1_ad.png",
        font_path=font_path,
    )

    # Call 2: independently audit and redesign the rendered draft.

    design_candidate_pool = build_design_candidate_pool(
        computed_analysis,
        ad_copy,
        design_spec["scene_analysis"].get("subject_region"),
    )
    final_review = _request_json(
        client,
        model=model,
        instructions=FINAL_REVIEW_SYSTEM_PROMPT,
        request_text=build_final_review_request(
            ad_copy, computed_analysis, design_candidate_pool
        ),
        image_path=[draft_path, image_path],
        detail=detail,
        schema_name="completed_ad_final_independent_redesign",
        schema=FINAL_REVIEW_SCHEMA,
        temperature=temperature,
    )
    final_review = _enforce_final_review_revision(final_review, draft_layout)
    final_review["review_attempts"] = 1
    final_review["design_candidate_pool"] = design_candidate_pool
    before_redesign_state = _layout_state(draft_layout)
    final_layout = apply_final_review_revision(draft_layout, final_review)
    final_layout = fit_layout_typography(final_layout, font_path=font_path)
    final_layout = ensure_layout_contrast(image_path, final_layout)
    applied_final_state = _layout_state(final_layout)
    redesign_summary = _material_revision_summary(
        before_redesign_state, applied_final_state, final_layout["canvas"]
    )
    post_fit_issues = _material_revision_issues(redesign_summary)
    if post_fit_issues:
        raise ValueError(
            "Final redesign fitting erased material changes: "
            + "; ".join(post_fit_issues)
        )
    final_review["material_feedback_warnings"] = _material_feedback_warnings(
        final_review, redesign_summary
    )
    final_review["applied_target_layout"] = applied_final_state
    final_review["applied_material_changes"] = redesign_summary
    final_review["applied_changes"] = _applied_changes(
        before_redesign_state, applied_final_state
    )
    final_review["constraints_applied"] = final_layout.get(
        "final_review_constraints", []
    )
    layout_document = {
        "model": model,
        "source_image": str(image_path),
        "latency_sec": round(perf_counter() - started_at, 2),
        "copy": ad_copy,
        "design_rationale": design_spec["rationale"],
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

    keep_only(output_dir, (layout_path, draft_path, rendered_path))

    return LayoutGenerationResult(
        output_dir=output_dir,
        layout_json=layout_path,
        first_vlm_image=draft_path,
        rendered_image=rendered_path,
    )
