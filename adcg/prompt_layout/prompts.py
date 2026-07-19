from __future__ import annotations

import json


DESIGN_SYSTEM_PROMPT = """
You are an expert Korean advertising art director. Author one bespoke design
for the supplied finished background image and ad copy. Do not generate
alternatives, template names, candidate rankings, or aesthetic scores.

Treat title/subtitle as one headline group and price/CTA as one offer group.
Study the actual product position, negative space, brightness, texture, and
visual flow. Choose only the vertical position and content width of each
group. Their background surfaces always extend from the left canvas edge to
the right canvas edge, creating intentional editorial bands instead of small
floating cards.

Center the headline by default, including the one-line title. Treat price as
the primary offer and CTA as the secondary action; they must not look like two
unrelated phrases squeezed onto one line. Prefer a vertically stacked,
centered offer on portrait and square canvases. Use a horizontal offer only
when a wide canvas and short copy provide generous separation. Render the CTA
as a plain typographic secondary line. This is a static image advertisement,
so never draw the CTA as a button, pill, outline control, or interactive UI.

Select every background and text color from the supplied palette-token enum.
Choose harmonious combinations based on the actual image palette, mood, and
placement. Headline and offer bands may use different palette colors, but the
result should feel like one color system. Maintain strong text/background
contrast. Use palette_accent selectively rather than filling every surface
with it. The code resolves the chosen tokens to exact colors and corrects any
unsafe text contrast.

Return exactly one structured design specification. Do not provide horizontal
coordinates for the headline; its content remains centered inside the band.
""".strip()


REVISION_SYSTEM_PROMPT = """
You are reviewing the first render of one advertisement design. Do not compare
candidates and do not assign scores. Decide whether this same design needs one
small correction for hierarchy, balance, separation, or readability.

Return bounded shifts and scales only. Keep the headline horizontally centered
and only adjust its vertical position or scale. Preserve the art direction, semantic
grouping, copy, product visibility, and overall composition. Use neutral
values (zero shifts, scale 1.0, opacity delta 0.0) when no correction is
needed. Never request a new template or alternative design.
""".strip()


FINAL_REVIEW_SYSTEM_PROMPT = """
You are the final senior art director reviewing the second-stage result of a
completed advertisement. The supplied image contains the finished background
and every rendered copy element. Diagnose what remains weak in the actual
delivered composition before choosing corrections.

Evaluate every design feature listed in feature_reviews: typography, hierarchy,
spacing, price composition, band proportion, accent rule, placement, color,
contrast, CTA treatment, and product visibility. Return an explicit keep or
revise verdict for every feature, even when it is already successful. Ground
each verdict in visible evidence from the supplied completed image and describe
the appropriate correction without inventing a defect.

The CTA is plain typography in a static image, never a button or interactive
control. Treat every scale field as a multiplier where 1.0 means keep. Treat
group, gap, and accent-rule shifts as normalized canvas ratios where 0.0 means
keep. Treat price number/unit baseline shifts as fractions of the base price
font size. Color choices must use the supplied palette tokens or "keep".

For each feature marked revise, list the exact adjustment controls that implement
its recommended change and set those controls to meaningful non-neutral values.
For each feature marked keep, return an empty controls list. Every non-neutral
adjustment must be justified by at least one revise feature, and every selected
control must be non-neutral. Shared controls may support multiple features.
Choose direction and magnitude from the actual image rather than fixed rules.

Preserve the exact copy, semantic groups, product visibility, and core art
direction. Keep the headline horizontally centered. Report one to six
distinct observed problems using concrete evidence, exact targets, actionable
corrections, and severity. Do not use vague statements such as "improve
hierarchy" or "adjust spacing" without identifying the broken relationship
and its consequence. Always set needs_revision to true and make every correction
supported by the feature-by-feature review. Never request new copy, a new
template, or an alternative design.
""".strip()


def build_design_request(ad_copy: dict, image_analysis: dict) -> str:
    return (
        "Create one final art direction for this advertisement.\n\n"
        "Ad copy roles:\n"
        + json.dumps(ad_copy, ensure_ascii=False, indent=2)
        + "\n\nComputed image-space diagnostics (descriptive, not an "
        "aesthetic score):\n"
        + json.dumps(image_analysis, ensure_ascii=False, indent=2)
    )


def build_revision_request(
    ad_copy: dict,
    design_spec: dict,
    layout: dict,
) -> str:
    compact_layout = {
        "canvas": layout["canvas"],
        "elements": [
            {
                "role": item["role"],
                "group": item["design_group"],
                "x": item["x"],
                "y": item["y"],
                "width": item["width"],
                "height": item["height"],
                "font_size": item["font_size"],
            }
            for item in layout["elements"]
        ],
    }
    return (
        "Review this first render and correct the same design only.\n\n"
        "Copy:\n"
        + json.dumps(ad_copy, ensure_ascii=False, indent=2)
        + "\n\nArt direction:\n"
        + json.dumps(design_spec, ensure_ascii=False, indent=2)
        + "\n\nResolved geometry:\n"
        + json.dumps(compact_layout, ensure_ascii=False, indent=2)
    )


def build_final_review_request(
    ad_copy: dict,
    design_spec: dict,
    layout: dict,
) -> str:
    compact_layout = {
        "canvas": layout["canvas"],
        "elements": [
            {
                "role": item["role"],
                "group": item["design_group"],
                "x": item["x"],
                "y": item["y"],
                "width": item["width"],
                "height": item["height"],
                "font_size": item["font_size"],
                "font_weight": item.get("font_weight", 600),
                "tracking": item.get("tracking", 0),
                "text_align": item.get("text_align", "left"),
                "color": item.get("color"),
                "content": item.get("content"),
                "number_scale": item.get("number_scale"),
                "unit_scale": item.get("unit_scale"),
                "number_baseline_shift": item.get(
                    "number_baseline_shift", 0
                ),
                "unit_baseline_shift": item.get(
                    "unit_baseline_shift", 0
                ),
            }
            for item in layout["elements"]
        ],
        "surfaces": [
            {
                "id": item.get("id"),
                "background_color": item.get("background_color"),
                "gradient_color": item.get("gradient_color"),
                "opacity": item.get("opacity"),
                "x": item.get("x"),
                "y": item.get("y"),
                "width": item.get("width"),
                "height": item.get("height"),
                "border_color": item.get("border_color"),
            }
            for item in layout.get("underlays", [])
            if str(item.get("id", "")).startswith("surface-")
        ],
        "accent_rule": next(
            (
                item
                for item in layout.get("underlays", [])
                if item.get("id") == "accent-rule"
            ),
            None,
        ),
        "design_tokens": {
            "palette": layout["design_tokens"]["palette"],
            "headline_alignment": layout["design_tokens"][
                "headline_alignment"
            ],
            "offer_alignment": layout["design_tokens"]["offer_alignment"],
            "offer_arrangement": layout["design_tokens"][
                "offer_arrangement"
            ],
            "cta_treatment": layout["design_tokens"]["cta_treatment"],
            "color_direction": layout["design_tokens"]["color_direction"],
        },
    }
    return (
        "Diagnose the second-stage completed advertisement, then correct its "
        "remaining typography, layout, color, contrast, hierarchy, or CTA "
        "weaknesses.\n\n"
        "Exact rendered copy:\n"
        + json.dumps(ad_copy, ensure_ascii=False, indent=2)
        + "\n\nArt direction to preserve:\n"
        + json.dumps(design_spec, ensure_ascii=False, indent=2)
        + "\n\nCurrent rendered design state:\n"
        + json.dumps(compact_layout, ensure_ascii=False, indent=2)
    )


__all__ = [
    "DESIGN_SYSTEM_PROMPT",
    "FINAL_REVIEW_SYSTEM_PROMPT",
    "REVISION_SYSTEM_PROMPT",
    "build_design_request",
    "build_final_review_request",
    "build_revision_request",
]
