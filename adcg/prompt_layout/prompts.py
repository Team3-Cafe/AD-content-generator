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
You are the final senior art director reviewing the second-stage completed
advertisement. The supplied image already contains the finished background and
all rendered Korean copy. Diagnose the actual delivered composition, then
return a complete ABSOLUTE target layout in canvas pixels. This is not a list
of deltas, shifts, or scale multipliers.

Evaluate every feature in feature_reviews: typography, hierarchy, spacing,
price composition, band proportion, accent rule, placement, color, contrast,
CTA treatment, and product visibility. Give each one a concrete keep/revise
verdict based on visible evidence. A revised feature must name every affected
target; a kept feature must have an empty affected_targets list. Do not invent
problems merely to increase a count, but make a decisive redesign wherever the
finished image is visibly weak. Avoid token 5% changes that preserve the same
composition without resolving the diagnosed relationship.

The target_layout is the complete desired final state, not just changed fields.
Return every currently rendered copy role exactly once and return exactly one
headline and one offer surface. Coordinates, boxes, font sizes, band geometry,
and accent-rule geometry are absolute pixels within the supplied canvas.
Choose enough text-box width and height for the requested font size and line
count. Keep the title, price, and CTA on one line. Preserve the exact copy.

Judge Korean typography as a composed system. In particular, when the price
contains a large number plus surrounding qualifier/unit text, balance the
number scale, unit scale, and baselines so the number is emphasized without
looking detached or oversized. Review clipping, wrapping, optical centering,
line gaps, band padding, and hierarchy together instead of changing each value
independently. Do not issue mutually cancelling movements.

The CTA is plain typography in a static image, never a button, pill, outline,
or interactive control. Maintain product visibility. Color fields use palette
tokens or "keep"; use the current-state colors when a field is kept. Always set
needs_revision to true. Report concrete observed problems with exact targets,
visible evidence, actionable corrections, and severity. Never request new copy,
a new template, an alternative image, or another VLM review.
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
                "max_lines": item.get("max_lines", 2),
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
                "group": item.get("design_group"),
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
        "protected_subject": layout.get("design_groups", {}).get(
            "protected_subject"
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
        "Return a complete absolute-pixel target layout for the second-stage "
        "completed advertisement after diagnosing every design feature.\n\n"
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
