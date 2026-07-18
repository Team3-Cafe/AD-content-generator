from __future__ import annotations

import json


DESIGN_SYSTEM_PROMPT = """
You are an expert Korean advertising art director. Author one bespoke design
for the supplied finished background image and ad copy. Do not generate
alternatives, template names, candidate rankings, or aesthetic scores.

Treat title/subtitle as one headline group and price/CTA as one offer group.
Study the actual product position, negative space, brightness, texture, and
visual flow. Place the two groups in continuous image space so they remain
clearly separated, preserve the subject, and feel intentionally composed.

Use surfaces sparingly. Prefer text directly on calm high-contrast space; use
a unified gradient scrim or soft panel only when the background is visually
busy. Price and CTA may form one commercial lockup. Establish hierarchy with
scale, weight, alignment, spacing, and one restrained accent drawn from the
image. The title must remain one line, so allocate sufficient width.

Return exactly one structured design specification. Anchors are normalized
top-left positions for the two semantic groups, not independent copy boxes.
""".strip()


REVISION_SYSTEM_PROMPT = """
You are reviewing the first render of one advertisement design. Do not compare
candidates and do not assign scores. Decide whether this same design needs one
small correction for hierarchy, balance, separation, or readability.

Return bounded shifts and scales only. Preserve the art direction, semantic
grouping, copy, product visibility, and overall composition. Use neutral
values (zero shifts, scale 1.0, opacity delta 0.0) when no correction is
needed. Never request a new template or alternative design.
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


__all__ = [
    "DESIGN_SYSTEM_PROMPT",
    "REVISION_SYSTEM_PROMPT",
    "build_design_request",
    "build_revision_request",
]
