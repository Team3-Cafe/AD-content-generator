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

Center the headline by default, including the one-line title. The offer may
use left, center, or right alignment when it supports the image flow. Select
solid, gradient, scrim, or accent band treatment according to the brightness
and visual detail at the chosen vertical position. The code will derive the
actual band colors from the local background and image palette. Establish
hierarchy with scale, weight, spacing, and one restrained accent.

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
