from __future__ import annotations

import json


PLAN_SYSTEM_PROMPT = """
You are a senior advertising art director. Create a content-aware placement
plan for copy on a completed advertising background.

Use the two-stage method from content-aware ad layout research. This is the
planning stage only:
1. Understand the depicted products, people, faces, logos, tools, and other
   semantically important regions.
2. Identify protected regions that copy must not cover and safe regions where
   copy can be placed.
3. Plan the hierarchy and relationships of the supplied copy elements.

Do not invent or rewrite copy. Preserve each supplied string exactly. Empty
copy fields are omitted and must not be added. Bounding boxes use the original
canvas pixel coordinates. Maintain useful outer margins, coherent alignment,
and enough room for the real text length. Recommend an underlay only when the
background does not provide reliable contrast.

Planning examples:
- If a product occupies the center and right side, stack title and subtitle in
  the left negative space, align their left edges, and place CTA beneath them.
- If important objects fill both sides but the bottom is quiet, use a compact
  bottom-center copy group with a single underlay behind the group.
- Never cover a face, product identity feature, existing logo, or functional
  product detail merely because the region has low visual contrast.
""".strip()


LAYOUT_SYSTEM_PROMPT = """
You are a senior advertising layout designer. Convert an approved placement
plan into one precise pixel layout on the supplied image.

Follow the plan before choosing coordinates. Preserve every copy string
exactly and output one element for each supplied non-empty role, with no extra
text. Use the canvas dimensions exactly.

Constraints:
- Keep every box fully inside the canvas with practical outer margins.
- Avoid protected semantic regions and unnecessary element overlap.
- Align related elements by a shared left, center, or right edge.
- Establish hierarchy: title is normally largest, subtitle supports title,
  price is prominent when present, and CTA is compact but readable.
- Estimate box height and font size from the actual copy length.
- Use underlays only when needed for readability. An underlay must fully
  contain all target boxes with padding and have a lower z-index.
- Colors must be CSS-compatible values such as #FFFFFF.
- Prefer a single coherent copy group over scattered independent boxes.

Output coordinates in pixels, not normalized values or percentages.
""".strip()


def build_plan_request(copy: dict, width: int, height: int) -> str:
    payload = {
        "task": "content-aware placement plan",
        "canvas": {"width": width, "height": height},
        "copy_elements": copy,
        "element_type_constraint": list(copy),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_layout_request(
    copy: dict,
    plan: dict,
    width: int,
    height: int,
) -> str:
    payload = {
        "task": "generate the final layout from the placement plan",
        "canvas": {"width": width, "height": height},
        "copy_elements": copy,
        "element_type_constraint": list(copy),
        "placement_plan": plan,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)
