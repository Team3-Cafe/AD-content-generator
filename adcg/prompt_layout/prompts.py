from __future__ import annotations

import json


ROLE_DESIGN_INTENT = {
    "title": {
        "meaning": "Primary promise or headline",
        "hierarchy": "highest",
        "font_size_ratio": "6-9% of the shorter canvas side",
        "font_weight": "700-900",
        "treatment": "large, concise, dominant, maximum 2 lines",
    },
    "subtitle": {
        "meaning": "Supporting explanation or credibility message",
        "hierarchy": "supporting",
        "font_size_ratio": "3.2-4.5% of the shorter canvas side",
        "font_weight": "400-550",
        "treatment": "readable body copy, maximum 3 lines",
    },
    "price": {
        "meaning": "Commercial offer or price emphasis",
        "hierarchy": "high emphasis",
        "font_size_ratio": "4.5-6.5% of the shorter canvas side",
        "font_weight": "650-850",
        "treatment": "accent color or emphasis block, maximum 2 lines",
    },
    "cta": {
        "meaning": "Action the viewer should take",
        "hierarchy": "action emphasis",
        "font_size_ratio": "3.5-4.8% of the shorter canvas side",
        "font_weight": "600-800",
        "treatment": "compact button-like element, maximum 2 lines",
    },
}


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

Treat every non-empty role as a separate layout element:
- title: primary headline and strongest visual hierarchy
- subtitle: supporting explanation, smaller than the title
- price: separate high-emphasis element only when non-empty
- cta: separate compact action element placed after the information hierarchy
Never concatenate two roles into one element or split one role into multiple
elements. Evaluate the available safe regions independently for every role.
Do not default to assigning all roles to one preferred zone. Use separate
zones when the image has enough negative space, while preserving visual
relationships through alignment, color, and hierarchy.

Planning examples:
- If a product occupies the center and right side, place the title in the
  upper-left negative space, give the subtitle its own smaller block nearby,
  and place CTA in a separate lower-left or lower-right safe region.
- If the product occupies the center, balance the canvas by placing title and
  subtitle in distinct top-side blocks and CTA in a separate bottom corner.
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
- Output exactly one separate box for each non-empty input role. Never merge
  title, subtitle, price, or CTA content into the same box.
- Keep every box fully inside the canvas with practical outer margins.
- Avoid protected semantic regions and unnecessary element overlap.
- Align related elements by a shared left, center, or right edge.
- Establish hierarchy: title is normally largest, subtitle supports title,
  price is prominent when present, and CTA is compact but readable.
- Keep title larger and heavier than subtitle. Place subtitle next to or below
  title only when that is the best content-aware choice; it must remain a
  distinct box with visible spacing. Omit price entirely when its input is
  empty.
- Do not default to a single vertical stack. Distribute independent role
  blocks across suitable negative space to balance visual weight.
- Place CTA in a safe region separate from title and subtitle whenever the
  image offers at least two viable safe regions. Make it read as an action
  element rather than another paragraph.
- If price exists, give it an independent emphasis block near the product or
  CTA without covering the product.
- Estimate box height and font size from the actual copy length.
- Interpret the semantic role before styling. Use the role-specific meaning,
  hierarchy, font-size ratio, weight, and treatment supplied in the test
  input. Do not give all roles the same font size, weight, or underlay style.
- Avoid orphan characters or syllables on their own final line. Make the box
  wider or adjust the font size while respecting the role hierarchy.
- Use underlays only when needed for readability. Each underlay may support
  exactly one text role, must fully contain its target box with padding, and
  must have a lower z-index.
- Colors must be CSS-compatible values such as #FFFFFF.
- Keep independent blocks visually coherent through a shared grid, consistent
  margins, and intentional alignment. Do not group every role into one panel.

Output coordinates in pixels, not normalized values or percentages.
""".strip()


def build_plan_request(copy: dict, width: int, height: int) -> str:
    copy_elements = [
        {
            "role": role,
            "content": content,
            "design_intent": ROLE_DESIGN_INTENT[role],
        }
        for role, content in copy.items()
    ]
    payload = {
        "task": "content-aware placement plan",
        "canvas": {"width": width, "height": height},
        "copy_elements": copy_elements,
        "element_type_constraint": list(copy),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_layout_request(
    copy: dict,
    plan: dict,
    width: int,
    height: int,
) -> str:
    copy_elements = [
        {
            "role": role,
            "content": content,
            "design_intent": ROLE_DESIGN_INTENT[role],
        }
        for role, content in copy.items()
    ]
    payload = {
        "task": "generate the final layout from the placement plan",
        "canvas": {"width": width, "height": height},
        "copy_elements": copy_elements,
        "element_type_constraint": list(copy),
        "placement_plan": plan,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)
