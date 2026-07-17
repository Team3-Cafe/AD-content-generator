from __future__ import annotations

import base64
import io
import json
from functools import lru_cache

from PIL import Image, ImageDraw


REFERENCE_SIZE = 512
GRID_SIZE = 5


EXAMPLE_SPECS = (
    {
        "name": "product_right_open_left",
        "protected": (280, 150, 215, 250),
        "copy": {
            "title": "Built for every shift",
            "subtitle": "Reliable equipment and trained operators",
            "cta": "Request a quote",
        },
        "placements": (
            ("title", "top_left", 24, 28, 260, 82, 34, 800, 1),
            ("subtitle", "middle_left", 24, 160, 220, 74, 19, 450, 3),
            ("cta", "bottom_left", 24, 408, 174, 54, 20, 700, 1),
        ),
        "role_underlays": {
            "subtitle": ("#101820", 0.72, "#FFFFFF"),
        },
    },
    {
        "name": "product_lower_center_open_corners",
        "protected": (145, 205, 225, 245),
        "copy": {
            "title": "Fresh comfort starts here",
            "subtitle": "Professional installation for your space",
            "cta": "Book today",
        },
        "placements": (
            ("title", "top_left", 24, 24, 270, 86, 33, 800, 1),
            ("subtitle", "top_right", 300, 42, 188, 88, 18, 450, 3),
            ("cta", "bottom_right", 320, 438, 168, 50, 20, 700, 1),
        ),
        "role_underlays": {
            "title": ("#F7F4EC", 0.90, "#101820"),
        },
    },
    {
        "name": "portrait_left_open_right",
        "protected": (24, 95, 245, 330),
        "copy": {
            "title": "Learn with confidence",
            "subtitle": "Personal guidance for every level",
            "cta": "Start your lesson",
        },
        "placements": (
            ("title", "top_right", 270, 34, 218, 98, 30, 800, 1),
            ("subtitle", "middle_right", 306, 190, 182, 76, 18, 450, 3),
            ("cta", "bottom_right", 318, 422, 170, 54, 19, 700, 1),
        ),
        "role_underlays": {
            "subtitle": ("#101820", 0.76, "#FFFFFF"),
        },
    },
    {
        "name": "wide_product_bottom_price_emphasis",
        "protected": (72, 280, 368, 175),
        "copy": {
            "title": "Power your next project",
            "subtitle": "Clean energy systems designed to last",
            "price": "From 990",
            "cta": "Get details",
        },
        "placements": (
            ("title", "top_center", 60, 24, 392, 76, 35, 800, 1),
            ("subtitle", "middle_left", 26, 150, 235, 70, 18, 450, 3),
            ("price", "middle_right", 322, 154, 166, 58, 27, 750, 1),
            ("cta", "bottom_right", 330, 448, 158, 44, 18, 700, 1),
        ),
        "role_underlays": {
            "title": ("#101820", 0.68, "#FFFFFF"),
        },
    },
    {
        "name": "product_right_lower_asymmetric",
        "protected": (250, 205, 245, 235),
        "copy": {
            "title": "Move more, wait less",
            "subtitle": "Fast handling by experienced professionals",
            "price": "Clear estimates",
            "cta": "Contact the team",
        },
        "placements": (
            ("title", "top_left", 24, 28, 280, 80, 34, 800, 1),
            ("subtitle", "middle_left", 24, 158, 205, 88, 18, 450, 3),
            ("price", "bottom_center", 170, 444, 150, 42, 21, 700, 1),
            ("cta", "bottom_left", 24, 390, 166, 48, 18, 700, 1),
        ),
        "role_underlays": {
            "subtitle": ("#F7F4EC", 0.88, "#101820"),
        },
    },
)


ROLE_PRIORITY = {
    "title": 1,
    "price": 2,
    "subtitle": 3,
    "cta": 4,
}


def _bbox(x: int, y: int, width: int, height: int) -> dict:
    return {
        "x": x,
        "y": y,
        "width": width,
        "height": height,
    }


def _scaled_bbox(
    values: tuple[int, int, int, int],
    canvas_width: int,
    canvas_height: int,
) -> tuple[int, int, int, int]:
    x, y, width, height = values
    scale_x = canvas_width / REFERENCE_SIZE
    scale_y = canvas_height / REFERENCE_SIZE
    return (
        int(round(x * scale_x)),
        int(round(y * scale_y)),
        max(1, int(round(width * scale_x))),
        max(1, int(round(height * scale_y))),
    )


def _scaled_placement(
    placement: tuple,
    canvas_width: int,
    canvas_height: int,
) -> tuple:
    (
        role,
        zone,
        x,
        y,
        width,
        height,
        font_size,
        font_weight,
        max_lines,
    ) = placement
    scaled_box = _scaled_bbox(
        (x, y, width, height),
        canvas_width,
        canvas_height,
    )
    font_scale = min(canvas_width, canvas_height) / REFERENCE_SIZE
    return (
        role,
        zone,
        *scaled_box,
        max(8, int(round(font_size * font_scale))),
        font_weight,
        max_lines,
    )


def _grid_fields(
    x: int,
    y: int,
    width: int,
    height: int,
    canvas_width: int,
    canvas_height: int,
) -> dict:
    col = min(GRID_SIZE - 1, x * GRID_SIZE // canvas_width)
    row = min(GRID_SIZE - 1, y * GRID_SIZE // canvas_height)
    end_col = min(
        GRID_SIZE,
        ((x + width) * GRID_SIZE + canvas_width - 1)
        // canvas_width,
    )
    end_row = min(
        GRID_SIZE,
        ((y + height) * GRID_SIZE + canvas_height - 1)
        // canvas_height,
    )
    col_span = max(1, end_col - col)
    row_span = max(1, end_row - row)
    region_x = round(col * canvas_width / GRID_SIZE)
    region_y = round(row * canvas_height / GRID_SIZE)
    region_end_x = round(
        (col + col_span) * canvas_width / GRID_SIZE
    )
    region_end_y = round(
        (row + row_span) * canvas_height / GRID_SIZE
    )
    return {
        "grid_row": row,
        "grid_col": col,
        "row_span": row_span,
        "col_span": col_span,
        "preferred_region": _bbox(
            region_x,
            region_y,
            max(1, region_end_x - region_x),
            max(1, region_end_y - region_y),
        ),
    }


def _plan_output(
    spec: dict,
    canvas_width: int = REFERENCE_SIZE,
    canvas_height: int = REFERENCE_SIZE,
) -> dict:
    px, py, pw, ph = _scaled_bbox(
        spec["protected"],
        canvas_width,
        canvas_height,
    )
    elements = []
    for raw_placement in spec["placements"]:
        (
            role,
            zone,
            x,
            y,
            width,
            height,
            _font_size,
            _font_weight,
            _max_lines,
        ) = _scaled_placement(
            raw_placement,
            canvas_width,
            canvas_height,
        )
        grid_fields = _grid_fields(
            x,
            y,
            width,
            height,
            canvas_width,
            canvas_height,
        )
        elements.append(
            {
                "role": role,
                "content": spec["copy"][role],
                "priority": ROLE_PRIORITY[role],
                **grid_fields,
                "alignment": (
                    "center" if zone.endswith("center") else
                    "right" if zone.endswith("right") else
                    "left"
                ),
                "relationship": (
                    "Primary headline" if role == "title" else
                    "Supporting information" if role == "subtitle" else
                    "Commercial emphasis" if role == "price" else
                    "Independent action element"
                ),
                "rationale": (
                    f"Use the {zone} negative space without covering "
                    "the protected subject."
                ),
            }
        )

    return {
        "scene_summary": (
            "A dominant commercial subject occupies the protected region; "
            "the remaining negative space supports distributed copy."
        ),
        "protected_regions": [
            {
                "label": "primary commercial subject",
                "importance": "critical",
                "bbox": _bbox(px, py, pw, ph),
                "reason": "Product identity and silhouette must stay visible.",
            }
        ],
        "safe_regions": [
            {
                "label": "negative space",
                "importance": "background",
                "bbox": _bbox(
                    *_scaled_bbox(
                        (16, 16, 480, 480),
                        canvas_width,
                        canvas_height,
                    )
                ),
                "reason": (
                    "Use only portions that do not intersect the protected "
                    "subject."
                ),
            }
        ],
        "placement_plan": (
            "Assign each semantic role its own cells on the normalized 5x5 "
            "grid. Preserve shared alignment while keeping CTA and price "
            "visually distinct."
        ),
        "elements": elements,
        "visual_strategy": {
            "text_color": "#FFFFFF",
            "accent_color": "#FFD23F",
            "underlay_recommended": True,
            "underlay_color": "#101820",
            "underlay_opacity": 0.72,
        },
        "risk_notes": [
            "Avoid narrow boxes that create orphan characters.",
            "Do not give every role identical visual weight.",
        ],
    }


def _layout_output(
    spec: dict,
    canvas_width: int = REFERENCE_SIZE,
    canvas_height: int = REFERENCE_SIZE,
) -> dict:
    elements = []
    underlays = []
    for raw_placement in spec["placements"]:
        (
            role,
            _zone,
            x,
            y,
            width,
            height,
            font_size,
            font_weight,
            max_lines,
        ) = _scaled_placement(
            raw_placement,
            canvas_width,
            canvas_height,
        )
        element_id = f"text-{role}"
        role_underlay = spec.get("role_underlays", {}).get(role)
        elements.append(
            {
                "id": element_id,
                "role": role,
                "content": spec["copy"][role],
                **_bbox(x, y, width, height),
                "z_index": 2,
                "text_align": (
                    "center" if role == "price" else "left"
                ),
                "vertical_align": "center",
                "font_size": font_size,
                "font_weight": font_weight,
                "line_height": 1.16,
                "color": (
                    role_underlay[2] if role_underlay else
                    "#101820" if role == "cta" else
                    "#FFD23F" if role == "price" else
                    "#FFFFFF"
                ),
                "max_lines": max_lines,
            }
        )
        if role == "cta" or role_underlay:
            pad_x = max(
                4,
                int(round(10 * canvas_width / REFERENCE_SIZE)),
            )
            pad_y = max(
                3,
                int(round(6 * canvas_height / REFERENCE_SIZE)),
            )
            radius = max(
                4,
                int(
                    round(
                        12
                        * min(canvas_width, canvas_height)
                        / REFERENCE_SIZE
                    )
                ),
            )
            background_color = (
                role_underlay[0] if role_underlay else "#FFD23F"
            )
            opacity = role_underlay[1] if role_underlay else 0.96
            underlays.append(
                {
                    "id": f"underlay-{role}",
                    "target_ids": [element_id],
                    **_bbox(
                        max(0, x - pad_x),
                        max(0, y - pad_y),
                        min(canvas_width - max(0, x - pad_x), width + 2 * pad_x),
                        min(canvas_height - max(0, y - pad_y), height + 2 * pad_y),
                    ),
                    "z_index": 1,
                    "background_color": background_color,
                    "opacity": opacity,
                    "border_radius": radius,
                }
            )

    return {
        "canvas": {
            "width": canvas_width,
            "height": canvas_height,
        },
        "elements": elements,
        "underlays": underlays,
        "rationale": (
            "Semantic hierarchy is expressed through scale, weight, color, "
            "and distributed negative-space placement."
        ),
        "warnings": [],
    }


@lru_cache(maxsize=32)
def _example_image_data_url(
    index: int,
    canvas_width: int,
    canvas_height: int,
) -> str:
    spec = EXAMPLE_SPECS[index]
    image = Image.new(
        "RGB",
        (canvas_width, canvas_height),
        (61 + index * 10, 78 + index * 6, 94 + index * 5),
    )
    draw = ImageDraw.Draw(image)
    grid_step = max(
        16,
        int(round(32 * canvas_height / REFERENCE_SIZE)),
    )
    for offset in range(0, canvas_height, grid_step):
        shade = 78 + (offset // grid_step) % 2 * 8
        draw.line(
            (0, offset, canvas_width, offset),
            fill=(shade, shade, shade),
        )

    x, y, width, height = _scaled_bbox(
        spec["protected"],
        canvas_width,
        canvas_height,
    )
    visual_scale = min(canvas_width, canvas_height) / REFERENCE_SIZE
    draw.rounded_rectangle(
        (x, y, x + width, y + height),
        radius=max(4, int(round(22 * visual_scale))),
        fill=(220, 150 + index * 8, 45 + index * 10),
        outline=(255, 230, 170),
        width=max(1, int(round(4 * visual_scale))),
    )
    draw.ellipse(
        (
            x + width * 0.12,
            y + height * 0.72,
            x + width * 0.36,
            y + height * 0.96,
        ),
        fill=(30, 35, 40),
    )
    draw.ellipse(
        (
            x + width * 0.64,
            y + height * 0.72,
            x + width * 0.88,
            y + height * 0.96,
        ),
        fill=(30, 35, 40),
    )
    draw.text(
        (
            x + max(4, int(round(12 * visual_scale))),
            y + max(4, int(round(12 * visual_scale))),
        ),
        "PROTECTED SUBJECT",
        fill=(20, 20, 20),
    )

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def build_few_shot_content(
    stage: str,
    canvas_width: int,
    canvas_height: int,
) -> list[dict]:
    if stage not in {"plan", "layout"}:
        raise ValueError("stage must be 'plan' or 'layout'.")
    if canvas_width < 1 or canvas_height < 1:
        raise ValueError("canvas dimensions must be positive.")

    content = [
        {
            "type": "input_text",
            "text": (
                "Study these five multimodal design demonstrations before "
                "solving the test sample. Each demonstration pairs a visual "
                "composition and semantic copy roles with an approved output."
            ),
        }
    ]
    for index, spec in enumerate(EXAMPLE_SPECS):
        example_input = {
            "canvas": {
                "width": canvas_width,
                "height": canvas_height,
            },
            "scene_pattern": spec["name"],
            "copy_elements": spec["copy"],
        }
        expected = (
            _plan_output(spec, canvas_width, canvas_height)
            if stage == "plan"
            else _layout_output(spec, canvas_width, canvas_height)
        )
        content.extend(
            [
                {
                    "type": "input_text",
                    "text": (
                        f"Demonstration {index + 1} input:\n"
                        + json.dumps(
                            example_input,
                            ensure_ascii=False,
                            indent=2,
                        )
                    ),
                },
                {
                    "type": "input_image",
                    "image_url": _example_image_data_url(
                        index,
                        canvas_width,
                        canvas_height,
                    ),
                    "detail": "low",
                },
                {
                    "type": "input_text",
                    "text": (
                        f"Demonstration {index + 1} approved {stage} output:\n"
                        + json.dumps(
                            expected,
                            ensure_ascii=False,
                            indent=2,
                        )
                    ),
                },
            ]
        )
    return content


__all__ = ["EXAMPLE_SPECS", "build_few_shot_content"]
