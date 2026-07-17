from __future__ import annotations

import colorsys
from copy import deepcopy
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageColor


STYLE_NAMES = (
    "minimal",
    "glass",
    "bold",
    "premium",
    "industrial",
)


STYLE_PRESETS = {
    "minimal": {
        "panels": ("cta",),
        "panel_mode": "accent",
        "opacity": 0.94,
        "gradient": False,
        "blur": 0,
        "border_width": 0,
        "radius_ratio": 0.18,
        "shadow": 2,
        "stroke": 0,
        "tracking": 0,
        "scales": {
            "title": 1.00,
            "subtitle": 0.96,
            "price": 1.04,
            "cta": 0.96,
        },
    },
    "glass": {
        "panels": ("title", "subtitle", "cta"),
        "panel_mode": "dark",
        "opacity": 0.58,
        "gradient": True,
        "blur": 12,
        "border_width": 1,
        "radius_ratio": 0.20,
        "shadow": 2,
        "stroke": 0,
        "tracking": 0,
        "scales": {
            "title": 0.98,
            "subtitle": 0.94,
            "price": 1.08,
            "cta": 0.94,
        },
    },
    "bold": {
        "panels": ("price", "cta"),
        "panel_mode": "accent",
        "opacity": 0.96,
        "gradient": True,
        "blur": 0,
        "border_width": 0,
        "radius_ratio": 0.10,
        "shadow": 3,
        "stroke": 1,
        "tracking": 1,
        "scales": {
            "title": 1.10,
            "subtitle": 0.98,
            "price": 1.18,
            "cta": 1.00,
        },
    },
    "premium": {
        "panels": ("title", "cta"),
        "panel_mode": "dark",
        "opacity": 0.72,
        "gradient": True,
        "blur": 6,
        "border_width": 1,
        "radius_ratio": 0.08,
        "shadow": 2,
        "stroke": 0,
        "tracking": 1,
        "scales": {
            "title": 1.04,
            "subtitle": 0.94,
            "price": 1.12,
            "cta": 0.92,
        },
    },
    "industrial": {
        "panels": ("title", "cta"),
        "panel_mode": "dark",
        "opacity": 0.90,
        "gradient": False,
        "blur": 0,
        "border_width": 2,
        "radius_ratio": 0.02,
        "shadow": 3,
        "stroke": 1,
        "tracking": 1,
        "scales": {
            "title": 1.08,
            "subtitle": 0.98,
            "price": 1.14,
            "cta": 1.00,
        },
    },
}


def _hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02X}{:02X}{:02X}".format(*rgb)


def _luminance(rgb: tuple[int, int, int]) -> float:
    return (
        0.2126 * rgb[0]
        + 0.7152 * rgb[1]
        + 0.0722 * rgb[2]
    ) / 255


@lru_cache(maxsize=16)
def extract_palette(image_path: str | Path) -> dict[str, str]:
    """Extract restrained dark, light, and accent colors from the image."""
    with Image.open(image_path) as source:
        image = source.convert("RGB")
        image.thumbnail((96, 96))
        quantized = image.quantize(colors=16)
        palette = quantized.getpalette() or []
        colors = []
        for count, index in quantized.getcolors() or []:
            offset = index * 3
            rgb = tuple(palette[offset:offset + 3])
            if len(rgb) == 3:
                colors.append((count, rgb))

    if not colors:
        return {
            "dark": "#101820",
            "light": "#F7F4EC",
            "accent": "#FFD23F",
        }

    dark = min(colors, key=lambda item: _luminance(item[1]))[1]
    light = max(colors, key=lambda item: _luminance(item[1]))[1]

    def accent_score(item) -> float:
        count, rgb = item
        hue, saturation, value = colorsys.rgb_to_hsv(
            rgb[0] / 255,
            rgb[1] / 255,
            rgb[2] / 255,
        )
        del hue
        usable_value = 1.0 if 0.20 <= value <= 0.92 else 0.25
        return saturation * usable_value * max(1, count) ** 0.25

    accent = max(colors, key=accent_score)[1]
    if colorsys.rgb_to_hsv(
        accent[0] / 255,
        accent[1] / 255,
        accent[2] / 255,
    )[1] < 0.22:
        accent = (255, 210, 63)
    return {
        "dark": _hex(dark),
        "light": _hex(light),
        "accent": _hex(accent),
    }


def _panel_color(mode: str, palette: dict[str, str]) -> str:
    if mode == "accent":
        return palette["accent"]
    if mode == "light":
        return palette["light"]
    return palette["dark"]


def _text_color_for_panel(
    panel_color: str,
    palette: dict[str, str],
) -> str:
    rgb = ImageColor.getrgb(panel_color)[:3]
    return palette["dark"] if _luminance(rgb) >= 0.55 else "#FFFFFF"


def _make_panel(
    item: dict,
    canvas: dict,
    preset: dict,
    palette: dict[str, str],
) -> dict:
    padding = max(
        5,
        round(min(int(item["width"]), int(item["height"])) * 0.10),
    )
    x = max(0, int(item["x"]) - padding)
    y = max(0, int(item["y"]) - padding)
    width = min(
        int(canvas["width"]) - x,
        int(item["width"]) + (int(item["x"]) - x) + padding,
    )
    height = min(
        int(canvas["height"]) - y,
        int(item["height"]) + (int(item["y"]) - y) + padding,
    )
    panel_color = _panel_color(preset["panel_mode"], palette)
    radius = round(min(width, height) * preset["radius_ratio"])
    return {
        "id": f"style-panel-{item['role']}",
        "target_ids": [item["id"]],
        "x": x,
        "y": y,
        "width": width,
        "height": height,
        "z_index": max(0, int(item.get("z_index", 2)) - 1),
        "background_color": panel_color,
        "gradient_color": (
            palette["dark"] if preset["gradient"] else panel_color
        ),
        "opacity": preset["opacity"],
        "border_radius": max(0, radius),
        "border_color": (
            palette["accent"]
            if preset["border_width"]
            else panel_color
        ),
        "border_width": preset["border_width"],
        "blur_radius": preset["blur"],
    }


def _normalize_geometry(layout: dict) -> None:
    canvas = layout["canvas"]
    canvas_width = int(canvas["width"])
    canvas_height = int(canvas["height"])
    margin = max(6, round(min(canvas_width, canvas_height) * 0.025))
    tolerance = max(8, round(min(canvas_width, canvas_height) * 0.04))
    for item in layout["elements"]:
        item["x"] = max(
            margin,
            min(int(item["x"]), canvas_width - margin - 1),
        )
        item["y"] = max(
            margin,
            min(int(item["y"]), canvas_height - margin - 1),
        )
        item["width"] = min(
            int(item["width"]),
            canvas_width - margin - int(item["x"]),
        )
        item["height"] = min(
            int(item["height"]),
            canvas_height - margin - int(item["y"]),
        )

    roles = {item["role"]: item for item in layout["elements"]}
    title = roles.get("title")
    subtitle = roles.get("subtitle")
    if title is None or subtitle is None:
        return
    alignment = str(title.get("text_align", "left"))
    if alignment != str(subtitle.get("text_align", "left")):
        return
    if alignment == "right":
        title_edge = int(title["x"]) + int(title["width"])
        subtitle_edge = int(subtitle["x"]) + int(subtitle["width"])
        if abs(title_edge - subtitle_edge) <= tolerance:
            subtitle["x"] += title_edge - subtitle_edge
    elif alignment == "center":
        title_center = int(title["x"]) + int(title["width"]) // 2
        subtitle_center = (
            int(subtitle["x"]) + int(subtitle["width"]) // 2
        )
        if abs(title_center - subtitle_center) <= tolerance:
            subtitle["x"] += title_center - subtitle_center
    elif abs(int(title["x"]) - int(subtitle["x"])) <= tolerance:
        subtitle["x"] = int(title["x"])
    subtitle["x"] = max(
        margin,
        min(
            int(subtitle["x"]),
            canvas_width - margin - int(subtitle["width"]),
        ),
    )


def apply_style(
    layout: dict,
    image_path: str | Path,
    style_name: str,
) -> tuple[dict, dict]:
    if style_name not in STYLE_PRESETS:
        raise ValueError(f"Unknown style preset: {style_name}")
    preset = STYLE_PRESETS[style_name]
    palette = extract_palette(image_path)
    adjusted = deepcopy(layout)
    _normalize_geometry(adjusted)
    adjusted["underlays"] = [
        item
        for item in adjusted.get("underlays", [])
        if str(item.get("id", "")).startswith("auto-contrast-")
    ]

    panels = []
    for item in adjusted["elements"]:
        role = str(item["role"])
        scale = preset["scales"].get(role, 1.0)
        item["font_size"] = max(
            8,
            round(int(item["font_size"]) * scale),
        )
        if role == "title":
            item["font_weight"] = max(750, int(item["font_weight"]))
        elif role in {"price", "cta"}:
            item["font_weight"] = max(650, int(item["font_weight"]))
        item["tracking"] = preset["tracking"] if role == "title" else 0
        item["shadow_offset"] = preset["shadow"]
        item["shadow_color"] = "#000000"
        item["stroke_width"] = preset["stroke"] if role == "title" else 0
        item["stroke_color"] = palette["dark"]
        item["number_scale"] = 1.22 if role == "price" else 1.0
        item["unit_scale"] = 0.82 if role == "price" else 1.0

        if role in preset["panels"]:
            panel = _make_panel(
                item,
                adjusted["canvas"],
                preset,
                palette,
            )
            panels.append(panel)
            item["color"] = _text_color_for_panel(
                panel["background_color"],
                palette,
            )
        elif role == "price":
            item["color"] = palette["accent"]

    adjusted["underlays"].extend(panels[:3])
    metadata = {
        "style": style_name,
        "palette": palette,
        "decorative_panel_count": len(panels[:3]),
    }
    return adjusted, metadata


def _overlap(first: dict, second: dict) -> bool:
    return (
        int(first["x"]) < int(second["x"]) + int(second["width"])
        and int(second["x"]) < int(first["x"]) + int(first["width"])
        and int(first["y"]) < int(second["y"]) + int(second["height"])
        and int(second["y"]) < int(first["y"]) + int(first["height"])
    )


def score_layout_design(layout: dict) -> dict[str, float]:
    """Return a deterministic composition score used as VLM context."""
    elements = layout["elements"]
    overlap_count = sum(
        1
        for index, first in enumerate(elements)
        for second in elements[index + 1:]
        if _overlap(first, second)
    )
    roles = {item["role"]: item for item in elements}
    hierarchy = 100.0
    if "title" in roles and "subtitle" in roles:
        title_size = int(roles["title"]["font_size"])
        subtitle_size = int(roles["subtitle"]["font_size"])
        if title_size <= subtitle_size:
            hierarchy = 45.0

    canvas = layout["canvas"]
    quadrants = set()
    for item in elements:
        center_x = int(item["x"]) + int(item["width"]) / 2
        center_y = int(item["y"]) + int(item["height"]) / 2
        quadrants.add(
            (
                int(center_x >= int(canvas["width"]) / 2),
                int(center_y >= int(canvas["height"]) / 2),
            )
        )
    distribution = min(100.0, 45.0 + len(quadrants) * 18.0)
    canvas_width = int(canvas["width"])
    canvas_height = int(canvas["height"])
    tolerance = max(8, round(min(canvas_width, canvas_height) * 0.04))
    aligned_pairs = 0
    pair_count = 0
    for index, first in enumerate(elements):
        for second in elements[index + 1:]:
            pair_count += 1
            first_edges = (
                int(first["x"]),
                int(first["x"]) + int(first["width"]),
            )
            second_edges = (
                int(second["x"]),
                int(second["x"]) + int(second["width"]),
            )
            if any(
                abs(first_edge - second_edge) <= tolerance
                for first_edge in first_edges
                for second_edge in second_edges
            ):
                aligned_pairs += 1
    alignment = (
        75.0
        if pair_count == 0
        else 55.0 + 45.0 * aligned_pairs / pair_count
    )
    safe_margin = max(
        6,
        round(min(canvas_width, canvas_height) * 0.025),
    )
    margin_violations = sum(
        1
        for item in elements
        if (
            int(item["x"]) < safe_margin
            or int(item["y"]) < safe_margin
            or int(item["x"]) + int(item["width"])
            > canvas_width - safe_margin
            or int(item["y"]) + int(item["height"])
            > canvas_height - safe_margin
        )
    )
    margins = max(40.0, 100.0 - margin_violations * 20.0)
    panel_count = len(
        [
            item
            for item in layout.get("underlays", [])
            if str(item.get("id", "")).startswith("style-panel-")
        ]
    )
    restraint = max(40.0, 100.0 - max(0, panel_count - 2) * 18.0)
    non_overlap = max(0.0, 100.0 - overlap_count * 35.0)
    total = (
        hierarchy * 0.22
        + distribution * 0.18
        + alignment * 0.15
        + margins * 0.15
        + restraint * 0.12
        + non_overlap * 0.18
    )
    return {
        "total": round(total, 2),
        "hierarchy": round(hierarchy, 2),
        "distribution": round(distribution, 2),
        "alignment": round(alignment, 2),
        "margins": round(margins, 2),
        "restraint": round(restraint, 2),
        "non_overlap": round(non_overlap, 2),
    }


__all__ = [
    "STYLE_NAMES",
    "STYLE_PRESETS",
    "apply_style",
    "extract_palette",
    "score_layout_design",
]
