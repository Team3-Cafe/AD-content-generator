from __future__ import annotations

from copy import deepcopy
from pathlib import Path


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(value, high))


def _overlap(first: dict, second: dict) -> bool:
    return (
        first["x"] < second["x"] + second["width"]
        and second["x"] < first["x"] + first["width"]
        and first["y"] < second["y"] + second["height"]
        and second["y"] < first["y"] + first["height"]
    )


def _normalized_box(box: dict, width: int, height: int) -> dict:
    return {
        "x": round(float(box["x"]) * width),
        "y": round(float(box["y"]) * height),
        "width": max(1, round(float(box["width"]) * width)),
        "height": max(1, round(float(box["height"]) * height)),
    }


def _place_group(
    desired: dict,
    *,
    canvas_width: int,
    canvas_height: int,
    margin: int,
    avoid: list[dict],
) -> dict:
    width = min(desired["width"], canvas_width - margin * 2)
    height = min(desired["height"], canvas_height - margin * 2)
    x = _clamp(desired["x"], margin, canvas_width - margin - width)
    y = _clamp(desired["y"], margin, canvas_height - margin - height)
    initial = {"x": x, "y": y, "width": width, "height": height}
    if not any(_overlap(initial, obstacle) for obstacle in avoid):
        return initial

    candidates = [
        {**initial, "x": margin},
        {**initial, "x": canvas_width - margin - width},
        {**initial, "y": margin},
        {**initial, "y": canvas_height - margin - height},
    ]
    candidates.extend(
        [
            {**initial, "x": obstacle["x"] - margin - width},
            {
                **initial,
                "x": obstacle["x"] + obstacle["width"] + margin,
            },
            {**initial, "y": obstacle["y"] - margin - height},
            {
                **initial,
                "y": obstacle["y"] + obstacle["height"] + margin,
            },
        ]
        for obstacle in avoid
    )
    flattened = []
    for candidate in candidates:
        if isinstance(candidate, list):
            flattened.extend(candidate)
        else:
            flattened.append(candidate)
    valid = []
    evaluated = []
    for candidate in flattened:
        candidate = {
            **candidate,
            "x": _clamp(
                candidate["x"],
                margin,
                canvas_width - margin - width,
            ),
            "y": _clamp(
                candidate["y"],
                margin,
                canvas_height - margin - height,
            ),
        }
        evaluated.append(candidate)
        if not any(_overlap(candidate, obstacle) for obstacle in avoid):
            distance = abs(candidate["x"] - x) + abs(candidate["y"] - y)
            valid.append((distance, candidate))
    if valid:
        return min(valid, key=lambda item: item[0])[1]
    def overlap_area(candidate: dict) -> int:
        total = 0
        for obstacle in avoid:
            overlap_width = max(
                0,
                min(
                    candidate["x"] + candidate["width"],
                    obstacle["x"] + obstacle["width"],
                ) - max(candidate["x"], obstacle["x"]),
            )
            overlap_height = max(
                0,
                min(
                    candidate["y"] + candidate["height"],
                    obstacle["y"] + obstacle["height"],
                ) - max(candidate["y"], obstacle["y"]),
            )
            total += overlap_width * overlap_height
        return total

    return min(
        evaluated,
        key=lambda item: (
            overlap_area(item),
            abs(item["x"] - x) + abs(item["y"] - y),
        ),
    )


def _place_centered_group(
    desired: dict,
    *,
    canvas_width: int,
    canvas_height: int,
    margin: int,
    avoid: list[dict],
) -> dict:
    """Keep content centered and resolve collisions only on the y-axis."""
    width = min(desired["width"], canvas_width - margin * 2)
    height = min(desired["height"], canvas_height - margin * 2)
    x = (canvas_width - width) // 2
    requested_y = _clamp(
        desired["y"],
        margin,
        canvas_height - margin - height,
    )
    y_positions = [requested_y, margin, canvas_height - margin - height]
    for obstacle in avoid:
        y_positions.extend(
            [
                obstacle["y"] - margin - height,
                obstacle["y"] + obstacle["height"] + margin,
            ]
        )
    positions = []
    for y in y_positions:
        position = {
            "x": x,
            "y": _clamp(y, margin, canvas_height - margin - height),
            "width": width,
            "height": height,
        }
        if position not in positions:
            positions.append(position)
    valid = [
        position
        for position in positions
        if not any(_overlap(position, obstacle) for obstacle in avoid)
    ]
    if valid:
        return min(
            valid,
            key=lambda position: abs(position["y"] - requested_y),
        )

    def overlap_area(position: dict) -> int:
        total = 0
        for obstacle in avoid:
            overlap_width = max(
                0,
                min(position["x"] + width, obstacle["x"] + obstacle["width"])
                - max(position["x"], obstacle["x"]),
            )
            overlap_height = max(
                0,
                min(position["y"] + height, obstacle["y"] + obstacle["height"])
                - max(position["y"], obstacle["y"]),
            )
            total += overlap_width * overlap_height
        return total

    return min(
        positions,
        key=lambda position: (
            overlap_area(position),
            abs(position["y"] - requested_y),
        ),
    )


def _mix_hex(first: str, second: str, ratio: float) -> str:
    ratio = max(0.0, min(1.0, ratio))
    first_rgb = tuple(int(first[index:index + 2], 16) for index in (1, 3, 5))
    second_rgb = tuple(int(second[index:index + 2], 16) for index in (1, 3, 5))
    mixed = tuple(
        round(start + (end - start) * ratio)
        for start, end in zip(first_rgb, second_rgb)
    )
    return "#{:02X}{:02X}{:02X}".format(*mixed)


def _hex_luminance(color: str) -> float:
    red, green, blue = (
        int(color[index:index + 2], 16)
        for index in (1, 3, 5)
    )
    return (0.2126 * red + 0.7152 * green + 0.0722 * blue) / 255.0


def _band_style(
    image_analysis: dict,
    *,
    center_y: int,
    surface: str,
) -> dict:
    """Derive band and text colors from the actual placement region."""
    height = max(1, int(image_analysis["canvas"]["height"]))
    ratio = center_y / height
    bands = image_analysis.get("horizontal_bands", [])
    band = min(
        bands,
        key=lambda item: abs(
            float(item["y"]) + float(item["height"]) / 2 - ratio
        ),
    ) if bands else {
        "luminance": image_analysis["overall_luminance"],
        "contrast": 0.0,
        "edge_density": 0.0,
    }
    palette = image_analysis["palette"]
    if surface == "accent_band":
        return {
            "background": palette["accent"],
            "gradient": _mix_hex(palette["accent"], palette["light"], 0.16),
            "text": (
                "#101820"
                if _hex_luminance(palette["accent"]) >= 0.52
                else "#FFFFFF"
            ),
            "opacity": 0.96,
            "local_region": band,
        }

    busy = float(band["contrast"]) + float(band["edge_density"]) > 0.42
    use_dark_band = float(band["luminance"]) >= 0.48 or busy
    if use_dark_band:
        background = _mix_hex(palette["dark"], "#000000", 0.18)
        gradient = _mix_hex(background, palette["accent"], 0.12)
        text = "#FFFFFF"
    else:
        background = _mix_hex(palette["light"], "#FFFFFF", 0.14)
        gradient = _mix_hex(background, palette["accent"], 0.14)
        text = "#101820"
    opacity = {
        "full_width_solid": 0.90,
        "full_width_gradient": 0.84,
        "full_width_scrim": 0.78,
    }.get(surface, 0.84)
    if busy:
        opacity = min(0.92, opacity + 0.08)
    return {
        "background": background,
        "gradient": gradient,
        "text": text,
        "opacity": opacity,
        "local_region": band,
    }


def _panel(
    panel_id: str,
    group: str,
    box: dict,
    *,
    background: str,
    opacity: float,
    radius: int,
    gradient: str | None = None,
    z_index: int = 0,
) -> dict:
    item = {
        "id": panel_id,
        "design_group": group,
        "target_ids": [],
        **box,
        "z_index": z_index,
        "background_color": background,
        "opacity": opacity,
        "border_radius": radius,
    }
    if gradient is not None:
        item["gradient_color"] = gradient
    return item


def _element(
    role: str,
    content: str,
    group: str,
    box: dict,
    *,
    font_size: int,
    font_weight: int,
    color: str,
    align: str,
    max_lines: int,
    line_height: float,
    z_index: int = 2,
) -> dict:
    return {
        "id": f"copy-{role}",
        "role": role,
        "design_group": group,
        "content": content,
        **box,
        "font_size": font_size,
        "font_weight": font_weight,
        "color": color,
        "text_align": align,
        "vertical_align": "center",
        "max_lines": max_lines,
        "line_height": line_height,
        "tracking": 0,
        "shadow_offset": 1,
        "shadow_color": "#000000",
        "stroke_width": 0,
        "stroke_color": "#000000",
        "z_index": z_index,
    }


def build_design_layout(
    image_analysis: dict,
    ad_copy: dict[str, str],
    design_spec: dict,
) -> dict:
    """Compile one VLM direction into two adaptive full-width bands."""
    canvas = image_analysis["canvas"]
    width = int(canvas["width"])
    height = int(canvas["height"])
    short_side = min(width, height)
    margin = max(12, round(short_side * 0.045))
    palette = image_analysis["palette"]
    direction = design_spec["art_direction"]
    composition = design_spec["composition"]
    offer_alignment = direction["alignment"]
    density = direction["spacing_density"]
    gap_ratio = {
        "compact": 0.012,
        "balanced": 0.020,
        "airy": 0.030,
    }[density]
    inner_gap = max(6, round(short_side * gap_ratio))
    band_padding = max(12, round(short_side * 0.028))

    title_size = max(
        22,
        round(short_side * 0.072 * float(composition["title_scale"])),
    )
    subtitle_size = max(14, round(title_size * 0.47))
    price_size = max(20, round(title_size * 0.84))
    cta_size = max(14, round(title_size * 0.44))
    headline_width = round(
        width * float(composition["headline_content_width_ratio"])
    )
    offer_width = round(
        width * float(composition["offer_content_width_ratio"])
    )
    title_height = round(title_size * 1.35)
    subtitle_height = (
        round(subtitle_size * 2.8)
        if ad_copy.get("subtitle")
        else 0
    )
    headline_height = title_height + (
        inner_gap + subtitle_height if subtitle_height else 0
    )
    protected = _normalized_box(
        design_spec["scene_analysis"]["subject_region"],
        width,
        height,
    )
    headline_group = _place_centered_group(
        {
            "x": 0,
            "y": round(float(composition["headline_y_ratio"]) * height),
            "width": headline_width,
            "height": headline_height,
        },
        canvas_width=width,
        canvas_height=height,
        margin=margin,
        avoid=[protected],
    )

    has_price = bool(ad_copy.get("price"))
    has_cta = bool(ad_copy.get("cta"))
    arrangement = composition["offer_arrangement"]
    if arrangement == "horizontal" and has_price and has_cta:
        offer_height = max(
            round(price_size * 1.55),
            round(cta_size * 2.5),
        )
    else:
        offer_height = (
            (round(price_size * 1.45) if has_price else 0)
            + (inner_gap if has_price and has_cta else 0)
            + (round(cta_size * 2.35) if has_cta else 0)
        )
    offer_group = _place_centered_group(
        {
            "x": 0,
            "y": round(float(composition["offer_y_ratio"]) * height),
            "width": offer_width,
            "height": max(1, offer_height),
        },
        canvas_width=width,
        canvas_height=height,
        margin=margin,
        avoid=[protected, headline_group],
    )

    headline_band = _band_style(
        image_analysis,
        center_y=headline_group["y"] + headline_group["height"] // 2,
        surface=direction["headline_surface"],
    )
    offer_band = _band_style(
        image_analysis,
        center_y=offer_group["y"] + offer_group["height"] // 2,
        surface=direction["offer_surface"],
    )

    elements = []
    elements.append(
        _element(
            "title",
            ad_copy["title"],
            "headline",
            {
                "x": headline_group["x"],
                "y": headline_group["y"],
                "width": headline_group["width"],
                "height": title_height,
            },
            font_size=title_size,
            font_weight=820,
            color=headline_band["text"],
            align="center",
            max_lines=1,
            line_height=1.1,
        )
    )
    if ad_copy.get("subtitle"):
        elements.append(
            _element(
                "subtitle",
                ad_copy["subtitle"],
                "headline",
                {
                    "x": headline_group["x"],
                    "y": headline_group["y"] + title_height + inner_gap,
                    "width": headline_group["width"],
                    "height": subtitle_height,
                },
                font_size=subtitle_size,
                font_weight=480,
                color=headline_band["text"],
                align="center",
                max_lines=2,
                line_height=1.25,
            )
        )

    offer_boxes = {}
    if arrangement == "horizontal" and has_price and has_cta:
        price_width = round(offer_group["width"] * 0.43)
        offer_boxes["price"] = {
            "x": offer_group["x"],
            "y": offer_group["y"],
            "width": price_width,
            "height": offer_group["height"],
        }
        offer_boxes["cta"] = {
            "x": offer_group["x"] + price_width + inner_gap,
            "y": offer_group["y"],
            "width": max(
                1,
                offer_group["width"] - price_width - inner_gap,
            ),
            "height": offer_group["height"],
        }
    else:
        cursor_y = offer_group["y"]
        if has_price:
            price_height = round(price_size * 1.45)
            offer_boxes["price"] = {
                "x": offer_group["x"],
                "y": cursor_y,
                "width": offer_group["width"],
                "height": price_height,
            }
            cursor_y += price_height + (inner_gap if has_cta else 0)
        if has_cta:
            offer_boxes["cta"] = {
                "x": offer_group["x"],
                "y": cursor_y,
                "width": offer_group["width"],
                "height": round(cta_size * 2.35),
            }

    accent_role = direction["accent_role"]
    if has_price:
        price_color = offer_band["text"]
        if (
            accent_role in {"price", "price_and_cta"}
            and direction["offer_surface"] != "accent_band"
        ):
            price_color = palette["accent"]
        price_item = _element(
            "price",
            ad_copy["price"],
            "offer",
            offer_boxes["price"],
            font_size=price_size,
            font_weight=800,
            color=price_color,
            align=offer_alignment,
            max_lines=1,
            line_height=1.0,
        )
        price_item["number_scale"] = 1.22
        price_item["unit_scale"] = 0.80
        elements.append(price_item)
    if has_cta:
        elements.append(
            _element(
                "cta",
                ad_copy["cta"],
                "offer",
                offer_boxes["cta"],
                font_size=cta_size,
                font_weight=720,
                color=offer_band["text"],
                align=(
                    "center"
                    if accent_role in {"cta", "price_and_cta"}
                    else offer_alignment
                ),
                max_lines=1,
                line_height=1.0,
            )
        )

    headline_band_y = max(0, headline_group["y"] - band_padding)
    headline_band_height = min(
        height - headline_band_y,
        headline_group["height"] + band_padding * 2,
    )
    offer_band_y = max(0, offer_group["y"] - band_padding)
    offer_band_height = min(
        height - offer_band_y,
        offer_group["height"] + band_padding * 2,
    )
    underlays = [
        _panel(
            "surface-headline",
            "headline",
            {
                "x": 0,
                "y": headline_band_y,
                "width": width,
                "height": headline_band_height,
            },
            background=headline_band["background"],
            gradient=headline_band["gradient"],
            opacity=headline_band["opacity"],
            radius=0,
        )
    ]
    if has_price or has_cta:
        underlays.append(
            _panel(
                "surface-offer",
                "offer",
                {
                    "x": 0,
                    "y": offer_band_y,
                    "width": width,
                    "height": offer_band_height,
                },
                background=offer_band["background"],
                gradient=offer_band["gradient"],
                opacity=offer_band["opacity"],
                radius=0,
            )
        )

    rule_width = max(3, round(short_side * 0.008))
    rule_length = max(
        rule_width * 8,
        round(headline_group["width"] * 0.16),
    )
    underlays.append(
        _panel(
            "accent-rule",
            "headline",
            {
                "x": (width - rule_length) // 2,
                "y": min(
                    headline_band_y + headline_band_height - rule_width * 2,
                    headline_group["y"] + title_height + max(2, inner_gap // 3),
                ),
                "width": rule_length,
                "height": rule_width,
            },
            background=palette["accent"],
            opacity=1.0,
            radius=max(1, rule_width // 2),
            z_index=1,
        )
    )

    return {
        "canvas": {"width": width, "height": height},
        "elements": elements,
        "underlays": underlays,
        "warnings": [],
        "design_groups": {
            "headline": headline_group,
            "offer": offer_group,
            "protected_subject": protected,
        },
        "design_tokens": {
            "palette": palette,
            "headline_alignment": "center",
            "offer_alignment": offer_alignment,
            "mood": direction["mood"],
            "spacing_density": density,
            "headline_band": headline_band,
            "offer_band": offer_band,
        },
    }


def apply_design_revision(layout: dict, revision: dict) -> dict:
    """Apply one bounded VLM critique to the same design, never a candidate."""
    adjusted = deepcopy(layout)
    if not revision.get("needs_revision"):
        return adjusted
    changes = revision["adjustments"]
    width = int(adjusted["canvas"]["width"])
    height = int(adjusted["canvas"]["height"])
    margin = max(8, round(min(width, height) * 0.025))

    for group, prefix in (("headline", "headline"), ("offer", "offer")):
        dx = (
            0
            if group == "headline"
            else round(float(changes["offer_x_shift"]) * width)
        )
        dy = round(float(changes[f"{prefix}_y_shift"]) * height)
        scale = float(changes[f"{prefix}_scale"])
        element_items = [
            item
            for item in adjusted["elements"]
            if item.get("design_group") == group
        ]
        surface_items = [
            item
            for item in adjusted.get("underlays", [])
            if item.get("design_group") == group
        ]
        group_items = element_items + surface_items
        if not element_items:
            continue
        left = min(int(item["x"]) for item in element_items)
        right = max(
            int(item["x"]) + int(item["width"])
            for item in element_items
        )
        top = min(int(item["y"]) for item in group_items)
        bottom = max(
            int(item["y"]) + int(item["height"])
            for item in group_items
        )
        dx = _clamp(dx, margin - left, width - margin - right)
        dy = _clamp(dy, -top, height - bottom)
        for item in element_items:
            item["x"] = int(item["x"]) + dx
            item["y"] = int(item["y"]) + dy
            item["font_size"] = max(
                10,
                round(int(item["font_size"]) * scale),
            )
        for item in surface_items:
            if int(item["width"]) < width:
                item["x"] = int(item["x"]) + dx
            item["y"] = int(item["y"]) + dy
        if group in adjusted.get("design_groups", {}):
            adjusted["design_groups"][group]["x"] += dx
            adjusted["design_groups"][group]["y"] += dy

    opacity_delta = float(changes["surface_opacity_delta"])
    for underlay in adjusted.get("underlays", []):
        if str(underlay.get("id", "")).startswith("surface-"):
            underlay["opacity"] = round(
                max(0.25, min(0.95, float(underlay["opacity"]) + opacity_delta)),
                3,
            )
    return adjusted


__all__ = ["apply_design_revision", "build_design_layout"]
