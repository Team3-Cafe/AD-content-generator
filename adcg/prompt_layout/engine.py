from __future__ import annotations

from copy import deepcopy
from pathlib import Path


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(value, high))


def _scale_element_box(
    item: dict,
    scale: float,
    *,
    canvas_width: int,
    canvas_height: int,
    margin: int = 0,
) -> None:
    """Scale a text box around its center so font growth survives fitting."""
    if abs(scale - 1.0) < 1e-9:
        return
    old_width = int(item["width"])
    old_height = int(item["height"])
    center_x = int(item["x"]) + old_width / 2
    center_y = int(item["y"]) + old_height / 2
    max_width = max(1, canvas_width - margin * 2)
    max_height = max(1, canvas_height - margin * 2)
    new_width = max(1, min(max_width, round(old_width * scale)))
    new_height = max(1, min(max_height, round(old_height * scale)))
    item["width"] = new_width
    item["height"] = new_height
    item["x"] = _clamp(
        round(center_x - new_width / 2),
        margin,
        canvas_width - margin - new_width,
    )
    item["y"] = _clamp(
        round(center_y - new_height / 2),
        margin,
        canvas_height - margin - new_height,
    )


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


def _resolve_color_token(token: str, palette: dict[str, str]) -> str:
    return {
        "palette_dark": palette["dark"],
        "palette_light": palette["light"],
        "palette_accent": palette["accent"],
        "neutral_dark": "#101820",
        "neutral_light": "#FFFFFF",
    }.get(token, palette["dark"])


def _contrast_ratio(first: str, second: str) -> float:
    def relative_luminance(color: str) -> float:
        channels = [
            int(color[index:index + 2], 16) / 255.0
            for index in (1, 3, 5)
        ]
        linear = [
            value / 12.92
            if value <= 0.04045
            else ((value + 0.055) / 1.055) ** 2.4
            for value in channels
        ]
        return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]

    first_luminance = relative_luminance(first)
    second_luminance = relative_luminance(second)
    lighter = max(first_luminance, second_luminance)
    darker = min(first_luminance, second_luminance)
    return (lighter + 0.05) / (darker + 0.05)


def _safe_text_color(
    background: str,
    requested: str,
    palette: dict[str, str],
) -> str:
    candidates = [
        requested,
        palette["light"],
        palette["dark"],
        "#FFFFFF",
        "#101820",
    ]
    if _contrast_ratio(background, requested) >= 4.5:
        return requested
    return max(
        candidates,
        key=lambda color: _contrast_ratio(background, color),
    )


def _band_style(
    image_analysis: dict,
    *,
    center_y: int,
    surface: str,
    background_token: str,
    text_token: str,
) -> dict:
    """Resolve VLM palette choices and enforce readable contrast."""
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
    busy = float(band["contrast"]) + float(band["edge_density"]) > 0.42
    background = _resolve_color_token(background_token, palette)
    requested_text = _resolve_color_token(text_token, palette)
    text = _safe_text_color(background, requested_text, palette)
    if surface == "full_width_gradient":
        gradient_target = (
            palette["dark"]
            if _hex_luminance(background) >= 0.55
            else palette["accent"]
        )
        gradient = _mix_hex(background, gradient_target, 0.18)
    else:
        gradient = background
    opacity = {
        "full_width_solid": 0.90,
        "full_width_gradient": 0.84,
        "full_width_scrim": 0.78,
        "accent_band": 0.94,
    }.get(surface, 0.84)
    if busy:
        opacity = min(0.92, opacity + 0.08)
    return {
        "background": background,
        "gradient": gradient,
        "text": text,
        "requested_text": requested_text,
        "background_token": background_token,
        "text_token": text_token,
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
    border_color: str | None = None,
    border_width: int = 0,
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
    if border_color is not None and border_width > 0:
        item["border_color"] = border_color
        item["border_width"] = border_width
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
    color_direction = design_spec["color_direction"]
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
    requested_arrangement = composition["offer_arrangement"]
    estimated_horizontal_width = round(
        len(ad_copy.get("price", "")) * price_size * 0.72
        + len(ad_copy.get("cta", "")) * cta_size * 0.66
        + inner_gap * 3
    )
    horizontal_has_room = (
        width / max(1, height) >= 1.15
        and estimated_horizontal_width <= offer_width
    )
    arrangement = (
        "horizontal"
        if requested_arrangement == "horizontal" and horizontal_has_room
        else "vertical"
    )
    if arrangement == "horizontal" and has_price and has_cta:
        offer_height = max(
            round(price_size * 1.55),
            round(cta_size * 1.6),
        )
    else:
        offer_height = (
            (round(price_size * 1.45) if has_price else 0)
            + (inner_gap if has_price and has_cta else 0)
            + (round(cta_size * 1.6) if has_cta else 0)
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
        background_token=color_direction["headline_background"],
        text_token=color_direction["headline_text"],
    )
    offer_band = _band_style(
        image_analysis,
        center_y=offer_group["y"] + offer_group["height"] // 2,
        surface=direction["offer_surface"],
        background_token=color_direction["offer_background"],
        text_token=color_direction["offer_text"],
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
            cta_width = min(
                offer_group["width"],
                max(
                    round(offer_group["width"] * 0.25),
                    round(len(ad_copy["cta"]) * cta_size * 0.72),
                ),
            )
            offer_boxes["cta"] = {
                "x": offer_group["x"]
                + (offer_group["width"] - cta_width) // 2,
                "y": cursor_y,
                "width": cta_width,
                "height": round(cta_size * 1.6),
            }

    accent_role = direction["accent_role"]
    cta_treatment = "plain"
    requested_cta_background = _resolve_color_token(
        color_direction["cta_background"],
        palette,
    )
    requested_cta_text = _resolve_color_token(
        color_direction["cta_text"],
        palette,
    )
    if cta_treatment == "accent_pill":
        cta_text_color = _safe_text_color(
            requested_cta_background,
            requested_cta_text,
            palette,
        )
    else:
        cta_text_color = _safe_text_color(
            offer_band["background"],
            requested_cta_text,
            palette,
        )
    if has_price:
        price_color = offer_band["text"]
        if (
            accent_role in {"price", "price_and_cta"}
            and direction["offer_surface"] != "accent_band"
        ):
            price_color = palette["accent"]
        price_color = _safe_text_color(
            offer_band["background"],
            price_color,
            palette,
        )
        price_item = _element(
            "price",
            ad_copy["price"],
            "offer",
            offer_boxes["price"],
            font_size=price_size,
            font_weight=800,
            color=price_color,
            align=("center" if arrangement == "vertical" else offer_alignment),
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
                color=(
                    offer_band["text"]
                    if cta_treatment == "plain"
                    else cta_text_color
                ),
                align=(
                    "center"
                    if arrangement == "vertical"
                    or accent_role in {"cta", "price_and_cta"}
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
    if has_cta and cta_treatment in {"accent_pill", "outline"}:
        cta_box = offer_boxes["cta"]
        underlays.append(
            _panel(
                "surface-cta",
                "offer",
                cta_box,
                background=requested_cta_background,
                gradient=requested_cta_background,
                opacity=(0.96 if cta_treatment == "accent_pill" else 0.10),
                radius=max(8, round(cta_box["height"] * 0.48)),
                border_color=(
                    requested_cta_background
                    if cta_treatment == "outline"
                    else None
                ),
                border_width=(
                    max(1, round(short_side * 0.004))
                    if cta_treatment == "outline"
                    else 0
                ),
                z_index=1,
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
            "offer_arrangement": arrangement,
            "requested_offer_arrangement": requested_arrangement,
            "cta_treatment": cta_treatment,
            "color_direction": color_direction,
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
            _scale_element_box(
                item,
                scale,
                canvas_width=width,
                canvas_height=height,
                margin=margin,
            )
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


def apply_final_review_revision(layout: dict, revision: dict) -> dict:
    """Apply the completed-ad VLM's geometry, typography, and color fixes."""
    adjusted = apply_design_revision(layout, revision)
    if not revision.get("needs_revision"):
        return adjusted

    changes = revision["adjustments"]
    width = int(adjusted["canvas"]["width"])
    height = int(adjusted["canvas"]["height"])
    role_scales = {
        role: float(changes[f"{role}_scale"])
        for role in ("title", "subtitle", "price", "cta")
    }
    weight_steps = {
        "keep": 0,
        "lighter": -100,
        "bolder": 100,
    }
    tracking_deltas = {
        "headline": int(changes["headline_tracking_delta"]),
        "offer": int(changes["offer_tracking_delta"]),
    }
    for item in adjusted["elements"]:
        role = str(item.get("role", ""))
        group = str(item.get("design_group", ""))
        if role in role_scales:
            role_scale = role_scales[role]
            _scale_element_box(
                item,
                role_scale,
                canvas_width=width,
                canvas_height=height,
            )
            item["font_size"] = max(
                8,
                round(int(item["font_size"]) * role_scale),
            )
        weight_choice = str(changes.get(f"{group}_weight", "keep"))
        item["font_weight"] = max(
            300,
            min(
                900,
                int(item.get("font_weight", 600))
                + weight_steps.get(weight_choice, 0),
            ),
        )
        item["tracking"] = max(
            0,
            int(item.get("tracking", 0)) + tracking_deltas.get(group, 0),
        )

    elements_by_role = {
        str(item.get("role")): item
        for item in adjusted["elements"]
    }
    price = elements_by_role.get("price")
    if price is not None:
        segment_box_scale = max(
            1.0,
            float(changes["price_number_scale"]),
            float(changes["price_unit_scale"]),
        )
        _scale_element_box(
            price,
            segment_box_scale,
            canvas_width=width,
            canvas_height=height,
        )
        price["number_scale"] = round(
            float(price.get("number_scale", 1.22))
            * float(changes["price_number_scale"]),
            4,
        )
        price["unit_scale"] = round(
            float(price.get("unit_scale", 0.80))
            * float(changes["price_unit_scale"]),
            4,
        )
        for prefix in ("number", "unit"):
            key = f"{prefix}_baseline_shift"
            shift = round(
                float(price.get(key, 0.0))
                + float(changes[f"price_{key}"]),
                4,
            )
            if key in price or abs(shift) >= 1e-9:
                price[key] = shift

    def shift_element(item: dict | None, dx: int = 0, dy: int = 0) -> None:
        if item is None:
            return
        item["x"] = _clamp(
            int(item["x"]) + dx,
            0,
            max(0, width - int(item["width"])),
        )
        item["y"] = _clamp(
            int(item["y"]) + dy,
            0,
            max(0, height - int(item["height"])),
        )

    subtitle_gap = round(
        float(changes["headline_subtitle_gap_delta"]) * height
    )
    shift_element(elements_by_role.get("subtitle"), dy=subtitle_gap)

    offer_gap = float(changes["price_cta_gap_delta"])
    if adjusted["design_tokens"].get("offer_arrangement") == "horizontal":
        shift_element(elements_by_role.get("cta"), dx=round(offer_gap * width))
    else:
        shift_element(elements_by_role.get("cta"), dy=round(offer_gap * height))

    for group in ("headline", "offer"):
        group_items = [
            item
            for item in adjusted["elements"]
            if item.get("design_group") == group
        ]
        if not group_items:
            continue
        left = min(int(item["x"]) for item in group_items)
        top = min(int(item["y"]) for item in group_items)
        right = max(
            int(item["x"]) + int(item["width"])
            for item in group_items
        )
        bottom = max(
            int(item["y"]) + int(item["height"])
            for item in group_items
        )
        adjusted["design_groups"][group].update(
            {"x": left, "y": top, "width": right - left, "height": bottom - top}
        )
    offer_alignment = str(changes["offer_alignment"])
    if offer_alignment != "keep":
        for item in adjusted["elements"]:
            if item.get("design_group") == "offer":
                item["text_align"] = offer_alignment
        adjusted["design_tokens"]["offer_alignment"] = offer_alignment

    tokens = adjusted["design_tokens"]
    tokens["cta_treatment"] = "plain"
    palette = tokens["palette"]
    color_direction = tokens["color_direction"]

    def selected_color(name: str) -> str | None:
        token = str(changes[name])
        if token == "keep":
            return None
        color_direction[name] = token
        return _resolve_color_token(token, palette)

    adjusted["underlays"] = [
        item
        for item in adjusted.get("underlays", [])
        if str(item.get("id")) != "surface-cta"
    ]
    underlays = {
        str(item.get("id")): item
        for item in adjusted.get("underlays", [])
    }
    headline_surface = underlays.get("surface-headline")
    offer_surface = underlays.get("surface-offer")

    def scale_band(surface: dict | None, group: str, scale: float) -> None:
        if surface is None:
            return
        current_height = int(surface["height"])
        group_box = adjusted["design_groups"][group]
        padding = max(4, round(min(width, height) * 0.01))
        minimum_height = int(group_box["height"]) + padding * 2
        new_height = max(
            minimum_height,
            min(height, round(current_height * scale)),
        )
        center_y = int(group_box["y"]) + int(group_box["height"]) / 2
        surface["height"] = new_height
        surface["y"] = _clamp(
            round(center_y - new_height / 2),
            0,
            height - new_height,
        )

    scale_band(
        headline_surface,
        "headline",
        float(changes["headline_band_height_scale"]),
    )
    scale_band(
        offer_surface,
        "offer",
        float(changes["offer_band_height_scale"]),
    )

    accent_rule = underlays.get("accent-rule")
    if accent_rule is not None:
        rule_center = int(accent_rule["x"]) + int(accent_rule["width"]) / 2
        accent_rule["width"] = max(
            3,
            min(
                width,
                round(
                    int(accent_rule["width"])
                    * float(changes["accent_rule_width_scale"])
                ),
            ),
        )
        accent_rule["x"] = _clamp(
            round(rule_center - int(accent_rule["width"]) / 2),
            0,
            width - int(accent_rule["width"]),
        )
        shift_element(
            accent_rule,
            dy=round(float(changes["accent_rule_y_shift"]) * height),
        )

    for name, surface in (
        ("headline_background", headline_surface),
        ("offer_background", offer_surface),
    ):
        color = selected_color(name)
        if color is not None and surface is not None:
            surface["background_color"] = color
            surface["gradient_color"] = color
            if surface.get("border_color") is not None:
                surface["border_color"] = color

    headline_background = (
        str(headline_surface.get("background_color", palette["dark"]))
        if headline_surface is not None
        else palette["dark"]
    )
    offer_background = (
        str(offer_surface.get("background_color", palette["dark"]))
        if offer_surface is not None
        else palette["dark"]
    )
    cta_background = offer_background
    requested_text = {
        "headline": selected_color("headline_text"),
        "offer": selected_color("offer_text"),
        "cta": selected_color("cta_text"),
    }
    for item in adjusted["elements"]:
        role = str(item.get("role", ""))
        group = str(item.get("design_group", ""))
        key = "cta" if role == "cta" else group
        requested = requested_text.get(key) or str(item.get("color", "#FFFFFF"))
        background = (
            cta_background
            if key == "cta"
            else headline_background
            if group == "headline"
            else offer_background
        )
        item["color"] = _safe_text_color(background, requested, palette)

    if headline_surface is not None:
        tokens["headline_band"]["background"] = headline_surface.get(
            "background_color", palette["dark"]
        )
        tokens["headline_band"]["gradient"] = headline_surface.get(
            "gradient_color", tokens["headline_band"]["background"]
        )
    if offer_surface is not None:
        tokens["offer_band"]["background"] = offer_surface.get(
            "background_color", palette["dark"]
        )
        tokens["offer_band"]["gradient"] = offer_surface.get(
            "gradient_color", tokens["offer_band"]["background"]
        )
    return adjusted


__all__ = [
    "apply_design_revision",
    "apply_final_review_revision",
    "build_design_layout",
]
