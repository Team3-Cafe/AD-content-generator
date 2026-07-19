from __future__ import annotations

from copy import deepcopy


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
    """Respect the requested horizontal position and resolve y collisions."""
    width = min(desired["width"], canvas_width - margin * 2)
    height = min(desired["height"], canvas_height - margin * 2)
    x = _clamp(
        int(desired.get("x", margin)),
        margin,
        canvas_width - margin - width,
    )
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


def _resolve_color_token(token: str, palette: dict[str, str]) -> str:
    value = str(token).strip()
    if len(value) == 7 and value.startswith("#"):
        try:
            int(value[1:], 16)
        except ValueError:
            pass
        else:
            return value.upper()
    return {
        "palette_dark": palette["dark"],
        "palette_light": palette["light"],
        "palette_accent": palette["accent"],
    }.get(value, palette["dark"])


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
    del palette  # Palette does not override the art director's requested hue.
    if _contrast_ratio(background, requested) >= 4.5:
        return requested
    source = tuple(
        int(requested[index:index + 2], 16) for index in (1, 3, 5)
    )
    variants = []
    for destination in ((255, 255, 255), (0, 0, 0)):
        for ratio in (0.18, 0.32, 0.48, 0.64, 0.80, 0.90):
            rgb = tuple(
                round(channel + (target - channel) * ratio)
                for channel, target in zip(source, destination)
            )
            variants.append("#{:02X}{:02X}{:02X}".format(*rgb))
    passing = [
        color for color in variants
        if _contrast_ratio(background, color) >= 4.5
    ]
    return (
        min(
            passing,
            key=lambda color: sum(
                abs(int(color[index:index + 2], 16) - source[offset])
                for offset, index in enumerate((1, 3, 5))
            ),
        )
        if passing
        else max(variants, key=lambda color: _contrast_ratio(background, color))
    )


def _resolve_surface_effect(
    effect: dict,
    palette: dict[str, str],
    current: dict | None = None,
) -> dict:
    """Resolve VLM surface primitives into renderer-ready values."""
    current = current or {}
    current_colors = list(current.get("fill_colors", [])) or [
        str(current.get("background_color", palette["dark"])),
        str(current.get("gradient_color", palette["dark"])),
    ]

    def resolve_color(value: str, index: int = 0) -> str:
        if str(value).strip() == "keep":
            return current_colors[min(index, len(current_colors) - 1)]
        return _resolve_color_token(str(value), palette)

    colors = [
        resolve_color(value, index)
        for index, value in enumerate(effect["fill_colors"])
    ]
    stops = [float(value) for value in effect["fill_stops"]]
    if len(stops) != len(colors):
        denominator = max(1, len(colors) - 1)
        stops = [index / denominator for index in range(len(colors))]
    pairs = sorted(zip(stops, colors), key=lambda pair: pair[0])
    stops = [max(0.0, min(1.0, pair[0])) for pair in pairs]
    colors = [pair[1] for pair in pairs]

    def optional_color(field: str, fallback: str) -> str:
        value = str(effect[field])
        if value == "keep":
            return str(current.get(field, fallback))
        return _resolve_color_token(value, palette)

    return {
        "fill_type": str(effect["fill_type"]),
        "fill_colors": colors,
        "fill_stops": stops,
        "gradient_angle": float(effect["gradient_angle"]),
        "opacity": float(effect["opacity"]),
        "border_radius": int(effect["corner_radius"]),
        "backdrop_blur": int(effect["backdrop_blur"]),
        "blend_mode": str(effect["blend_mode"]),
        "border_enabled": bool(effect["border_enabled"]),
        "border_color": optional_color("border_color", colors[0]),
        "border_width": int(effect["border_width"]),
        "border_opacity": float(effect["border_opacity"]),
        "shadow_enabled": bool(effect["shadow_enabled"]),
        "shadow_color": optional_color("shadow_color", palette["dark"]),
        "shadow_offset_x": int(effect["shadow_offset_x"]),
        "shadow_offset_y": int(effect["shadow_offset_y"]),
        "shadow_blur": int(effect["shadow_blur"]),
        "shadow_opacity": float(effect["shadow_opacity"]),
    }


def _band_style(
    image_analysis: dict,
    *,
    surface: str,
    effect: dict,
    text_token: str,
) -> dict:
    """Resolve one VLM-authored surface effect and readable text color."""
    palette = image_analysis["palette"]
    resolved_effect = _resolve_surface_effect(effect, palette)
    requested_text = _resolve_color_token(text_token, palette)
    text = (
        requested_text
        if surface == "none"
        else _safe_text_color(
            resolved_effect["fill_colors"][0], requested_text, palette
        )
    )
    return {
        "text": text,
        "requested_text": requested_text,
        "text_token": text_token,
        "effect": resolved_effect,
    }


def _panel(
    panel_id: str,
    group: str,
    box: dict,
    *,
    effect: dict,
    z_index: int = 0,
) -> dict:
    return {
        "id": panel_id,
        "design_group": group,
        **box,
        "z_index": z_index,
        **effect,
    }


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
        "shadow_offset": 0,
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
    """Compile one VLM direction into freely placed copy groups and surfaces."""
    canvas = image_analysis["canvas"]
    width = int(canvas["width"])
    height = int(canvas["height"])
    short_side = min(width, height)
    margin = max(12, round(short_side * 0.045))
    palette = image_analysis["palette"]
    direction = design_spec["art_direction"]
    color_direction = design_spec["color_direction"]
    composition = design_spec["composition"]
    headline_alignment = direction["headline_alignment"]
    offer_alignment = direction["offer_alignment"]
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
    headline_group = _place_group(
        {
            "x": round(float(composition["headline_x_ratio"]) * width),
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
    offer_group = _place_group(
        {
            "x": round(float(composition["offer_x_ratio"]) * width),
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
        surface=direction["headline_surface"],
        effect=direction["headline_effect"],
        text_token=color_direction["headline_text"],
    )
    offer_band = _band_style(
        image_analysis,
        surface=direction["offer_surface"],
        effect=direction["offer_effect"],
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
            align=headline_alignment,
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
                align=headline_alignment,
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
    requested_cta_text = _resolve_color_token(
        color_direction["cta_text"],
        palette,
    )
    cta_text_color = _safe_text_color(
        offer_band["effect"]["fill_colors"][0],
        requested_cta_text,
        palette,
    )
    if has_price:
        price_color = offer_band["text"]
        if accent_role == "price":
            price_color = palette["accent"]
        price_color = _safe_text_color(
            offer_band["effect"]["fill_colors"][0],
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
                color=cta_text_color,
                align=offer_alignment,
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
    underlays = []

    def add_group_surface(
        group: str,
        surface_style: str,
        group_box: dict,
        band_y: int,
        band_height: int,
        band: dict,
    ) -> None:
        if surface_style == "none":
            return
        full_width = group == "headline" or surface_style == "full_width"
        content_x = max(0, group_box["x"] - band_padding)
        box = (
            {"x": 0, "y": band_y, "width": width, "height": band_height}
            if full_width
            else {
                "x": content_x,
                "y": band_y,
                "width": min(
                    width - content_x,
                    group_box["width"] + band_padding * 2,
                ),
                "height": band_height,
            }
        )
        underlays.append(
            _panel(
                f"surface-{group}",
                group,
                box,
                effect=band["effect"],
            )
        )

    add_group_surface(
        "headline", direction["headline_surface"], headline_group,
        headline_band_y, headline_band_height, headline_band,
    )
    if has_price or has_cta:
        add_group_surface(
            "offer", direction["offer_surface"], offer_group,
            offer_band_y, offer_band_height, offer_band,
        )


    if accent_role == "rule":
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
                effect={
                    "fill_type": "solid",
                    "fill_colors": [palette["accent"], palette["accent"]],
                    "fill_stops": [0.0, 1.0],
                    "gradient_angle": 0.0,
                    "opacity": 1.0,
                    "border_radius": max(1, rule_width // 2),
                    "backdrop_blur": 0, "blend_mode": "normal",
                    "border_enabled": False, "border_color": palette["accent"],
                    "border_width": 0, "border_opacity": 0.0,
                    "shadow_enabled": False, "shadow_color": palette["dark"],
                    "shadow_offset_x": 0, "shadow_offset_y": 0,
                    "shadow_blur": 0, "shadow_opacity": 0.0,
                },
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
            "headline_alignment": headline_alignment,
            "offer_alignment": offer_alignment,
            "offer_arrangement": arrangement,
            "requested_offer_arrangement": requested_arrangement,
            "color_direction": color_direction,
            "mood": direction["mood"],
            "spacing_density": density,
            "headline_band": headline_band,
            "offer_band": offer_band,
        },
    }


def apply_final_review_revision(layout: dict, revision: dict) -> dict:
    """Build a fresh final copy overlay from the VLM's absolute target."""
    if not revision.get("needs_revision"):
        return deepcopy(layout)

    target = revision["target_layout"]
    width = int(layout["canvas"]["width"])
    height = int(layout["canvas"]["height"])
    palette = layout["design_tokens"]["palette"]
    constraints = []
    source_elements = {
        str(item.get("role")): item for item in layout["elements"]
    }
    source_underlays = {
        str(item.get("id")): item for item in layout.get("underlays", [])
    }

    def bounded_box(source: dict, label: str) -> dict[str, int]:
        box_width = _clamp(int(source["width"]), 1, width)
        box_height = _clamp(int(source["height"]), 1, height)
        x = _clamp(int(source["x"]), 0, width - box_width)
        y = _clamp(int(source["y"]), 0, height - box_height)
        requested = {
            "x": int(source["x"]), "y": int(source["y"]),
            "width": int(source["width"]), "height": int(source["height"]),
        }
        applied = {"x": x, "y": y, "width": box_width, "height": box_height}
        if requested != applied:
            constraints.append({
                "target": label, "reason": "clamped_to_canvas",
                "requested": requested, "applied": applied,
            })
        return applied

    def resolve(value: str, current: str) -> str:
        selected = str(value).strip()
        if selected == "keep":
            return current
        return _resolve_color_token(selected, palette)

    elements = []
    for element_target in target["elements"]:
        role = str(element_target["role"])
        source = source_elements.get(role)
        if source is None:
            continue
        requested_lines = int(element_target["max_lines"])
        max_lines = 1 if role in {"title", "price", "cta"} else requested_lines
        if max_lines != requested_lines:
            constraints.append({
                "target": role, "reason": "single_line_role",
                "requested": requested_lines, "applied": max_lines,
            })
        item = {
            "id": f"copy-{role}",
            "role": role,
            "design_group": (
                "headline" if role in {"title", "subtitle"} else "offer"
            ),
            "content": source["content"],
            **bounded_box(element_target, role),
            "font_size": max(8, int(element_target["font_size"])),
            "font_weight": int(element_target["font_weight"]),
            "tracking": int(element_target["tracking"]),
            "text_align": str(element_target["text_align"]),
            "vertical_align": "center",
            "max_lines": max_lines,
            "line_height": float(element_target["line_height"]),
            "color": resolve(
                str(element_target["color"]),
                str(source.get("color", palette["dark"])),
            ),
            "shadow_offset": int(element_target["shadow_offset"]),
            "shadow_color": resolve(
                str(element_target["shadow_color"]),
                str(source.get("shadow_color", "#000000")),
            ),
            "stroke_width": int(element_target["stroke_width"]),
            "stroke_color": resolve(
                str(element_target["stroke_color"]),
                str(source.get("stroke_color", "#000000")),
            ),
            "z_index": 2,
        }
        elements.append(item)

    elements_by_role = {item["role"]: item for item in elements}
    price = elements_by_role.get("price")
    if price is not None:
        composition = target["price_composition"]
        for name in (
            "number_scale", "unit_scale", "number_baseline_shift",
            "unit_baseline_shift",
        ):
            price[name] = round(float(composition[name]), 4)
        if price["number_baseline_shift"] > price["unit_baseline_shift"]:
            requested_shift = price["number_baseline_shift"]
            price["number_baseline_shift"] = price["unit_baseline_shift"]
            constraints.append({
                "target": "price_composition",
                "reason": "number_baseline_cannot_sit_below_price_text",
                "requested": requested_shift,
                "applied": price["number_baseline_shift"],
            })

    underlays = []
    for surface_target in target["surfaces"]:
        group = str(surface_target["group"])
        if not surface_target["enabled"]:
            continue
        source = source_underlays.get(f"surface-{group}", {})
        effect = _resolve_surface_effect(
            surface_target["effect"], palette, source
        )
        underlays.append({
            "id": f"surface-{group}",
            "design_group": group,
            **bounded_box(surface_target, f"{group}_surface"),
            **effect,
            "z_index": 0,
        })

    accent_target = target["accent_rule"]
    if accent_target["present"]:
        source = source_underlays.get("accent-rule", {})
        rule_box = bounded_box(accent_target, "accent_rule")
        underlays.append({
            "id": "accent-rule",
            "design_group": "headline",
            **rule_box,
            "fill_type": "solid",
            "fill_colors": [
                resolve(
                    str(accent_target["color"]),
                    str((source.get("fill_colors") or [palette["accent"]])[0]),
                )
            ] * 2,
            "fill_stops": [0.0, 1.0],
            "gradient_angle": 0.0, "opacity": 1.0,
            "border_radius": max(0, int(rule_box["height"]) // 2),
            "backdrop_blur": 0, "blend_mode": "normal",
            "border_enabled": False, "border_color": palette["accent"],
            "border_width": 0, "border_opacity": 0.0,
            "shadow_enabled": False, "shadow_color": palette["dark"],
            "shadow_offset_x": 0, "shadow_offset_y": 0,
            "shadow_blur": 0, "shadow_opacity": 0.0,
            "z_index": 1,
        })

    design_groups = {}
    for group in ("headline", "offer"):
        group_items = [
            item for item in elements if item["design_group"] == group
        ]
        if not group_items:
            continue
        left = min(int(item["x"]) for item in group_items)
        top = min(int(item["y"]) for item in group_items)
        right = max(int(item["x"]) + int(item["width"]) for item in group_items)
        bottom = max(int(item["y"]) + int(item["height"]) for item in group_items)
        design_groups[group] = {
            "x": left, "y": top, "width": right - left, "height": bottom - top,
        }
    if "protected_subject" in layout.get("design_groups", {}):
        design_groups["protected_subject"] = deepcopy(
            layout["design_groups"]["protected_subject"]
        )

    surface_items = {
        item["design_group"]: item
        for item in underlays
        if str(item.get("id", "")).startswith("surface-")
    }
    padding = max(4, round(min(width, height) * 0.012))
    for group in ("headline", "offer"):
        surface = surface_items.get(group)
        group_box = design_groups.get(group)
        if surface is None or group_box is None:
            continue
        left = min(int(surface["x"]), max(0, int(group_box["x"]) - padding))
        top = min(int(surface["y"]), max(0, int(group_box["y"]) - padding))
        right = max(
            int(surface["x"]) + int(surface["width"]),
            min(width, int(group_box["x"]) + int(group_box["width"]) + padding),
        )
        bottom = max(
            int(surface["y"]) + int(surface["height"]),
            min(height, int(group_box["y"]) + int(group_box["height"]) + padding),
        )
        contained = {
            "x": left, "y": top, "width": right - left, "height": bottom - top,
        }
        if group == "headline":
            contained["x"] = 0
            contained["width"] = width
        current = {key: int(surface[key]) for key in ("x", "y", "width", "height")}
        if contained != current:
            surface.update(contained)
            constraints.append({
                "target": f"{group}_surface",
                "reason": (
                    "full_canvas_width_headline_surface"
                    if group == "headline"
                    else "expanded_to_contain_text"
                ),
                "requested": current, "applied": contained,
            })

    headline_alignment = elements_by_role.get("title", {}).get(
        "text_align", "left"
    )
    offer_role = "price" if "price" in elements_by_role else "cta"
    offer_alignment = elements_by_role.get(offer_role, {}).get(
        "text_align", "left"
    )
    offer_items = [
        elements_by_role[role]
        for role in ("price", "cta") if role in elements_by_role
    ]
    arrangement = "vertical"
    if len(offer_items) == 2:
        first, second = offer_items
        if abs(int(first["y"]) - int(second["y"])) < max(
            int(first["height"]), int(second["height"])
        ):
            arrangement = "horizontal"

    return {
        "canvas": {"width": width, "height": height},
        "elements": elements,
        "underlays": underlays,
        "warnings": [],
        "design_groups": design_groups,
        "design_tokens": {
            "palette": deepcopy(palette),
            "headline_alignment": headline_alignment,
            "offer_alignment": offer_alignment,
            "offer_arrangement": arrangement,
            "redesign_concept": revision.get("redesign_plan", {}).get(
                "concept",
                layout.get("design_tokens", {}).get("redesign_concept", ""),
            ),
            "headline_band": {},
            "offer_band": {},
        },
        "final_review_constraints": constraints,
    }


__all__ = [
    "apply_final_review_revision",
    "build_design_layout",
]
