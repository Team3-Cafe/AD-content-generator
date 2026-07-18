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
    """Compile one VLM art direction into a relational pixel layout."""
    canvas = image_analysis["canvas"]
    width = int(canvas["width"])
    height = int(canvas["height"])
    short_side = min(width, height)
    margin = max(12, round(short_side * 0.045))
    palette = image_analysis["palette"]
    direction = design_spec["art_direction"]
    composition = design_spec["composition"]
    alignment = direction["alignment"]
    density = direction["spacing_density"]
    gap_ratio = {"compact": 0.012, "balanced": 0.020, "airy": 0.030}[
        density
    ]
    inner_gap = max(6, round(short_side * gap_ratio))

    contrast_mode = direction["contrast_mode"]
    if contrast_mode == "light":
        text_color = palette["light"]
    elif contrast_mode == "dark":
        text_color = palette["dark"]
    else:
        text_color = (
            palette["dark"]
            if image_analysis["overall_luminance"] >= 0.58
            else palette["light"]
        )
    opposite = (
        palette["light"]
        if text_color == palette["dark"]
        else palette["dark"]
    )

    title_size = max(
        22,
        round(short_side * 0.072 * float(composition["title_scale"])),
    )
    subtitle_size = max(14, round(title_size * 0.47))
    price_size = max(20, round(title_size * 0.84))
    cta_size = max(14, round(title_size * 0.44))
    headline_width = round(width * float(composition["headline_width_ratio"]))
    title_height = round(title_size * 1.35)
    subtitle_height = round(subtitle_size * 2.8) if ad_copy.get("subtitle") else 0
    headline_height = title_height + (
        inner_gap + subtitle_height if subtitle_height else 0
    )
    headline_anchor = composition["headline_anchor"]
    desired_headline = {
        "x": round(float(headline_anchor["x"]) * width),
        "y": round(float(headline_anchor["y"]) * height),
        "width": headline_width,
        "height": headline_height,
    }
    protected = _normalized_box(
        design_spec["scene_analysis"]["subject_region"],
        width,
        height,
    )
    headline_group = _place_group(
        desired_headline,
        canvas_width=width,
        canvas_height=height,
        margin=margin,
        avoid=[protected],
    )

    has_price = bool(ad_copy.get("price"))
    has_cta = bool(ad_copy.get("cta"))
    offer_width = round(width * float(composition["offer_width_ratio"]))
    arrangement = composition["offer_arrangement"]
    if arrangement == "horizontal" and has_price and has_cta:
        offer_height = max(round(price_size * 1.55), round(cta_size * 2.5))
    else:
        offer_height = (
            (round(price_size * 1.45) if has_price else 0)
            + (inner_gap if has_price and has_cta else 0)
            + (round(cta_size * 2.35) if has_cta else 0)
        )
    offer_anchor = composition["offer_anchor"]
    desired_offer = {
        "x": round(float(offer_anchor["x"]) * width),
        "y": round(float(offer_anchor["y"]) * height),
        "width": offer_width,
        "height": max(1, offer_height),
    }
    offer_group = _place_group(
        desired_offer,
        canvas_width=width,
        canvas_height=height,
        margin=margin,
        avoid=[protected, headline_group],
    )

    elements = []
    title_box = {
        "x": headline_group["x"],
        "y": headline_group["y"],
        "width": headline_group["width"],
        "height": title_height,
    }
    elements.append(
        _element(
            "title",
            ad_copy["title"],
            "headline",
            title_box,
            font_size=title_size,
            font_weight=820,
            color=text_color,
            align=alignment,
            max_lines=1,
            line_height=1.1,
        )
    )
    if ad_copy.get("subtitle"):
        subtitle_box = {
            "x": headline_group["x"],
            "y": headline_group["y"] + title_height + inner_gap,
            "width": headline_group["width"],
            "height": subtitle_height,
        }
        elements.append(
            _element(
                "subtitle",
                ad_copy["subtitle"],
                "headline",
                subtitle_box,
                font_size=subtitle_size,
                font_weight=480,
                color=text_color,
                align=alignment,
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
            "width": max(1, offer_group["width"] - price_width - inner_gap),
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
        price_color = (
            palette["accent"]
            if accent_role in {"price", "price_and_cta"}
            else text_color
        )
        price_item = _element(
            "price",
            ad_copy["price"],
            "offer",
            offer_boxes["price"],
            font_size=price_size,
            font_weight=800,
            color=price_color,
            align=alignment,
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
                    palette["dark"]
                    if accent_role in {"cta", "price_and_cta"}
                    else text_color
                ),
                align="center" if accent_role in {"cta", "price_and_cta"} else alignment,
                max_lines=1,
                line_height=1.0,
            )
        )

    underlays = []
    panel_padding = max(8, round(short_side * 0.018))
    radius = max(8, round(short_side * 0.022))
    headline_surface = direction["headline_surface"]
    if headline_surface != "none":
        box = {
            "x": max(0, headline_group["x"] - panel_padding),
            "y": max(0, headline_group["y"] - panel_padding),
            "width": min(
                width - max(0, headline_group["x"] - panel_padding),
                headline_group["width"] + panel_padding * 2,
            ),
            "height": min(
                height - max(0, headline_group["y"] - panel_padding),
                headline_group["height"] + panel_padding * 2,
            ),
        }
        panel_color = opposite
        underlays.append(
            _panel(
                "surface-headline",
                "headline",
                box,
                background=panel_color,
                gradient=(
                    panel_color
                    if headline_surface == "soft_panel"
                    else palette["dark"]
                ),
                opacity=0.68 if headline_surface == "soft_panel" else 0.58,
                radius=radius,
            )
        )
        for item in elements:
            if item["design_group"] == "headline":
                item["color"] = text_color

    rule_width = max(3, round(short_side * 0.009))
    if alignment == "center":
        rule_length = max(rule_width * 6, round(headline_group["width"] * 0.18))
        rule_box = {
            "x": headline_group["x"] + (headline_group["width"] - rule_length) // 2,
            "y": headline_group["y"] + title_height + max(2, inner_gap // 3),
            "width": rule_length,
            "height": rule_width,
        }
    else:
        rule_box = {
            "x": (
                headline_group["x"] + headline_group["width"] + panel_padding - rule_width
                if alignment == "right"
                else max(0, headline_group["x"] - panel_padding)
            ),
            "y": headline_group["y"],
            "width": rule_width,
            "height": min(
                headline_group["height"],
                round(short_side * 0.14),
            ),
        }
    underlays.append(
        _panel(
            "accent-rule",
            "headline",
            rule_box,
            background=palette["accent"],
            opacity=1.0,
            radius=max(1, rule_width // 2),
            z_index=1,
        )
    )

    offer_surface = direction["offer_surface"]
    if offer_surface != "none" and (has_price or has_cta):
        box = {
            "x": max(0, offer_group["x"] - panel_padding),
            "y": max(0, offer_group["y"] - panel_padding),
            "width": min(
                width - max(0, offer_group["x"] - panel_padding),
                offer_group["width"] + panel_padding * 2,
            ),
            "height": min(
                height - max(0, offer_group["y"] - panel_padding),
                offer_group["height"] + panel_padding * 2,
            ),
        }
        underlays.append(
            _panel(
                "surface-offer",
                "offer",
                box,
                background=(
                    palette["accent"]
                    if offer_surface == "solid_lockup"
                    else palette["dark"]
                ),
                gradient=(
                    None
                    if offer_surface == "solid_lockup"
                    else palette["dark"]
                ),
                opacity=0.92 if offer_surface == "solid_lockup" else 0.62,
                radius=radius,
            )
        )
        if offer_surface == "solid_lockup":
            for item in elements:
                if item["design_group"] == "offer":
                    item["color"] = palette["dark"]

    if (
        has_cta
        and accent_role in {"cta", "price_and_cta"}
        and offer_surface != "solid_lockup"
    ):
        cta_box = offer_boxes["cta"]
        underlays.append(
            _panel(
                "surface-cta",
                "offer",
                cta_box,
                background=palette["accent"],
                opacity=0.94,
                radius=max(6, round(cta_box["height"] * 0.22)),
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
            "alignment": alignment,
            "mood": direction["mood"],
            "spacing_density": density,
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
        dx = round(float(changes[f"{prefix}_x_shift"]) * width)
        dy = round(float(changes[f"{prefix}_y_shift"]) * height)
        scale = float(changes[f"{prefix}_scale"])
        group_items = [
            item
            for item in adjusted["elements"] + adjusted.get("underlays", [])
            if item.get("design_group") == group
        ]
        if not group_items:
            continue
        left = min(int(item["x"]) for item in group_items)
        top = min(int(item["y"]) for item in group_items)
        right = max(int(item["x"]) + int(item["width"]) for item in group_items)
        bottom = max(int(item["y"]) + int(item["height"]) for item in group_items)
        dx = _clamp(dx, margin - left, width - margin - right)
        dy = _clamp(dy, margin - top, height - margin - bottom)
        for item in group_items:
            item["x"] = int(item["x"]) + dx
            item["y"] = int(item["y"]) + dy
            if item in adjusted["elements"]:
                item["font_size"] = max(
                    10,
                    round(int(item["font_size"]) * scale),
                )

    opacity_delta = float(changes["surface_opacity_delta"])
    for underlay in adjusted.get("underlays", []):
        if str(underlay.get("id", "")).startswith("surface-"):
            underlay["opacity"] = round(
                max(0.25, min(0.95, float(underlay["opacity"]) + opacity_delta)),
                3,
            )
    return adjusted


__all__ = ["apply_design_revision", "build_design_layout"]
