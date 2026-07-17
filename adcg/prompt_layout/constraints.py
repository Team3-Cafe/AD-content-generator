from __future__ import annotations


def _expanded(item: dict, padding: int) -> dict:
    return {
        "x": int(item["x"]) - padding,
        "y": int(item["y"]) - padding,
        "width": int(item["width"]) + padding * 2,
        "height": int(item["height"]) + padding * 2,
    }


def _overlap(first: dict, second: dict) -> bool:
    return (
        int(first["x"]) < int(second["x"]) + int(second["width"])
        and int(second["x"]) < int(first["x"]) + int(first["width"])
        and int(first["y"]) < int(second["y"]) + int(second["height"])
        and int(second["y"]) < int(first["y"]) + int(first["height"])
    )


def _targets(item: dict) -> set[str]:
    return {str(value) for value in item.get("target_ids", [])}


def layout_geometry_violations(
    layout: dict,
    *,
    plan: dict | None = None,
    include_underlays: bool = True,
) -> list[str]:
    """Return hard geometry violations that disqualify a candidate."""
    canvas = layout["canvas"]
    width = int(canvas["width"])
    height = int(canvas["height"])
    short_side = min(width, height)
    outer_margin = max(6, round(short_side * 0.03))
    text_padding = max(4, round(short_side * 0.015))
    panel_padding = max(3, round(short_side * 0.01))
    violations = []
    elements = layout["elements"]

    for item in elements:
        if (
            int(item["x"]) < outer_margin
            or int(item["y"]) < outer_margin
            or int(item["x"]) + int(item["width"])
            > width - outer_margin
            or int(item["y"]) + int(item["height"])
            > height - outer_margin
        ):
            violations.append(
                f"{item['role']} violates the outer margin."
            )
        if item.get("role") == "cta" and int(
            item.get("max_lines", 1)
        ) != 1:
            violations.append("CTA must be one line.")

    for index, first in enumerate(elements):
        for second in elements[index + 1:]:
            if _overlap(
                _expanded(first, text_padding),
                _expanded(second, text_padding),
            ):
                violations.append(
                    f"{first['role']} is too close to {second['role']}."
                )

    if plan is not None:
        for protected in plan.get("protected_regions", []):
            bbox = protected.get("bbox")
            if not isinstance(bbox, dict):
                continue
            for item in elements:
                if _overlap(item, bbox):
                    violations.append(
                        f"{item['role']} intersects protected region "
                        f"{protected.get('label', 'unknown')}."
                    )

    if include_underlays:
        underlays = layout.get("underlays", [])
        for index, first in enumerate(underlays):
            for second in underlays[index + 1:]:
                if _targets(first) == _targets(second):
                    continue
                if _overlap(
                    _expanded(first, panel_padding),
                    _expanded(second, panel_padding),
                ):
                    violations.append(
                        f"{first.get('id')} collides with "
                        f"{second.get('id')}."
                    )
        for underlay in underlays:
            targets = _targets(underlay)
            for item in elements:
                if item["id"] in targets:
                    continue
                if _overlap(underlay, item):
                    violations.append(
                        f"{underlay.get('id')} covers {item['role']}."
                    )

    return list(dict.fromkeys(violations))


def layout_geometry_signature(layout: dict) -> tuple:
    """Return a coarse position signature for geometry diversity checks."""
    canvas = layout["canvas"]
    width = max(1, int(canvas["width"]))
    height = max(1, int(canvas["height"]))
    return tuple(
        (
            item["role"],
            round((int(item["x"]) + int(item["width"]) / 2) * 5 / width),
            round((int(item["y"]) + int(item["height"]) / 2) * 5 / height),
        )
        for item in sorted(
            layout["elements"],
            key=lambda value: str(value["role"]),
        )
    )


__all__ = [
    "layout_geometry_signature",
    "layout_geometry_violations",
]
