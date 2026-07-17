from __future__ import annotations

from copy import deepcopy

from .schemas import COPY_ROLES


GRID_SIZE = 5


def _integer(value, default: int) -> int:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return default


def _clamp_bbox(item: dict, width: int, height: int) -> dict:
    x = max(0, min(_integer(item.get("x"), 0), width - 1))
    y = max(0, min(_integer(item.get("y"), 0), height - 1))
    box_width = max(1, _integer(item.get("width"), 1))
    box_height = max(1, _integer(item.get("height"), 1))
    box_width = min(box_width, width - x)
    box_height = min(box_height, height - y)
    return {
        "x": x,
        "y": y,
        "width": box_width,
        "height": box_height,
    }


def _grid_region(
    row: int,
    col: int,
    row_span: int,
    col_span: int,
    width: int,
    height: int,
) -> dict:
    x0 = round(col * width / GRID_SIZE)
    y0 = round(row * height / GRID_SIZE)
    x1 = round((col + col_span) * width / GRID_SIZE)
    y1 = round((row + row_span) * height / GRID_SIZE)
    return {
        "x": x0,
        "y": y0,
        "width": max(1, x1 - x0),
        "height": max(1, y1 - y0),
    }


def normalize_plan_grid(
    plan: dict,
    width: int,
    height: int,
) -> dict:
    """Normalize every planned region onto an aspect-ratio-aware 5x5 grid."""
    if not isinstance(plan, dict):
        raise ValueError("Placement plan response must be an object.")
    adjusted = deepcopy(plan)
    elements = adjusted.get("elements")
    if not isinstance(elements, list):
        raise ValueError("Placement plan must contain an elements array.")

    for element in elements:
        row = max(
            0,
            min(_integer(element.get("grid_row"), 0), GRID_SIZE - 1),
        )
        col = max(
            0,
            min(_integer(element.get("grid_col"), 0), GRID_SIZE - 1),
        )
        row_span = max(
            1,
            min(_integer(element.get("row_span"), 1), GRID_SIZE - row),
        )
        requested_col_span = _integer(element.get("col_span"), 1)
        if element.get("role") == "title":
            requested_col_span = max(2, requested_col_span)
            col = min(col, GRID_SIZE - 2)
        col_span = max(
            1,
            min(requested_col_span, GRID_SIZE - col),
        )
        element.update(
            {
                "grid_row": row,
                "grid_col": col,
                "row_span": row_span,
                "col_span": col_span,
                "preferred_region": _grid_region(
                    row,
                    col,
                    row_span,
                    col_span,
                    width,
                    height,
                ),
            }
        )
    return adjusted


def _boxes_overlap(first: dict, second: dict) -> bool:
    return (
        int(first["x"]) < int(second["x"]) + int(second["width"])
        and int(second["x"]) < int(first["x"]) + int(first["width"])
        and int(first["y"]) < int(second["y"]) + int(second["height"])
        and int(second["y"]) < int(first["y"]) + int(first["height"])
    )


def normalize_layout(
    layout: dict,
    copy: dict[str, str],
    width: int,
    height: int,
    plan: dict | None = None,
) -> dict:
    if not isinstance(layout, dict):
        raise ValueError("Layout response must be an object.")

    raw_elements = layout.get("elements")
    if not isinstance(raw_elements, list):
        raise ValueError("Layout response must contain an elements array.")

    by_role: dict[str, dict] = {}
    for raw in raw_elements:
        if not isinstance(raw, dict):
            continue
        role = str(raw.get("role", ""))
        if role in copy and role not in by_role:
            by_role[role] = raw

    missing = [role for role in copy if role not in by_role]
    if missing:
        raise ValueError(
            "Layout response omitted copy roles: " + ", ".join(missing)
        )

    elements = []
    for role in COPY_ROLES:
        if role not in copy:
            continue
        element = deepcopy(by_role[role])
        element.update(_clamp_bbox(element, width, height))
        element["id"] = str(element.get("id") or f"text-{role}")
        element["role"] = role
        element["content"] = copy[role]
        element["z_index"] = max(1, _integer(element.get("z_index"), 2))
        element["font_size"] = max(
            8,
            min(_integer(element.get("font_size"), 24), height),
        )
        element["font_weight"] = max(
            100,
            min(_integer(element.get("font_weight"), 600), 900),
        )
        element["max_lines"] = (
            1
            if role == "title"
            else max(
                1,
                min(_integer(element.get("max_lines"), 2), 4),
            )
        )
        elements.append(element)

    element_ids = {item["id"] for item in elements}
    underlays = []
    for raw in layout.get("underlays", []):
        if not isinstance(raw, dict):
            continue
        targets = [
            str(target)
            for target in raw.get("target_ids", [])
            if str(target) in element_ids
        ]
        if not targets:
            continue
        underlay = deepcopy(raw)
        underlay.update(_clamp_bbox(underlay, width, height))
        underlay["id"] = str(
            underlay.get("id") or f"underlay-{len(underlays)}"
        )
        underlay["target_ids"] = targets
        underlay["z_index"] = max(
            0,
            min(
                _integer(underlay.get("z_index"), 1),
                min(
                    item["z_index"]
                    for item in elements
                    if item["id"] in targets
                )
                - 1,
            ),
        )
        underlay["opacity"] = max(
            0.0,
            min(float(underlay.get("opacity", 0.65)), 1.0),
        )
        underlay["border_radius"] = max(
            0,
            _integer(underlay.get("border_radius"), 0),
        )
        underlays.append(underlay)

    warnings = [
        str(item).strip()
        for item in layout.get("warnings", [])
        if str(item).strip()
    ]
    for index, first in enumerate(elements):
        for second in elements[index + 1:]:
            if _boxes_overlap(first, second):
                warnings.append(
                    f"{first['role']} overlaps {second['role']}."
                )

    if plan is not None:
        for protected in plan.get("protected_regions", []):
            bbox = protected.get("bbox")
            if not isinstance(bbox, dict):
                continue
            for element in elements:
                if _boxes_overlap(element, bbox):
                    warnings.append(
                        f"{element['role']} overlaps protected region "
                        f"{protected.get('label', 'unknown')}."
                    )

    return {
        "canvas": {"width": width, "height": height},
        "elements": elements,
        "underlays": underlays,
        "rationale": str(layout.get("rationale", "")).strip(),
        "warnings": list(dict.fromkeys(warnings)),
    }
