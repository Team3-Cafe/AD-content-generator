from __future__ import annotations

from copy import deepcopy

from .schemas import COPY_ROLES


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


def normalize_layout(
    layout: dict,
    copy: dict[str, str],
    width: int,
    height: int,
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
        element["max_lines"] = max(
            1,
            min(_integer(element.get("max_lines"), 2), 4),
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

    return {
        "canvas": {"width": width, "height": height},
        "elements": elements,
        "underlays": underlays,
        "rationale": str(layout.get("rationale", "")).strip(),
        "warnings": [
            str(item).strip()
            for item in layout.get("warnings", [])
            if str(item).strip()
        ],
    }
