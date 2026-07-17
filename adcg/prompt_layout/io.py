from __future__ import annotations

import base64
import json
from pathlib import Path

from .schemas import COPY_ROLES


MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


def image_to_data_url(image_path: str | Path) -> str:
    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    mime_type = MIME_TYPES.get(image_path.suffix.lower())
    if mime_type is None:
        raise ValueError(
            f"Unsupported image format: {image_path.suffix}. "
            f"Supported formats: {sorted(MIME_TYPES)}"
        )

    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def load_ad_copy(
    copy_path: str | Path,
    copy_index: int = 0,
) -> dict[str, str]:
    copy_path = Path(copy_path)
    if not copy_path.is_file():
        raise FileNotFoundError(f"Ad copy JSON not found: {copy_path}")

    data = json.loads(copy_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Ad copy JSON must contain an object.")

    if "copies" in data:
        copies = data["copies"]
        if not isinstance(copies, list) or not copies:
            raise ValueError("'copies' must be a non-empty array.")
        if not 0 <= copy_index < len(copies):
            raise IndexError(
                f"copy_index {copy_index} is outside 0..{len(copies) - 1}."
            )
        source = copies[copy_index]
    else:
        source = data

    if not isinstance(source, dict):
        raise ValueError("Selected ad copy must be an object.")

    copy = {
        role: str(source.get(role, "")).strip()
        for role in COPY_ROLES
        if str(source.get(role, "")).strip()
    }
    if not copy:
        raise ValueError(
            "No non-empty title, subtitle, price, or cta was found."
        )
    return copy
