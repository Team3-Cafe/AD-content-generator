from __future__ import annotations

import colorsys
from pathlib import Path

from PIL import Image, ImageFilter, ImageStat


def _hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02X}{:02X}{:02X}".format(*rgb)


def _luminance(rgb: tuple[int, int, int]) -> float:
    return (
        0.2126 * rgb[0]
        + 0.7152 * rgb[1]
        + 0.0722 * rgb[2]
    ) / 255.0


def _palette(image: Image.Image) -> dict[str, str]:
    thumbnail = image.convert("RGB")
    thumbnail.thumbnail((128, 128))
    quantized = thumbnail.quantize(colors=16)
    raw_palette = quantized.getpalette() or []
    colors = []
    for count, index in quantized.getcolors() or []:
        offset = index * 3
        rgb = tuple(raw_palette[offset:offset + 3])
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
        _hue, saturation, value = colorsys.rgb_to_hsv(
            rgb[0] / 255,
            rgb[1] / 255,
            rgb[2] / 255,
        )
        usable = 1.0 if 0.22 <= value <= 0.92 else 0.2
        return saturation * usable * max(1, count) ** 0.25

    accent = max(colors, key=accent_score)[1]
    saturation = colorsys.rgb_to_hsv(
        accent[0] / 255,
        accent[1] / 255,
        accent[2] / 255,
    )[1]
    if saturation < 0.20:
        accent = (255, 210, 63)
    return {
        "dark": _hex(dark),
        "light": _hex(light),
        "accent": _hex(accent),
    }


def analyze_image_space(
    image_path: str | Path,
    *,
    grid_size: int = 6,
) -> dict:
    """Describe color, brightness, and low-detail space without scoring art."""
    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    width, height = image.size
    gray = image.convert("L")
    edges = gray.filter(ImageFilter.FIND_EDGES)
    cells = []
    for row in range(grid_size):
        for column in range(grid_size):
            left = round(column * width / grid_size)
            top = round(row * height / grid_size)
            right = round((column + 1) * width / grid_size)
            bottom = round((row + 1) * height / grid_size)
            gray_crop = gray.crop((left, top, right, bottom))
            edge_crop = edges.crop((left, top, right, bottom))
            gray_stats = ImageStat.Stat(gray_crop)
            edge_stats = ImageStat.Stat(edge_crop)
            luminance = float(gray_stats.mean[0]) / 255.0
            contrast = float(gray_stats.stddev[0]) / 128.0
            edge_density = float(edge_stats.mean[0]) / 255.0
            quietness = max(
                0.0,
                1.0 - contrast * 0.45 - edge_density * 0.75,
            )
            cells.append(
                {
                    "column": column,
                    "row": row,
                    "bbox": {
                        "x": round(left / width, 4),
                        "y": round(top / height, 4),
                        "width": round((right - left) / width, 4),
                        "height": round((bottom - top) / height, 4),
                    },
                    "luminance": round(luminance, 4),
                    "contrast": round(contrast, 4),
                    "edge_density": round(edge_density, 4),
                    "quietness": round(quietness, 4),
                }
            )

    quiet_regions = sorted(
        cells,
        key=lambda item: item["quietness"],
        reverse=True,
    )[:8]
    overall_luminance = float(ImageStat.Stat(gray).mean[0]) / 255.0
    return {
        "image": str(image_path.resolve()),
        "canvas": {
            "width": width,
            "height": height,
            "aspect_ratio": round(width / max(1, height), 4),
        },
        "palette": _palette(image),
        "overall_luminance": round(overall_luminance, 4),
        "quiet_regions": quiet_regions,
    }


__all__ = ["analyze_image_space"]
