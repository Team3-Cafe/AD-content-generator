from __future__ import annotations

import colorsys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageStat


def _hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02X}{:02X}{:02X}".format(*rgb)


def _luminance(rgb: tuple[int, int, int]) -> float:
    return (
        0.2126 * rgb[0]
        + 0.7152 * rgb[1]
        + 0.0722 * rgb[2]
    ) / 255.0


def _mix_color(
    rgb: tuple[int, int, int],
    target: tuple[int, int, int],
    amount: float,
) -> tuple[int, int, int]:
    return tuple(
        round(channel + (destination - channel) * amount)
        for channel, destination in zip(rgb, target)
    )


def _shift_hue(rgb: tuple[int, int, int], degrees: float) -> tuple[int, int, int]:
    hue, saturation, value = colorsys.rgb_to_hsv(
        *(channel / 255 for channel in rgb)
    )
    shifted = colorsys.hsv_to_rgb(
        (hue + degrees / 360.0) % 1.0, saturation, value
    )
    return tuple(round(channel * 255) for channel in shifted)


def _cluster_colors(image: Image.Image, count: int = 20) -> list[tuple[int, tuple[int, int, int]]]:
    """Extract perceptually grouped image colors with deterministic OpenCV k-means."""
    thumbnail = image.convert("RGB")
    thumbnail.thumbnail((192, 192))
    pixels = np.asarray(thumbnail, dtype=np.uint8).reshape(-1, 3)
    if not len(pixels):
        return []
    lab = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB)
    samples = lab.reshape(-1, 3).astype(np.float32)
    cluster_count = min(count, len(np.unique(samples, axis=0)))
    if cluster_count < 1:
        return []
    cv2.setRNGSeed(17)
    _compactness, labels, centers = cv2.kmeans(
        samples,
        cluster_count,
        None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.25),
        5,
        cv2.KMEANS_PP_CENTERS,
    )
    rgb_centers = cv2.cvtColor(
        np.clip(np.rint(centers), 0, 255).astype(np.uint8).reshape(-1, 1, 3),
        cv2.COLOR_LAB2RGB,
    ).reshape(-1, 3)
    counts = np.bincount(labels.reshape(-1), minlength=cluster_count)
    return sorted(
        (
            (int(counts[index]), tuple(int(value) for value in rgb_centers[index]))
            for index in range(cluster_count)
        ),
        reverse=True,
    )


def _palette(image: Image.Image) -> dict[str, object]:
    colors = _cluster_colors(image)
    if not colors:
        return {
            "dark": "#101820",
            "light": "#F7F4EC",
            "accent": "#FFD23F",
            "swatches": ["#101820", "#F7F4EC", "#FFD23F"],
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
    swatches: list[tuple[int, int, int]] = []
    for _count, rgb in sorted(colors, key=lambda item: item[0], reverse=True):
        if all(
            sum((rgb[index] - existing[index]) ** 2 for index in range(3))
            >= 32 ** 2
            for existing in swatches
        ):
            swatches.append(rgb)
        if len(swatches) == 16:
            break
    for required in (dark, light, accent):
        if required not in swatches:
            swatches.append(required)
    return {
        "dark": _hex(dark),
        "light": _hex(light),
        "accent": _hex(accent),
        "swatches": [_hex(rgb) for rgb in swatches[:16]],
        "dominant_colors": [_hex(rgb) for _count, rgb in colors[:12]],
        "accent_candidates": [
            _hex(rgb)
            for _count, rgb in sorted(colors, key=accent_score, reverse=True)[:8]
        ],
        "tonal_variants": {
            _hex(rgb): [
                _hex(_mix_color(rgb, (255, 255, 255), amount))
                for amount in (0.18, 0.35, 0.55)
            ] + [
                _hex(_mix_color(rgb, (0, 0, 0), amount))
                for amount in (0.18, 0.35, 0.55)
            ]
            for _count, rgb in colors[:6]
        },
        "harmony_sets": [
            {
                "type": "analogous",
                "colors": [_hex(_shift_hue(accent, angle)) for angle in (-30, 0, 30)],
            },
            {
                "type": "complementary",
                "colors": [_hex(accent), _hex(_shift_hue(accent, 180))],
            },
            {
                "type": "split_complementary",
                "colors": [
                    _hex(accent), _hex(_shift_hue(accent, 150)),
                    _hex(_shift_hue(accent, 210)),
                ],
            },
            {
                "type": "tonal",
                "colors": [
                    _hex(_mix_color(accent, (255, 255, 255), 0.55)),
                    _hex(accent),
                    _hex(_mix_color(accent, (0, 0, 0), 0.55)),
                ],
            },
        ],
    }


def build_design_candidate_pool(
    image_analysis: dict,
    ad_copy: dict[str, str],
    subject_region: dict | None = None,
) -> dict:
    """Build broad, image-aware affordances for the final VLM redesign."""
    canvas = image_analysis["canvas"]
    width, height = int(canvas["width"]), int(canvas["height"])
    quiet_regions = image_analysis.get("quiet_regions", [])

    def pixel_region(item: dict) -> dict:
        box = item["bbox"]
        return {
            "x": round(float(box["x"]) * width),
            "y": round(float(box["y"]) * height),
            "width": round(float(box["width"]) * width),
            "height": round(float(box["height"]) * height),
            "quietness": item.get("quietness"),
            "luminance": item.get("luminance"),
            "mean_color": item.get("mean_color"),
        }

    safe_regions = [pixel_region(item) for item in quiet_regions]
    palette = image_analysis["palette"]
    copy_roles = [role for role in ("title", "subtitle", "price", "cta") if ad_copy.get(role)]
    return {
        "usage": (
            "These are affordances, not templates or limits. Compare, combine, alter, "
            "or reject them and author exact values when a better design is visible."
        ),
        "typography": [
            "single-weight editorial restraint", "high-contrast title/subtitle scale",
            "compact industrial grotesk", "wide-tracked premium display",
            "dense headline with quiet support copy", "price-led numeric display",
            "soft hierarchy with moderate weights", "outlined or shadow-assisted overlay",
        ],
        "hierarchy": [
            "title dominant", "price dominant", "product dominant with restrained copy",
            "title and price dual anchors", "progressive title-subtitle-offer sequence",
            "compact headline with isolated offer", "balanced equal-weight groups",
        ],
        "spacing": [
            "open editorial spacing", "compact campaign spacing", "tight internal loose external",
            "loose internal compact external", "rhythmic stepped gaps", "group overlap with clear padding",
            "edge-anchored negative space", "full-width bands with asymmetric insets",
        ],
        "price_composition": ([
            "oversized number with compact qualifier and unit", "single-size inline price",
            "baseline-stepped number and unit", "right-aligned numeric anchor",
            "left-aligned qualifier-number-unit sequence", "price isolated from supporting CTA",
            "price integrated with offer copy", "compact price without numeric exaggeration",
        ] if ad_copy.get("price") else ["omit price composition because no price copy exists"]),
        "band_proportion": [
            "no surfaces", "content-width headline only", "content-width offer only",
            "two independent content surfaces", "full-width headline band", "full-width offer band",
            "asymmetric partial-width surface", "thin scrim", "large translucent field",
            "overlapping surface and open text group",
        ],
        "accent_rule": [
            "no accent", "short title underline", "long group divider", "vertical side rule",
            "price-adjacent rule", "offset asymmetric dash", "full-width hairline",
            "small color block used as punctuation",
        ],
        "placement": {
            "safe_region_candidates_px": safe_regions,
            "composition_models": [
                "top headline and bottom offer", "top headline and mid-side offer",
                "side-stacked copy column", "opposing-corner groups", "single consolidated copy zone",
                "asymmetric floating groups", "edge-anchored editorial layout",
                "open overlay without surfaces", "split horizontal bands", "product-adjacent offer anchor",
            ],
        },
        "color": {
            "swatches": palette.get("swatches", []),
            "dominant_colors": palette.get("dominant_colors", []),
            "accent_candidates": palette.get("accent_candidates", []),
            "tonal_variants": palette.get("tonal_variants", {}),
            "harmony_sets": palette.get("harmony_sets", []),
            "strategies": [
                "image-tonal monochrome", "product-color accent", "analogous harmony",
                "complementary contrast", "split-complementary accent", "warm-neutral editorial",
                "cool industrial", "dark translucent overlay", "light translucent overlay",
                "independent headline and offer palettes",
            ],
        },
        "contrast": [
            "direct high-contrast text", "local scrim behind text", "subtle text shadow",
            "thin text stroke", "opaque surface", "translucent blurred surface",
            "gradient fading toward image", "relocate text to a quieter region",
            "dark-on-light and light-on-dark split treatment",
        ],
        "cta": ([
            "quiet supporting line", "accent-colored plain text", "price-aligned plain text",
            "separate typographic anchor", "integrated offer line", "small uppercase-style emphasis",
            "omit visual emphasis while preserving copy",
        ] if ad_copy.get("cta") else ["omit CTA and reserve no CTA-specific space"]),
        "product_visibility": {
            "subject_region_normalized": subject_region,
            "strategies": [
                "strict no-overlap", "allow only transparent scrim near subject edge",
                "place copy on opposing side", "frame subject with two copy groups",
                "use negative space above subject", "use negative space below subject",
                "keep high-contrast accents away from product silhouette",
                "align copy to subject edge without covering it",
            ],
        },
        "alignment_combinations": [
            {"headline": headline, "offer": offer}
            for headline in ("left", "center", "right")
            for offer in ("left", "center", "right")
        ],
        "surface_effect_components": {
            "fill_types": ["none", "solid", "linear_gradient", "radial_gradient", "scrim"],
            "blend_modes": ["normal", "multiply", "screen", "overlay"],
            "edge_treatments": ["square", "rounded", "bordered", "shadowed", "blurred"],
            "width_modes": ["none", "content_width", "partial_width", "full_width"],
        },
        "copy_roles": copy_roles,
    }


def analyze_image_space(
    image_path: str | Path,
    *,
    grid_size: int = 6,
    band_count: int = 12,
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
            color_crop = image.crop((left, top, right, bottom))
            gray_crop = gray.crop((left, top, right, bottom))
            edge_crop = edges.crop((left, top, right, bottom))
            color_stats = ImageStat.Stat(color_crop)
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
                    "mean_color": _hex(tuple(
                        round(value) for value in color_stats.mean[:3]
                    )),
                }
            )

    quiet_regions = sorted(
        cells,
        key=lambda item: item["quietness"],
        reverse=True,
    )[:8]
    horizontal_bands = []
    for index in range(band_count):
        top = round(index * height / band_count)
        bottom = round((index + 1) * height / band_count)
        gray_crop = gray.crop((0, top, width, bottom))
        edge_crop = edges.crop((0, top, width, bottom))
        gray_stats = ImageStat.Stat(gray_crop)
        edge_stats = ImageStat.Stat(edge_crop)
        horizontal_bands.append(
            {
                "index": index,
                "y": round(top / height, 4),
                "height": round((bottom - top) / height, 4),
                "luminance": round(float(gray_stats.mean[0]) / 255.0, 4),
                "contrast": round(float(gray_stats.stddev[0]) / 128.0, 4),
                "edge_density": round(
                    float(edge_stats.mean[0]) / 255.0,
                    4,
                ),
            }
        )
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
        "horizontal_bands": horizontal_bands,
    }


__all__ = ["analyze_image_space", "build_design_candidate_pool"]
