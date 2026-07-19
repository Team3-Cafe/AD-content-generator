from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import re
import shutil
import subprocess

from PIL import Image, ImageChops, ImageColor, ImageDraw, ImageFilter, ImageFont


REGULAR_FONT_CANDIDATES = (
    "C:/Windows/Fonts/malgun.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansKR-Regular.ttf",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
)

BOLD_FONT_CANDIDATES = (
    "C:/Windows/Fonts/malgunbd.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansKR-Bold.ttf",
    "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
)


def _contains_korean(text: str) -> bool:
    return any(
        "\uac00" <= character <= "\ud7a3"
        or "\u1100" <= character <= "\u11ff"
        or "\u3130" <= character <= "\u318f"
        for character in text
    )


def _fontconfig_korean_fonts() -> list[Path]:
    executable = shutil.which("fc-list")
    if executable is None:
        return []
    try:
        result = subprocess.run(
            [executable, ":lang=ko", "file"],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return []

    fonts = []
    for line in result.stdout.splitlines():
        raw_path = line.split(":", 1)[0].strip()
        if raw_path:
            path = Path(raw_path)
            if path.is_file() and path not in fonts:
                fonts.append(path)
    return fonts


def _resolve_font_path(
    font_path: str | Path | None,
    bold: bool,
    text: str,
) -> Path | None:
    if font_path is not None:
        path = Path(font_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Font not found: {path}")
        return path

    environment_font = os.getenv("ADCG_FONT_PATH")
    if environment_font:
        path = Path(environment_font).expanduser()
        if not path.is_file():
            raise FileNotFoundError(
                f"ADCG_FONT_PATH does not exist: {path}"
            )
        return path

    candidates = []
    candidates.extend(
        BOLD_FONT_CANDIDATES if bold else REGULAR_FONT_CANDIDATES
    )
    candidates.extend(
        REGULAR_FONT_CANDIDATES if bold else BOLD_FONT_CANDIDATES
    )
    for candidate in candidates:
        path = Path(candidate)
        if path.is_file():
            if (
                _contains_korean(text)
                and "dejavu" in path.name.lower()
            ):
                continue
            return path

    if _contains_korean(text):
        fontconfig_fonts = _fontconfig_korean_fonts()
        if fontconfig_fonts:
            bold_fonts = [
                path
                for path in fontconfig_fonts
                if "bold" in path.name.lower()
            ]
            if bold and bold_fonts:
                return bold_fonts[0]
            return fontconfig_fonts[0]
    return None


def _load_font(
    size: int,
    resolved_font: Path | None,
    weight: int,
):
    if resolved_font is not None:
        font = ImageFont.truetype(str(resolved_font), size=size)
        try:
            axes = font.get_variation_axes()
            values = []
            for axis in axes:
                name = axis.get("name", b"")
                if isinstance(name, bytes):
                    name = name.decode("utf-8", errors="ignore")
                if "weight" in str(name).lower():
                    value = max(
                        int(axis["minimum"]),
                        min(int(weight), int(axis["maximum"])),
                    )
                else:
                    value = int(axis["default"])
                values.append(value)
            if values:
                font.set_variation_by_axes(values)
        except (AttributeError, KeyError, OSError, TypeError, ValueError):
            pass
        return font
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _text_width(
    draw: ImageDraw.ImageDraw,
    text: str,
    font,
    tracking: int = 0,
) -> int:
    if not text:
        return 0
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0] + max(0, len(text) - 1) * tracking


def _wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font,
    max_width: int,
    tracking: int = 0,
) -> list[str]:
    lines = []
    for paragraph in text.splitlines() or [""]:
        if not paragraph:
            lines.append("")
            continue
        line = ""
        for character in paragraph:
            candidate = line + character
            if (
                line
                and _text_width(draw, candidate, font, tracking)
                > max_width
            ):
                lines.append(line.rstrip())
                line = character.lstrip()
            else:
                line = candidate
        if line or not lines:
            lines.append(line.rstrip())
    return lines


def _fit_text(
    draw: ImageDraw.ImageDraw,
    item: dict,
    font_path: str | Path | None,
):
    text = str(item["content"])
    max_width = int(item["width"])
    max_height = int(item["height"])
    max_lines = int(item.get("max_lines", 2))
    start_size = int(item.get("font_size", 24))
    weight = int(item.get("font_weight", 600))
    line_height = float(item.get("line_height", 1.2))
    tracking = max(0, int(item.get("tracking", 0)))
    resolved_font = _resolve_font_path(
        font_path,
        bold=weight >= 600,
        text=text,
    )
    if _contains_korean(text) and resolved_font is None:
        raise RuntimeError(
            "No Korean-capable font was found. Pass --layout-font to "
            "run_pipeline.py, --font to adcg.prompt_layout, or set "
            "ADCG_FONT_PATH."
        )

    selected = None
    for size in range(start_size, 7, -1):
        font = _load_font(size, resolved_font, weight)
        lines = (
            [text]
            if str(item.get("role")) == "title"
            else _wrap_text(
                draw,
                text,
                font,
                max_width,
                tracking,
            )
        )
        step = max(1, int(round(size * line_height)))
        total_height = step * len(lines)
        selected = (font, lines, step, total_height)
        widest_line = max(
            (
                _text_width(draw, line, font, tracking)
                for line in lines
            ),
            default=0,
        )
        if (
            len(lines) <= max_lines
            and total_height <= max_height
            and widest_line <= max_width
        ):
            break
    return selected


def fit_layout_typography(
    layout: dict,
    *,
    font_path: str | Path | None = None,
) -> dict:
    """Persist the font sizes that fit each VLM-selected text box."""
    adjusted = deepcopy(layout)
    measuring_image = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(measuring_image)
    for item in adjusted["elements"]:
        font, _lines, _step, _height = _fit_text(
            draw,
            item,
            font_path,
        )
        fitted_size = getattr(font, "size", item.get("font_size", 24))
        item["font_size"] = max(8, int(fitted_size))
        if item.get("role") == "title":
            item["max_lines"] = 1
    return adjusted


def _relative_luminance(rgb: tuple[int, int, int]) -> float:
    channels = []
    for value in rgb[:3]:
        channel = value / 255.0
        channels.append(
            channel / 12.92
            if channel <= 0.04045
            else ((channel + 0.055) / 1.055) ** 2.4
        )
    return (
        0.2126 * channels[0]
        + 0.7152 * channels[1]
        + 0.0722 * channels[2]
    )


def _contrast_ratio(first: float, second: float) -> float:
    lighter = max(first, second)
    darker = min(first, second)
    return (lighter + 0.05) / (darker + 0.05)


def _region_luminances(image: Image.Image, item: dict) -> list[float]:
    x = int(item["x"])
    y = int(item["y"])
    crop = image.crop(
        (
            x,
            y,
            x + int(item["width"]),
            y + int(item["height"]),
        )
    ).convert("RGB")
    crop.thumbnail((48, 48))
    return [_relative_luminance(pixel) for pixel in crop.getdata()]


def _contrast_score(
    luminances: list[float],
    text_color: tuple[int, int, int],
) -> float:
    if not luminances:
        return 1.0
    text_luminance = _relative_luminance(text_color)
    ratios = sorted(
        _contrast_ratio(text_luminance, background)
        for background in luminances
    )
    percentile_index = max(0, int(len(ratios) * 0.10) - 1)
    return ratios[percentile_index]


def _surface_gradient(item: dict, size: tuple[int, int]) -> Image.Image:
    """Create a color surface from VLM-selected stops and gradient geometry."""
    width, height = size
    fill_type = str(item.get("fill_type", "solid"))
    colors = [
        ImageColor.getrgb(str(value))[:3]
        for value in item.get("fill_colors", ["#000000", "#000000"])
    ]
    if len(colors) < 2:
        colors = colors * 2
    stops = [float(value) for value in item.get("fill_stops", [0.0, 1.0])]
    if len(stops) != len(colors):
        denominator = max(1, len(colors) - 1)
        stops = [index / denominator for index in range(len(colors))]
    pairs = sorted(zip(stops, colors), key=lambda pair: pair[0])
    stops = [max(0.0, min(1.0, pair[0])) for pair in pairs]
    colors = [pair[1] for pair in pairs]

    if fill_type in {"solid", "scrim"}:
        return Image.new("RGB", size, colors[0])
    if fill_type == "radial_gradient":
        gradient = Image.radial_gradient("L").resize(size, Image.Resampling.BICUBIC)
    else:
        diagonal = max(2, int(round((width ** 2 + height ** 2) ** 0.5)))
        gradient = Image.linear_gradient("L").resize(
            (diagonal, diagonal), Image.Resampling.BICUBIC
        )
        angle = float(item.get("gradient_angle", 0.0))
        gradient = gradient.rotate(
            90.0 - angle,
            resample=Image.Resampling.BICUBIC,
            expand=False,
        )
        left = (diagonal - width) // 2
        top = (diagonal - height) // 2
        gradient = gradient.crop((left, top, left + width, top + height))

    channel_luts = [[], [], []]
    for value in range(256):
        ratio = value / 255.0
        upper = next(
            (index for index, stop in enumerate(stops) if stop >= ratio),
            len(stops) - 1,
        )
        lower = max(0, upper - 1)
        span = max(1e-9, stops[upper] - stops[lower])
        local_ratio = 0.0 if upper == lower else (ratio - stops[lower]) / span
        for channel in range(3):
            channel_luts[channel].append(round(
                colors[lower][channel]
                + (colors[upper][channel] - colors[lower][channel])
                * local_ratio
            ))
    return Image.merge(
        "RGB",
        tuple(gradient.point(lut) for lut in channel_luts),
    )


def _blend_surface(
    background: Image.Image,
    surface: Image.Image,
    mode: str,
) -> Image.Image:
    background = background.convert("RGB")
    surface = surface.convert("RGB")
    if mode == "multiply":
        return ImageChops.multiply(background, surface)
    if mode == "screen":
        return ImageChops.screen(background, surface)
    if mode == "overlay":
        overlay = getattr(ImageChops, "overlay", None)
        if overlay is not None:
            return overlay(background, surface)
    return surface


def _draw_underlays(
    image: Image.Image,
    underlays: list[dict],
) -> Image.Image:
    result = image.convert("RGBA")
    for item in sorted(
        underlays,
        key=lambda value: int(value.get("z_index", 0)),
    ):
        x = int(item["x"]); y = int(item["y"])
        width = int(item["width"]); height = int(item["height"])
        radius = max(0, int(item.get("border_radius", 0)))
        if width < 1 or height < 1:
            continue

        local_mask = Image.new("L", (width, height), 0)
        ImageDraw.Draw(local_mask).rounded_rectangle(
            (0, 0, width - 1, height - 1), radius=radius, fill=255
        )

        if item.get("shadow_enabled") and float(item.get("shadow_opacity", 0)) > 0:
            shadow_mask = Image.new("L", result.size, 0)
            offset_x = int(item.get("shadow_offset_x", 0))
            offset_y = int(item.get("shadow_offset_y", 0))
            shadow_mask.paste(local_mask, (x + offset_x, y + offset_y))
            blur = max(0, int(item.get("shadow_blur", 0)))
            if blur:
                shadow_mask = shadow_mask.filter(ImageFilter.GaussianBlur(blur))
            shadow_alpha = float(item.get("shadow_opacity", 0))
            shadow_mask = shadow_mask.point(
                lambda value: round(value * shadow_alpha)
            )
            shadow_rgb = ImageColor.getrgb(
                str(item.get("shadow_color", "#000000"))
            )[:3]
            shadow_layer = Image.new("RGBA", result.size, (*shadow_rgb, 0))
            shadow_layer.putalpha(shadow_mask)
            result = Image.alpha_composite(result, shadow_layer)

        backdrop_blur = max(0, int(item.get("backdrop_blur", 0)))
        if backdrop_blur:
            blurred = result.filter(ImageFilter.GaussianBlur(backdrop_blur))
            full_mask = Image.new("L", result.size, 0)
            full_mask.paste(local_mask, (x, y))
            result = Image.composite(blurred, result, full_mask)

        surface = _surface_gradient(item, (width, height))
        background = result.crop((x, y, x + width, y + height))
        surface = _blend_surface(
            background, surface, str(item.get("blend_mode", "normal"))
        )
        alpha = max(0.0, min(1.0, float(item.get("opacity", 0.65))))
        panel = surface.convert("RGBA")
        panel.putalpha(local_mask.point(lambda value: round(value * alpha)))
        result.alpha_composite(panel, (x, y))

        if (
            item.get("border_enabled")
            and int(item.get("border_width", 0)) > 0
            and float(item.get("border_opacity", 0)) > 0
        ):
            border_layer = Image.new("RGBA", result.size, (0, 0, 0, 0))
            border_rgb = ImageColor.getrgb(
                str(item.get("border_color", "#FFFFFF"))
            )[:3]
            border_alpha = round(
                255 * max(0.0, min(1.0, float(item["border_opacity"])))
            )
            ImageDraw.Draw(border_layer).rounded_rectangle(
                (x, y, x + width - 1, y + height - 1),
                radius=radius,
                outline=(*border_rgb, border_alpha),
                width=int(item["border_width"]),
            )
            result = Image.alpha_composite(result, border_layer)
    return result


def ensure_layout_contrast(
    image_path: str | Path,
    layout: dict,
) -> dict:
    """Return an idempotently contrast-corrected copy of a layout."""
    adjusted = deepcopy(layout)
    adjusted["underlays"] = list(adjusted.get("underlays", []))
    with Image.open(image_path) as source:
        working = source.convert("RGBA")
    working = _draw_underlays(working, adjusted["underlays"])

    for item in adjusted["elements"]:
        threshold = (
            3.0
            if item.get("role") in {"title", "price"}
            else 4.5
        )
        luminances = _region_luminances(working, item)
        requested = ImageColor.getrgb(
            str(item.get("color", "#FFFFFF"))
        )[:3]
        requested_score = _contrast_score(luminances, requested)
        item["color"] = "#{:02X}{:02X}{:02X}".format(*requested)
        if requested_score >= threshold:
            continue

        variants = []
        for target in ((255, 255, 255), (0, 0, 0)):
            for ratio in (0.18, 0.32, 0.48, 0.64, 0.80, 0.90):
                variants.append(tuple(
                    round(channel + (destination - channel) * ratio)
                    for channel, destination in zip(requested, target)
                ))
        passing = [
            color for color in variants
            if _contrast_score(luminances, color) >= threshold
        ]
        best_color = (
            min(
                passing,
                key=lambda color: sum(
                    abs(channel - original)
                    for channel, original in zip(color, requested)
                ),
            )
            if passing
            else max(
                variants,
                key=lambda color: _contrast_score(luminances, color),
            )
        )
        best_score = _contrast_score(luminances, best_color)
        item["color"] = "#{:02X}{:02X}{:02X}".format(*best_color)
        if best_score >= threshold:
            continue
        warning = (
            f"Contrast remains below target for {item['role']}; "
            "VLM surface choice preserved without an automatic box."
        )
        if warning not in adjusted.setdefault("warnings", []):
            adjusted["warnings"].append(warning)
    return adjusted


def _draw_text_run(
    draw: ImageDraw.ImageDraw,
    position: tuple[int, int],
    text: str,
    font,
    color: tuple[int, int, int],
    item: dict,
    *,
    anchor: str | None = None,
) -> None:
    tracking = max(0, int(item.get("tracking", 0)))
    shadow_offset = max(0, int(item.get("shadow_offset", 0)))
    shadow_color = ImageColor.getrgb(
        str(item.get("shadow_color", "#000000"))
    )[:3]
    stroke_width = max(0, int(item.get("stroke_width", 0)))
    stroke_color = ImageColor.getrgb(
        str(item.get("stroke_color", "#000000"))
    )[:3]

    def draw_at(x: int, y: int, fill, use_stroke: bool) -> None:
        if tracking == 0:
            draw.text(
                (x, y),
                text,
                font=font,
                fill=fill,
                stroke_width=stroke_width if use_stroke else 0,
                stroke_fill=(
                    (*stroke_color, 255)
                    if use_stroke and stroke_width
                    else None
                ),
                anchor=anchor,
            )
            return
        cursor = x
        for character in text:
            draw.text(
                (cursor, y),
                character,
                font=font,
                fill=fill,
                stroke_width=stroke_width if use_stroke else 0,
                stroke_fill=(
                    (*stroke_color, 255)
                    if use_stroke and stroke_width
                    else None
                ),
                anchor=anchor,
            )
            cursor += _text_width(draw, character, font) + tracking

    x, y = position
    if shadow_offset:
        draw_at(
            x + shadow_offset,
            y + shadow_offset,
            (*shadow_color, 150),
            False,
        )
    draw_at(x, y, (*color, 255), True)


def _draw_price_line(
    draw: ImageDraw.ImageDraw,
    line: str,
    cursor_y: int,
    step: int,
    item: dict,
    font_path: str | Path | None,
    color: tuple[int, int, int],
) -> bool:
    parts = [
        part
        for part in re.split(r"(\d[\d,.]*)", line)
        if part
    ]
    if not any(re.fullmatch(r"\d[\d,.]*", part) for part in parts):
        return False

    base_size = int(item.get("font_size", 24))
    number_scale = float(item.get("number_scale", 1.20))
    unit_scale = float(item.get("unit_scale", 0.84))
    resolved_font = _resolve_font_path(
        font_path,
        bold=True,
        text=line,
    )
    segments = []
    for part in parts:
        is_number = re.fullmatch(r"\d[\d,.]*", part) is not None
        size = round(
            base_size * (number_scale if is_number else unit_scale)
        )
        font = _load_font(
            max(8, size),
            resolved_font,
            max(700, int(item.get("font_weight", 700))),
        )
        segments.append((part, font, _text_width(draw, part, font)))

    total_width = sum(width for _part, _font, width in segments)
    box_width = int(item["width"])
    if total_width > box_width:
        shrink = box_width / max(1, total_width)
        resized_segments = []
        for part, font, _width in segments:
            resized_font = _load_font(
                max(8, int(getattr(font, "size", base_size) * shrink)),
                resolved_font,
                max(700, int(item.get("font_weight", 700))),
            )
            resized_segments.append(
                (part, resized_font, _text_width(draw, part, resized_font))
            )
        segments = resized_segments
        total_width = sum(
            width for _part, _font, width in segments
        )
        if total_width > box_width:
            return False
    align = str(item.get("text_align", "left"))
    if align == "right":
        cursor_x = int(item["x"]) + box_width - total_width
    elif align == "center":
        cursor_x = int(item["x"]) + (box_width - total_width) // 2
    else:
        cursor_x = int(item["x"])

    metrics = [
        (
            font.getmetrics()
            if hasattr(font, "getmetrics")
            else (int(getattr(font, "size", base_size)), 0)
        )
        for _part, font, _width in segments
    ]
    max_ascent = max(ascent for ascent, _descent in metrics)
    max_descent = max(descent for _ascent, descent in metrics)
    visual_height = max_ascent + max_descent
    common_baseline_y = (
        cursor_y + max(0, (step - visual_height) // 2) + max_ascent
    )

    for part, font, width in segments:
        is_number = re.fullmatch(r"\d[\d,.]*", part) is not None
        baseline_key = (
            "number_baseline_shift"
            if is_number
            else "unit_baseline_shift"
        )
        baseline_shift = round(
            base_size * float(item.get(baseline_key, 0.0))
        )
        segment_y = common_baseline_y + baseline_shift
        _draw_text_run(
            draw,
            (cursor_x, segment_y),
            part,
            font,
            color,
            item,
            anchor="ls",
        )
        cursor_x += width
    return True


def render_layout_image(
    image_path: str | Path,
    layout: dict,
    output_path: str | Path,
    *,
    font_path: str | Path | None = None,
) -> Path:
    """Render validated layout JSON directly with Pillow."""
    image_path = Path(image_path)
    layout = ensure_layout_contrast(image_path, layout)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(image_path) as source:
        image = source.convert("RGBA")

    canvas = layout["canvas"]
    expected_size = (
        int(canvas["width"]),
        int(canvas["height"]),
    )
    if image.size != expected_size:
        raise ValueError(
            f"Layout canvas {expected_size} does not match image {image.size}."
        )

    image = _draw_underlays(image, layout.get("underlays", []))
    draw = ImageDraw.Draw(image)
    for item in sorted(
        layout["elements"],
        key=lambda value: int(value.get("z_index", 1)),
    ):
        font, lines, step, total_height = _fit_text(
            draw,
            item,
            font_path,
        )
        x = int(item["x"])
        y = int(item["y"])
        width = int(item["width"])
        height = int(item["height"])
        vertical = str(item.get("vertical_align", "center"))
        if vertical == "bottom":
            cursor_y = y + height - total_height
        elif vertical == "top":
            cursor_y = y
        else:
            cursor_y = y + (height - total_height) // 2

        color = ImageColor.getrgb(str(item.get("color", "#FFFFFF")))
        align = str(item.get("text_align", "left"))
        tracking = max(0, int(item.get("tracking", 0)))
        for line in lines:
            if (
                item.get("role") == "price"
                and _draw_price_line(
                    draw,
                    line,
                    cursor_y,
                    step,
                    item,
                    font_path,
                    color[:3],
                )
            ):
                cursor_y += step
                continue
            line_width = _text_width(draw, line, font, tracking)
            if align == "right":
                cursor_x = x + width - line_width
            elif align == "center":
                cursor_x = x + (width - line_width) // 2
            else:
                cursor_x = x
            _draw_text_run(
                draw,
                (cursor_x, cursor_y),
                line,
                font,
                color[:3],
                item,
            )
            cursor_y += step

    image.convert("RGB").save(output_path)
    return output_path
