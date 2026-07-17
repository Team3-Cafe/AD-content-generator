from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

from PIL import Image, ImageColor, ImageDraw, ImageFont


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
):
    if resolved_font is not None:
        return ImageFont.truetype(str(resolved_font), size=size)
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _text_width(draw: ImageDraw.ImageDraw, text: str, font) -> int:
    if not text:
        return 0
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0]


def _wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font,
    max_width: int,
) -> list[str]:
    lines = []
    for paragraph in text.splitlines() or [""]:
        if not paragraph:
            lines.append("")
            continue
        line = ""
        for character in paragraph:
            candidate = line + character
            if line and _text_width(draw, candidate, font) > max_width:
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
        font = _load_font(size, resolved_font)
        lines = _wrap_text(draw, text, font, max_width)
        step = max(1, int(round(size * line_height)))
        total_height = step * len(lines)
        selected = (font, lines, step, total_height)
        if len(lines) <= max_lines and total_height <= max_height:
            break
    return selected


def _draw_underlays(
    image: Image.Image,
    underlays: list[dict],
) -> Image.Image:
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for item in sorted(
        underlays,
        key=lambda value: int(value.get("z_index", 0)),
    ):
        rgb = ImageColor.getrgb(
            str(item.get("background_color", "#000000"))
        )
        alpha = int(
            round(255 * float(item.get("opacity", 0.65)))
        )
        x = int(item["x"])
        y = int(item["y"])
        box = (
            x,
            y,
            x + int(item["width"]),
            y + int(item["height"]),
        )
        draw.rounded_rectangle(
            box,
            radius=int(item.get("border_radius", 0)),
            fill=(*rgb[:3], alpha),
        )
    return Image.alpha_composite(image, overlay)


def render_layout_image(
    image_path: str | Path,
    layout: dict,
    output_path: str | Path,
    *,
    font_path: str | Path | None = None,
) -> Path:
    """Render validated layout JSON onto the completed background image."""
    image_path = Path(image_path)
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
        for line in lines:
            line_width = _text_width(draw, line, font)
            if align == "right":
                cursor_x = x + width - line_width
            elif align == "center":
                cursor_x = x + (width - line_width) // 2
            else:
                cursor_x = x
            draw.text(
                (cursor_x, cursor_y),
                line,
                font=font,
                fill=(*color[:3], 255),
            )
            cursor_y += step

    image.convert("RGB").save(output_path)
    return output_path
