from __future__ import annotations

import base64
import html
from pathlib import Path
import re

from PIL import Image

from .io import image_to_data_url


class HtmlRendererUnavailable(RuntimeError):
    """Raised when Playwright or its Chromium runtime is unavailable."""


def _launch_chromium(playwright, playwright_error):
    """Translate browser startup failures into the supported fallback signal."""
    try:
        return playwright.chromium.launch(headless=True)
    except playwright_error as error:
        raise HtmlRendererUnavailable(
            "Playwright Chromium could not start. On Linux run "
            "'uv run playwright install --with-deps chromium' to install "
            "the browser and required shared libraries"
        ) from error


def _css_color_stops(item: dict) -> str:
    colors = list(item.get("fill_colors", ["#000000", "#000000"]))
    stops = list(item.get("fill_stops", [0.0, 1.0]))
    if len(colors) < 2:
        colors *= 2
    if len(stops) != len(colors):
        stops = [index / max(1, len(colors) - 1) for index in range(len(colors))]
    return ", ".join(
        f"{color} {max(0.0, min(1.0, float(stop))) * 100:.2f}%"
        for color, stop in zip(colors, stops)
    )


def _surface_background(item: dict) -> str:
    stops = _css_color_stops(item)
    fill_type = str(item.get("fill_type", "solid"))
    if fill_type == "linear_gradient":
        return f"linear-gradient({float(item.get('gradient_angle', 0)):.2f}deg, {stops})"
    if fill_type == "radial_gradient":
        return f"radial-gradient(circle at center, {stops})"
    if fill_type == "scrim":
        return f"linear-gradient(90deg, {stops})"
    return str((item.get("fill_colors") or ["#000000"])[0])


def _font_data_rule(font_path: Path | None) -> str:
    if font_path is None or not font_path.is_file():
        return ""
    mime = {
        ".woff2": "font/woff2", ".woff": "font/woff",
        ".otf": "font/otf", ".ttf": "font/ttf", ".ttc": "font/collection",
    }.get(font_path.suffix.lower(), "font/ttf")
    encoded = base64.b64encode(font_path.read_bytes()).decode("ascii")
    return (
        "@font-face{font-family:'ADCG Korean';"
        f"src:url(data:{mime};base64,{encoded});font-weight:100 900;"
        "font-style:normal;font-display:block;}"
    )


def _surface_html(item: dict) -> str:
    border = (
        f"{int(item.get('border_width', 0))}px solid "
        f"color-mix(in srgb, {item.get('border_color', '#000000')} "
        f"{float(item.get('border_opacity', 0)) * 100:.1f}%, transparent)"
        if item.get("border_enabled")
        else "none"
    )
    shadow = (
        f"{int(item.get('shadow_offset_x', 0))}px "
        f"{int(item.get('shadow_offset_y', 0))}px "
        f"{int(item.get('shadow_blur', 0))}px "
        f"color-mix(in srgb, {item.get('shadow_color', '#000000')} "
        f"{float(item.get('shadow_opacity', 0)) * 100:.1f}%, transparent)"
        if item.get("shadow_enabled")
        else "none"
    )
    style = (
        f"left:{int(item['x'])}px;top:{int(item['y'])}px;"
        f"width:{int(item['width'])}px;height:{int(item['height'])}px;"
        f"z-index:{int(item.get('z_index', 0))};"
        f"background:{_surface_background(item)};"
        f"opacity:{float(item.get('opacity', 1))};"
        f"border-radius:{int(item.get('border_radius', 0))}px;"
        f"backdrop-filter:blur({int(item.get('backdrop_blur', 0))}px);"
        f"-webkit-backdrop-filter:blur({int(item.get('backdrop_blur', 0))}px);"
        f"mix-blend-mode:{html.escape(str(item.get('blend_mode', 'normal')))};"
        f"border:{border};box-shadow:{shadow};"
    )
    return f'<div class="surface" style="{style}"></div>'


def _accent_html(item: dict) -> str:
    color = str((item.get("fill_colors") or ["#000000"])[0])
    style = (
        f"left:{int(item['x'])}px;top:{int(item['y'])}px;"
        f"width:{int(item['width'])}px;height:{int(item['height'])}px;"
        f"z-index:{int(item.get('z_index', 1))};background:{color};"
        f"opacity:{float(item.get('opacity', 1))};"
        f"border-radius:{int(item.get('border_radius', 0))}px;"
    )
    return f'<div class="accent" style="{style}"></div>'


def _price_inner(item: dict) -> str:
    parts = [part for part in re.split(r"(\d[\d,.]*)", str(item["content"])) if part]
    spans = []
    base_size = int(item.get("font_size", 24))
    for part in parts:
        is_number = re.fullmatch(r"\d[\d,.]*", part) is not None
        scale = float(item.get("number_scale" if is_number else "unit_scale", 1.0))
        shift = float(item.get(
            "number_baseline_shift" if is_number else "unit_baseline_shift", 0.0
        ))
        css_class = "price-number" if is_number else "price-unit"
        spans.append(
            f'<span class="{css_class}" style="font-size:calc(var(--base-size) * {scale});'
            f'transform:translateY({round(base_size * shift, 2)}px)">'
            f'{html.escape(part)}</span>'
        )
    return '<span class="price-inner">' + "".join(spans) + "</span>"


def _text_html(item: dict) -> str:
    role = str(item.get("role", "copy"))
    align = str(item.get("text_align", "left"))
    vertical = str(item.get("vertical_align", "center"))
    justify = {"left": "flex-start", "center": "center", "right": "flex-end"}[align]
    align_items = {"top": "flex-start", "center": "center", "bottom": "flex-end"}[vertical]
    shadow_offset = int(item.get("shadow_offset", 0))
    shadow = (
        f"{shadow_offset}px {shadow_offset}px 0 {item.get('shadow_color', '#000000')}"
        if shadow_offset else "none"
    )
    nowrap = role in {"title", "price", "cta"}
    style = (
        f"left:{int(item['x'])}px;top:{int(item['y'])}px;"
        f"width:{int(item['width'])}px;height:{int(item['height'])}px;"
        f"z-index:{int(item.get('z_index', 2))};"
        f"--base-size:{int(item.get('font_size', 24))}px;"
        f"font-size:var(--base-size);font-weight:{int(item.get('font_weight', 600))};"
        f"letter-spacing:{int(item.get('tracking', 0))}px;"
        f"line-height:{float(item.get('line_height', 1.2))};"
        f"color:{item.get('color', '#FFFFFF')};text-align:{align};"
        f"justify-content:{justify};align-items:{align_items};"
        f"white-space:{'nowrap' if nowrap else 'normal'};"
        f"text-shadow:{shadow};"
        f"-webkit-text-stroke:{int(item.get('stroke_width', 0))}px "
        f"{item.get('stroke_color', '#000000')};"
    )
    inner = (
        _price_inner(item)
        if role == "price"
        else f'<span class="copy-inner">{html.escape(str(item["content"]))}</span>'
    )
    return f'<div class="copy copy-{html.escape(role)}" data-role="{html.escape(role)}" style="{style}">{inner}</div>'


def _build_ad_html(
    image_path: str | Path,
    layout: dict,
    font_path: Path | None,
) -> str:
    canvas = layout["canvas"]
    width, height = int(canvas["width"]), int(canvas["height"])
    layers = []
    for item in sorted(layout.get("underlays", []), key=lambda value: int(value.get("z_index", 0))):
        layers.append(
            _accent_html(item) if item.get("id") == "accent-rule" else _surface_html(item)
        )
    layers.extend(
        _text_html(item)
        for item in sorted(layout["elements"], key=lambda value: int(value.get("z_index", 2)))
    )
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><style>
{_font_data_rule(font_path)}
*{{box-sizing:border-box}}html,body{{margin:0;width:{width}px;height:{height}px;overflow:hidden}}
body{{background:#000;font-family:'ADCG Korean','Noto Sans KR','Noto Sans CJK KR','Malgun Gothic','Apple SD Gothic Neo',sans-serif}}
#ad{{position:relative;width:{width}px;height:{height}px;overflow:hidden;background-image:url('{image_to_data_url(image_path)}');background-size:100% 100%;background-repeat:no-repeat}}
.surface,.accent,.copy{{position:absolute}}.copy{{display:flex;overflow:hidden}}
.copy-inner{{display:block;width:100%;overflow-wrap:break-word;word-break:keep-all}}
.price-inner{{display:inline-flex;align-items:baseline;white-space:nowrap}}
.price-number,.price-unit{{display:inline-block;line-height:1}}
</style></head><body><div id="ad">{''.join(layers)}</div>
<script>
document.fonts.ready.then(()=>{{
  for(const node of document.querySelectorAll('.copy')){{
    let size=parseFloat(getComputedStyle(node).getPropertyValue('--base-size'));
    let guard=0;
    while((node.scrollWidth>node.clientWidth+1 || node.scrollHeight>node.clientHeight+1) && size>8 && guard<256){{
      size-=1; node.style.setProperty('--base-size',size+'px'); guard+=1;
    }}
  }}
  document.documentElement.dataset.fontsReady='true';
}});
</script></body></html>"""


def render_layout_image_html(
    image_path: str | Path,
    layout: dict,
    output_path: str | Path,
    *,
    font_path: str | Path | None = None,
) -> Path:
    try:
        from playwright.sync_api import Error as PlaywrightError
        from playwright.sync_api import sync_playwright
    except ImportError as error:
        raise HtmlRendererUnavailable("Python package 'playwright' is not installed") from error

    from .renderer import _resolve_font_path

    image_path = Path(image_path)
    with Image.open(image_path) as source:
        actual_size = source.size
    expected_size = (
        int(layout["canvas"]["width"]),
        int(layout["canvas"]["height"]),
    )
    if actual_size != expected_size:
        raise ValueError(
            f"Layout canvas {expected_size} does not match image {actual_size}."
        )
    all_text = " ".join(str(item.get("content", "")) for item in layout["elements"])
    resolved_font = _resolve_font_path(font_path, bold=False, text=all_text)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    document = _build_ad_html(image_path, layout, resolved_font)
    width = int(layout["canvas"]["width"])
    height = int(layout["canvas"]["height"])
    with sync_playwright() as playwright:
        browser = _launch_chromium(playwright, PlaywrightError)
        page = browser.new_page(
            viewport={"width": width, "height": height},
            device_scale_factor=1,
        )
        page.set_content(document, wait_until="load")
        page.wait_for_function(
            "document.documentElement.dataset.fontsReady === 'true'",
            timeout=10_000,
        )
        page.locator("#ad").screenshot(path=str(output_path), type="png")
        browser.close()
    return output_path


__all__ = ["HtmlRendererUnavailable", "render_layout_image_html"]
