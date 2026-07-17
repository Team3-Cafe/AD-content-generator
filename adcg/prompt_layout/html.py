from __future__ import annotations

import html
from pathlib import Path

from .io import image_to_data_url


def _css_color(value: object, fallback: str) -> str:
    text = str(value or "").strip()
    allowed = set("#(),.% -_abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    return text if text and set(text) <= allowed else fallback


def render_layout_html(
    image_path: str | Path,
    layout: dict,
    output_path: str | Path,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas = layout["canvas"]
    width = int(canvas["width"])
    height = int(canvas["height"])

    layers = []
    for item in layout.get("underlays", []):
        layers.append(
            (
                int(item["z_index"]),
                (
                    '<div class="underlay" style="'
                    f'left:{int(item["x"])}px;top:{int(item["y"])}px;'
                    f'width:{int(item["width"])}px;'
                    f'height:{int(item["height"])}px;'
                    f'z-index:{int(item["z_index"])};'
                    f'background:{_css_color(item.get("background_color"), "#000000")};'
                    f'opacity:{float(item.get("opacity", 0.65))};'
                    f'border-radius:{int(item.get("border_radius", 0))}px'
                    '"></div>'
                ),
            )
        )

    for item in layout["elements"]:
        align = str(item.get("text_align", "left"))
        vertical = {
            "top": "flex-start",
            "center": "center",
            "bottom": "flex-end",
        }.get(str(item.get("vertical_align")), "center")
        content = html.escape(str(item["content"]))
        layers.append(
            (
                int(item["z_index"]),
                (
                    f'<div class="copy role-{html.escape(str(item["role"]))}" '
                    'style="'
                    f'left:{int(item["x"])}px;top:{int(item["y"])}px;'
                    f'width:{int(item["width"])}px;'
                    f'height:{int(item["height"])}px;'
                    f'z-index:{int(item["z_index"])};'
                    f'color:{_css_color(item.get("color"), "#FFFFFF")};'
                    f'font-size:{int(item.get("font_size", 24))}px;'
                    f'font-weight:{int(item.get("font_weight", 600))};'
                    f'line-height:{float(item.get("line_height", 1.2))};'
                    f'text-align:{align};align-items:{vertical};'
                    f'-webkit-line-clamp:{int(item.get("max_lines", 2))}'
                    f'">{content}</div>'
                ),
            )
        )

    layer_html = "\n".join(value for _, value in sorted(layers))
    document = f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Advertisement layout preview</title>
<style>
html, body {{ margin: 0; min-height: 100%; background: #202124; }}
.canvas {{
  position: relative;
  width: {width}px;
  height: {height}px;
  margin: 24px auto;
  overflow: hidden;
  background-image: url("{image_to_data_url(image_path)}");
  background-size: 100% 100%;
  box-shadow: 0 8px 32px rgba(0, 0, 0, .35);
}}
.underlay {{ position: absolute; box-sizing: border-box; }}
.copy {{
  position: absolute;
  box-sizing: border-box;
  display: -webkit-box;
  display: flex;
  flex-direction: column;
  justify-content: center;
  overflow: hidden;
  overflow-wrap: anywhere;
  font-family: Pretendard, "Noto Sans KR", Arial, sans-serif;
  text-shadow: 0 1px 2px rgba(0, 0, 0, .18);
}}
</style>
</head>
<body>
<div class="canvas">
{layer_html}
</div>
</body>
</html>
"""
    output_path.write_text(document, encoding="utf-8")
    return output_path
