# Prompt layout

This package independently finds content-aware positions for the text in
ad_copy.json. It is intentionally not connected to adcg.pipeline.

The implementation follows the paper's two-stage VLM workflow with five
self-contained multimodal design demonstrations:

1. GPT-4o studies five visual input/output layout examples.
2. GPT-4o analyzes the completed background and writes a placement plan.
3. GPT-4o converts that plan into exact pixel boxes and role-aware typography.

The five demonstrations are scaled at runtime to the completed image's actual
width, height, and aspect ratio. Their boxes and font sizes are not fixed to a
512-by-512 canvas.

The command writes:

- placement_plan.json: semantic regions and the placement rationale
- layout.json: validated pixel boxes, typography, and optional underlays
- layout_preview.html: a browser preview using the real copy strings
- final_ad.png: the copy rendered onto the completed background

The renderer keeps titles on one line by fitting their font size to the
selected box. It also checks local image contrast, preserves readable accent
colors, switches between light and dark text when needed, and adds a
role-specific translucent underlay when neither color is reliably readable.

## Usage

Run after the image pipeline has created the final background:

    python -m adcg.prompt_layout \
      --image outputs/lift_truck_01/05_final/final_identity_restored.png \
      --ad-copy outputs/lift_truck_01/02_prompt/ad_copy.json \
      --output-dir outputs/lift_truck_01/prompt_layout

The default model is gpt-4o, image detail is high, and temperature is 0.7,
matching the paper's reported sampling temperature. The command makes two
OpenAI API calls. OPENAI_API_KEY is loaded from the project-root .env when
present.

If ad_copy.json contains multiple entries, select one with:

    --copy-index 0

Use a specific Korean or brand font with:

    --font /path/to/font.ttf

Empty fields such as an unavailable price are omitted from the requested
layout. The generated HTML is a preview artifact; no pixels are written back
to the input image. The rendered advertisement is written to final_ad.png.

For Korean text, the renderer searches for Malgun Gothic, Noto Sans CJK/KR,
Nanum Gothic, and Korean fonts reported by fontconfig. It does not silently
fall back to a Latin-only font for Korean copy. Set ADCG_FONT_PATH or pass
--font/--layout-font when a specific Korean or brand font must be used.
