# Prompt layout

This package creates one bespoke advertisement design for a completed
background image. It does not generate layout candidates, select templates,
rank alternatives, or measure aesthetic scores.

## Workflow

1. Code describes the image's canvas, palette, luminance, edge density, and
   quiet regions. OpenCV LAB clustering expands dominant colors into swatches,
   accents, tonal variants, and harmony sets. These are diagnostics and design
   affordances rather than an aesthetic metric.
2. GPT-4o studies the actual image and copy roles, then authors one structured
   art direction. Title/subtitle form a headline group and price/CTA form an
   offer group. It also selects semantic color tokens from the extracted image
   palette for each band, text group, and CTA.
3. A relational layout engine converts independent headline/offer x/y positions,
   widths, and alignments into responsive pixel geometry. Each surface may be
   absent, content-width, or full-width. A shared effect model lets both the first
   and final VLM compose solid, multi-stop linear/radial gradient, scrim, exact
   colors, angle, opacity, radius, backdrop blur, blend mode, border, and shadow.
   Centering and surface styling are choices rather than fixed templates.
4. Price and CTA keep independent alignment and placement. CTA remains optional
   plain typography rather than a button, pill, outline, or interactive UI.
5. Selected palette colors are resolved to exact values and checked for WCAG
   text contrast. Unsafe foreground colors are replaced automatically while
   preserving the VLM's background-color direction.
6. The first design is rendered to `design_draft.png`.
7. A second GPT-4o call sees `design_draft.png` and its exact resolved state, then
   refines all design features into a coherent second-stage design.
8. That revision is fitted and rendered to `final_review_input.png`.
9. The third GPT-4o call receives no earlier art direction, resolved element boxes,
   or revision JSON. It reads the completed second-stage pixels, audits all eleven
   categories, receives broad image/copy/renderer-aware candidate pools for all
   categories, and independently rebuilds the copy layer as one complete
   absolute-pixel target. Candidate pools are neither templates nor limits; the
   VLM may combine, alter, reject, or exceed them.
   For every category it records visible evidence, a design objective, candidate
   benefits and risks, cross-feature dependencies, its selected direction, and
   concrete target-layout commitments. A final coherence review resolves conflicts
   between those decisions before rendering.
10. The final redesign must materially change copy geometry and multiple design
    systems. Requested/applied states, constraints,
    warnings, material-change summaries, and exact property changes are saved in
    `final_review.json`, `design_revision.json`, and `layout.json`. The copy remains
    fixed.


The primary renderer converts the resolved layout to an HTML/CSS scene and uses
headless Chromium through Playwright to capture the final PNG. Browser typography
provides native baseline layout, wrapping, font fallback, flex alignment, gradients,
backdrop blur, blend modes, borders, shadows, strokes, and layered surfaces. The
existing Pillow renderer remains an automatic fallback only when Playwright or its
Chromium runtime is unavailable. Both paths adapt text colors to local background
contrast without inventing an unrequested surface.

## Outputs

- `design_analysis.json`: computed image-space diagnostics and VLM scene notes
- `design_spec.json`: the single art direction and relational composition
- `design_draft.png`: first rendering of that design
- `design_revision.json`: second-stage state-based design refinement
- `final_review_input.png`: completed second-stage design supplied to final redesign
- `final_review.json`: final independent redesign exploration, diagnosis, absolute target, actual
  post-fit state, applied constraints, and changed properties
- `layout.json`: final resolved pixels, typography, colors, and surfaces
- `final_ad.png`: the completed advertisement image

## Usage

Run independently after background generation:

    python -m adcg.prompt_layout \
      --image outputs/lift_truck_01/05_final/final_identity_restored.png \
      --ad-copy outputs/lift_truck_01/02_prompt/ad_copy.json \
      --output-dir outputs/lift_truck_01/07_prompt_layout \
      --font adcg/assets/fonts/NotoSansKR.ttf

The default model is `gpt-4o`, image detail is `high`, and design temperature
is `0.4`. Exactly three OpenAI calls are made: one art-direction call, one
draft revision call, and one final review of the completed advertisement.
`OPENAI_API_KEY` is loaded from the project-root `.env` when present.

The Pillow renderer runs without a browser runtime or additional system package.
The repository includes `adcg/assets/fonts/NotoSansKR.ttf` under the SIL Open Font
License, and this bundled font is the deterministic default on every machine.
The renderer can also discover Korean-capable system fonts, including `Noto Sans KR`,
`Noto Sans CJK KR`, `Malgun Gothic`, and `Apple SD Gothic Neo`.
The `--font`/`--layout-font` options and `ADCG_FONT_PATH` remain optional overrides for
deterministic branding; they are not required when a suitable system font is
installed. The `--font`/`--layout-font` options and `ADCG_FONT_PATH` can still
override the bundled default. If `ADCG_FONT_PATH` points to a missing file, the
renderer emits a warning and safely falls back to the bundled font.
The renderer supports rounded, pill, ellipse, cut-corner, and diagonal surfaces;
image-derived color overlays; up to three surface shadow layers; character, word,
and balanced wrapping; optical text anchoring; automatic same-group collision
correction; and shared, cap-height, or optical-center price baselines. These
choices are exposed to the second and final VLM design stages and persisted in
the resolved layout JSON.
