# Prompt layout

This package creates one bespoke advertisement design for a completed
background image. It does not generate layout candidates, select templates,
rank alternatives, or measure aesthetic scores.

## Workflow

1. Code describes the image's canvas, palette, luminance, edge density, and
   quiet regions. These are spatial diagnostics rather than an aesthetic
   metric.
2. GPT-4o studies the actual image and copy roles, then authors one structured
   art direction. Title/subtitle form a headline group and price/CTA form an
   offer group. It also selects semantic color tokens from the extracted image
   palette for each band, text group, and CTA.
3. A relational layout engine converts independent headline/offer x/y positions,
   widths, and alignments into responsive pixel geometry. Each surface may be
   absent, content-width, or full-width, and colors may use image-palette tokens
   or exact hex values. Centering is an option rather than a fixed template.
4. Price and CTA keep independent alignment and placement. CTA remains optional
   plain typography rather than a button, pill, outline, or interactive UI.
5. Selected palette colors are resolved to exact values and checked for WCAG
   text contrast. Unsafe foreground colors are replaced automatically while
   preserving the VLM's background-color direction.
6. The first design is rendered to `design_draft.png`.
7. GPT-4o reviews that same render once and returns bounded position, scale,
   and surface-opacity corrections. It never compares or selects candidates.
8. The corrected design, including all copy, is rendered to
   `final_review_input.png`.
9. GPT-4o receives no earlier art direction, revision JSON, resolved element
   boxes, or surface values. It reads the completed advertisement pixels as the
   only previous-design reference, records distinct strengths and weaknesses
   across all eleven design categories (up to 24 observations), then creates an
   independent art direction and rebuilds all copy, bands, color, accent, and
   price construction as one complete absolute-pixel target. The same VLM call
   receives the completed ad for diagnosis and the clean background as its new
   canvas; optional surfaces and exact colors are rebuilt without inheriting the
   previous overlay object.
10. The rebuilt state must materially change copy geometry plus multiple other
    design systems. Feature claims are checked against actual target properties
    before and after typography fitting. Canvas constraints, requested/applied
    states, material-change summaries, and exact property changes are saved to
    `final_review.json` before `final_ad.png` is rendered. The copy remains fixed.

The renderer uses the configured Korean-capable font, fits the title to one
line, adapts text colors to local background contrast, and adds a contrast
underlay only when neither light nor dark text is sufficiently readable.

## Outputs

- `design_analysis.json`: computed image-space diagnostics and VLM scene notes
- `design_spec.json`: the single art direction and relational composition
- `design_draft.png`: first rendering of that design
- `design_revision.json`: one bounded critique of the same design
- `final_review_input.png`: completed advertisement supplied to the final VLM review
- `final_review.json`: final diagnosis, requested absolute target, actual
  post-fit state, applied constraints, and changed properties
- `layout.json`: final resolved pixels, typography, colors, and surfaces
- `final_ad.png`: the completed advertisement image

## Usage

Run independently after background generation:

    python -m adcg.prompt_layout \
      --image outputs/lift_truck_01/05_final/final_identity_restored.png \
      --ad-copy outputs/lift_truck_01/02_prompt/ad_copy.json \
      --output-dir outputs/lift_truck_01/07_prompt_layout \
      --font assets/fonts/NotoSansKR.ttf

The default model is `gpt-4o`, image detail is `high`, and design temperature
is `0.4`. Exactly three OpenAI calls are made: one art-direction call, one
draft revision call, and one final review of the completed advertisement.
`OPENAI_API_KEY` is loaded from the project-root `.env` when present.

No new font is downloaded. For Korean copy, pass `--font`/`--layout-font` or
set `ADCG_FONT_PATH` when the project font is not discoverable automatically.
