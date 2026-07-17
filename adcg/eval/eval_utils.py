from __future__ import annotations

import json
from pathlib import Path

from PIL import Image


def text_value(value) -> str:
    if isinstance(value, list):
        return ". ".join(
            str(item).strip() for item in value if str(item).strip()
        )
    return str(value or "").strip()


def load_background_prompt(prompt_json: str | Path) -> str:
    """Read both the current pipeline schema and the legacy eval schema."""
    prompt_json = Path(prompt_json)
    data = json.loads(prompt_json.read_text(encoding="utf-8"))

    if isinstance(data, dict):
        generation_prompt = data.get("generation_prompt") or {}
        prompt = (
            generation_prompt.get("background_prompt")
            or data.get("background_prompt")
            or data.get("answer")
        )
        prompt = text_value(prompt)
        if prompt:
            return prompt

    if isinstance(data, list):
        prompts = [
            text_value(item.get("answer") or item.get("background_prompt"))
            for item in data
            if isinstance(item, dict)
        ]
        prompt = ". ".join(value for value in prompts if value)
        if prompt:
            return prompt

    raise ValueError(f"background prompt not found in {prompt_json}")


def load_product_mask(
    product_mask: str | Path,
    size: tuple[int, int],
) -> Image.Image:
    product_mask = Path(product_mask)
    if not product_mask.is_file():
        raise FileNotFoundError(product_mask)
    return Image.open(product_mask).convert("L").resize(
        size,
        Image.Resampling.BILINEAR,
    )


def mask_product_region(
    image_path: str | Path,
    product_mask: str | Path,
    fill_value: int = 127,
) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    mask = load_product_mask(product_mask, image.size)
    fill = Image.new("RGB", image.size, (fill_value,) * 3)
    return Image.composite(fill, image, mask)


def crop_product_pair(
    generated_image: str | Path,
    reference_image: str | Path,
    product_mask: str | Path,
    *,
    alpha_threshold: int = 128,
    padding_ratio: float = 0.05,
    fill_value: int = 255,
) -> tuple[Image.Image, Image.Image]:
    """Crop generated and reference product content for DINO comparison."""
    generated = Image.open(generated_image).convert("RGB")
    mask = load_product_mask(product_mask, generated.size)
    binary = mask.point(lambda value: 255 if value >= alpha_threshold else 0)
    bbox = binary.getbbox()
    if bbox is None:
        raise ValueError(f"product mask has no visible region: {product_mask}")

    left, top, right, bottom = bbox
    padding = round(max(right - left, bottom - top) * padding_ratio)
    bbox = (
        max(0, left - padding),
        max(0, top - padding),
        min(generated.width, right + padding),
        min(generated.height, bottom + padding),
    )
    generated_crop = generated.crop(bbox)

    reference = Image.open(reference_image).convert("RGBA")
    reference_bbox = reference.getchannel("A").getbbox()
    if reference_bbox is not None:
        reference = reference.crop(reference_bbox)
    fill = Image.new("RGBA", reference.size, (fill_value,) * 3 + (255,))
    reference_crop = Image.alpha_composite(fill, reference).convert("RGB")
    return generated_crop, reference_crop
