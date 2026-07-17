"""Background CLIP similarity evaluation using the pipeline product mask."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .eval_result_store import update_eval_results
from .eval_utils import load_background_prompt, mask_product_region


DEFAULT_MODEL = "openai/clip-vit-base-patch32"


def build_background_eval_text(background_prompt, store_info=""):
    parts = [
        str(store_info).strip(),
        f"Background prompt: {background_prompt}",
        "advertising background, visually appealing, no text",
    ]
    return ". ".join(part for part in parts if part)


def clip_score(model, processor, image, text, device):
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        return float(torch.matmul(image_embeds, text_embeds.T).item())


def evaluate_clip(
    image_path,
    prompt_json,
    product_mask,
    output_json,
    *,
    masked_image_path=None,
    store_info="",
    fill_value=127,
    model_name=DEFAULT_MODEL,
    device=None,
):
    """Score generated background against the current pipeline prompt JSON."""
    image_path = Path(image_path)
    for path in (image_path, Path(prompt_json), Path(product_mask)):
        if not path.is_file():
            raise FileNotFoundError(path)

    background_prompt = load_background_prompt(prompt_json)
    eval_text = build_background_eval_text(background_prompt, store_info)
    background = mask_product_region(image_path, product_mask, fill_value)
    if masked_image_path:
        masked_image_path = Path(masked_image_path)
        masked_image_path.parent.mkdir(parents=True, exist_ok=True)
        background.save(masked_image_path)

    try:
        from transformers import CLIPModel, CLIPProcessor
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required for CLIP evaluation"
        ) from exc

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_name)
    score = clip_score(model, processor, background, eval_text, device)
    results = [{"image_id": image_path.name, "clip_score": score}]
    update_eval_results(output_json, results, "clip_score")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt-json", required=True)
    parser.add_argument("--product-mask", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--masked-image")
    parser.add_argument("--store-info", default="")
    parser.add_argument("--fill-value", type=int, default=127)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    results = evaluate_clip(
        args.image,
        args.prompt_json,
        args.product_mask,
        args.output_json,
        masked_image_path=args.masked_image,
        store_info=args.store_info,
        fill_value=args.fill_value,
        model_name=args.model,
    )
    print(f"{results[0]['image_id']}: {results[0]['clip_score']:.4f}")


if __name__ == "__main__":
    main()