"""DINOv2 product identity similarity evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from .eval_result_store import update_eval_results
from .eval_utils import crop_product_pair


DEFAULT_MODEL = "facebook/dinov2-base"


def dino_embeddings(model, processor, images, device):
    inputs = processor(images=images, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        return F.normalize(outputs.last_hidden_state[:, 0], dim=-1)


def evaluate_dino(
    image_path,
    reference_image,
    product_mask,
    output_json,
    *,
    model_name=DEFAULT_MODEL,
    alpha_threshold=128,
    padding_ratio=0.05,
    background_fill_value=255,
    device=None,
):
    """Compare the generated product region with the source product image."""
    image_path = Path(image_path)
    reference_image = Path(reference_image)
    for path in (image_path, reference_image, Path(product_mask)):
        if not path.is_file():
            raise FileNotFoundError(path)

    generated_crop, reference_crop = crop_product_pair(
        image_path,
        reference_image,
        product_mask,
        alpha_threshold=alpha_threshold,
        padding_ratio=padding_ratio,
        fill_value=background_fill_value,
    )
    try:
        from transformers import AutoImageProcessor, AutoModel
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required for DINO evaluation"
        ) from exc

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    embeddings = dino_embeddings(
        model, processor, [generated_crop, reference_crop], device
    )
    score = float((embeddings[0] * embeddings[1]).sum().cpu().item())
    results = [{"image_id": image_path.name, "dino_similarity": score}]
    update_eval_results(output_json, results, "dino_similarity")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--reference-image", required=True)
    parser.add_argument("--product-mask", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--alpha-threshold", type=int, default=128)
    parser.add_argument("--padding-ratio", type=float, default=0.05)
    parser.add_argument("--background-fill-value", type=int, default=255)
    args = parser.parse_args()

    results = evaluate_dino(
        args.image,
        args.reference_image,
        args.product_mask,
        args.output_json,
        model_name=args.model,
        alpha_threshold=args.alpha_threshold,
        padding_ratio=args.padding_ratio,
        background_fill_value=args.background_fill_value,
    )
    print(f"{results[0]['image_id']}: {results[0]['dino_similarity']:.4f}")


if __name__ == "__main__":
    main()