"""LAION aesthetic quality evaluation for generated images."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torch import nn

from .eval_result_store import update_eval_results


DEFAULT_MODEL = "openai/clip-vit-large-patch14"
DEFAULT_WEIGHTS_URL = (
    "https://github.com/LAION-AI/aesthetic-predictor/raw/main/"
    "sa_0_4_vit_l_14_linear.pth"
)


def load_aesthetic_head(weights_path, weights_url, embedding_dim, device):
    head = nn.Linear(embedding_dim, 1)
    if weights_path:
        state = torch.load(weights_path, map_location="cpu")
    else:
        state = torch.hub.load_state_dict_from_url(
            weights_url, map_location="cpu", progress=True
        )
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    cleaned_state = {}
    for key, value in state.items():
        key = key.removeprefix("module.").removeprefix("linear.")
        if key in {"layers.0.weight", "0.weight"}:
            key = "weight"
        elif key in {"layers.0.bias", "0.bias"}:
            key = "bias"
        cleaned_state[key] = value
    head.load_state_dict(cleaned_state, strict=False)
    return head.to(device).eval()


def aesthetic_scores(model, processor, head, image_paths, device, batch_size):
    rows = []
    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start:start + batch_size]
        images = [Image.open(path).convert("RGB") for path in batch_paths]
        inputs = processor(images=images, return_tensors="pt").to(device)
        with torch.no_grad():
            image_embeds = model.get_image_features(**inputs)
            if not torch.is_tensor(image_embeds):
                image_embeds = getattr(image_embeds, "pooler_output", None)
            if not torch.is_tensor(image_embeds):
                raise TypeError(
                    "CLIP get_image_features returned no pooled tensor"
                )
            image_embeds = image_embeds / image_embeds.norm(
                dim=-1, keepdim=True
            )
            scores = head(image_embeds).squeeze(-1)
        rows.extend(zip(batch_paths, scores.detach().cpu().tolist()))
    return rows


def evaluate_aesthetic(
    image_path,
    output_json,
    *,
    model_name=DEFAULT_MODEL,
    weights_path="",
    weights_url=DEFAULT_WEIGHTS_URL,
    batch_size=8,
    device=None,
):
    """Evaluate one final pipeline image and merge its aesthetic score."""
    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(image_path)

    try:
        from transformers import CLIPModel, CLIPProcessor
    except ImportError as exc:
        raise RuntimeError(
            "transformers is required for aesthetic evaluation"
        ) from exc

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_name)
    head = load_aesthetic_head(
        weights_path,
        weights_url,
        int(model.config.projection_dim),
        device,
    )
    scored_images = aesthetic_scores(
        model, processor, head, [image_path], device, batch_size
    )
    results = [
        {"image_id": path.name, "aesthetic_score": float(score)}
        for path, score in scored_images
    ]
    update_eval_results(output_json, results, "aesthetic_score")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--weights-path", default="")
    parser.add_argument("--weights-url", default=DEFAULT_WEIGHTS_URL)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    results = evaluate_aesthetic(
        args.image,
        args.output_json,
        model_name=args.model,
        weights_path=args.weights_path,
        weights_url=args.weights_url,
        batch_size=args.batch_size,
    )
    for result in results:
        print(f"{result['image_id']}: {result['aesthetic_score']:.4f}")


if __name__ == "__main__":
    main()