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
    "https://github.com/christophschuhmann/"
    "improved-aesthetic-predictor/raw/main/"
    "sac%2Blogos%2Bava1-l14-linearMSE.pth"
)


class AestheticPredictor(nn.Module):
    """MLP architecture used by LAION Aesthetic Predictor V2."""

    def __init__(self, embedding_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embedding_dim, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, embeddings):
        return self.layers(embeddings)


def load_aesthetic_head(weights_path, weights_url, embedding_dim, device):
    head = AestheticPredictor(embedding_dim)
    if weights_path:
        state = torch.load(weights_path, map_location="cpu")
    else:
        state = torch.hub.load_state_dict_from_url(
            weights_url, map_location="cpu", progress=True
        )
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    cleaned_state = {
        key.removeprefix("module."): value for key, value in state.items()
    }
    head.load_state_dict(cleaned_state)
    return head.to(device).eval()


def aesthetic_scores(model, processor, head, image_paths, device, batch_size):
    rows = []
    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start:start + batch_size]
        images = []
        for path in batch_paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
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


def score_aesthetic_images(
    image_paths,
    *,
    model_name=DEFAULT_MODEL,
    weights_path="",
    weights_url=DEFAULT_WEIGHTS_URL,
    batch_size=8,
    device=None,
):
    """Return LAION aesthetic scores without writing evaluation output."""
    paths = [Path(path) for path in image_paths]
    if not paths:
        return []
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing[0])

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
    return aesthetic_scores(
        model,
        processor,
        head,
        paths,
        device,
        batch_size,
    )


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

    scored_images = score_aesthetic_images(
        [image_path],
        model_name=model_name,
        weights_path=weights_path,
        weights_url=weights_url,
        batch_size=batch_size,
        device=device,
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
