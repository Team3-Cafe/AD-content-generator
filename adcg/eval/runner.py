"""Unified callable entry point for advertisement image evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path


SUPPORTED_METRICS = (
    "clip_score",
    "aesthetic_score",
    "dino_similarity",
    "hps_v2_score",
)


def run_evaluation(
    final_image,
    prompt_json,
    product_image,
    product_mask,
    output_json,
    *,
    metrics=SUPPORTED_METRICS,
    metric_options=None,
):
    """Run selected metrics and merge them into one result JSON."""
    final_image = Path(final_image)
    prompt_json = Path(prompt_json)
    product_image = Path(product_image)
    product_mask = Path(product_mask)
    output_json = Path(output_json)
    for path in (final_image, prompt_json, product_image, product_mask):
        if not path.is_file():
            raise FileNotFoundError(path)

    metrics = tuple(metrics)
    unknown = set(metrics) - set(SUPPORTED_METRICS)
    if unknown:
        raise ValueError(f"unsupported evaluation metrics: {sorted(unknown)}")
    options = dict(metric_options or {})
    unknown_options = set(options) - set(SUPPORTED_METRICS)
    if unknown_options:
        raise ValueError(
            f"options supplied for unsupported metrics: {sorted(unknown_options)}"
        )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    results = {}

    if "clip_score" in metrics:
        from .eval_clip_score_2 import evaluate_clip

        results["clip_score"] = evaluate_clip(
            final_image,
            prompt_json,
            product_mask,
            output_json,
            **dict(options.get("clip_score", {})),
        )[0]["clip_score"]

    if "aesthetic_score" in metrics:
        from .eval_LAION_aesthetic_score import evaluate_aesthetic

        results["aesthetic_score"] = evaluate_aesthetic(
            final_image,
            output_json,
            **dict(options.get("aesthetic_score", {})),
        )[0]["aesthetic_score"]

    if "dino_similarity" in metrics:
        from .eval_DINO_similarity import evaluate_dino

        results["dino_similarity"] = evaluate_dino(
            final_image,
            product_image,
            product_mask,
            output_json,
            **dict(options.get("dino_similarity", {})),
        )[0]["dino_similarity"]

    if "hps_v2_score" in metrics:
        from .eval_HPS_v2 import evaluate_hps

        results["hps_v2_score"] = evaluate_hps(
            final_image,
            prompt_json,
            output_json,
            **dict(options.get("hps_v2_score", {})),
        )[0]["hps_v2_score"]

    return {"output_json": output_json, "scores": results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--final-image", required=True)
    parser.add_argument("--prompt-json", required=True)
    parser.add_argument("--product-image", required=True)
    parser.add_argument("--product-mask", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=SUPPORTED_METRICS,
        default=list(SUPPORTED_METRICS),
    )
    args = parser.parse_args()

    result = run_evaluation(
        final_image=args.final_image,
        prompt_json=args.prompt_json,
        product_image=args.product_image,
        product_mask=args.product_mask,
        output_json=args.output_json,
        metrics=args.metrics,
    )
    print(f"evaluation saved: {result['output_json']}")
    for metric, score in result["scores"].items():
        print(f"{metric}: {score:.4f}")


if __name__ == "__main__":
    main()
