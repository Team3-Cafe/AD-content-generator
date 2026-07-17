"""HPS v2 image and prompt preference evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

from .eval_result_store import update_eval_results
from .eval_utils import load_background_prompt


def normalize_scores(scores):
    if hasattr(scores, "detach"):
        scores = scores.detach().cpu().flatten().tolist()
    elif hasattr(scores, "tolist"):
        scores = scores.tolist()
    if isinstance(scores, (int, float)):
        scores = [scores]
    return [float(score) for score in scores]


def evaluate_hps(
    image_path,
    prompt_json,
    output_json,
    *,
    hps_version="v2.1",
):
    """Evaluate one pipeline result against its background prompt."""
    image_path = Path(image_path)
    prompt_json = Path(prompt_json)
    for path in (image_path, prompt_json):
        if not path.is_file():
            raise FileNotFoundError(path)
    if hps_version not in {"v2.0", "v2.1"}:
        raise ValueError(f"unsupported HPS version: {hps_version}")

    try:
        import hpsv2
    except ImportError as exc:
        raise RuntimeError(
            "HPSv2 is not installed. Install it before enabling hps_v2."
        ) from exc

    prompt = load_background_prompt(prompt_json)
    try:
        raw_scores = hpsv2.score(
            [str(image_path)],
            prompt,
            hps_version=hps_version,
        )
    except FileNotFoundError as exc:
        if exc.filename and exc.filename.endswith(
            "bpe_simple_vocab_16e6.txt.gz"
        ):
            raise RuntimeError(
                "The PyPI hpsv2 wheel is missing its tokenizer vocabulary. "
                "Reinstall hpsv2 from the pinned GitHub source in "
                "requirements.txt."
            ) from exc
        raise

    scores = normalize_scores(raw_scores)
    if len(scores) != 1:
        raise RuntimeError("HPSv2 returned an unexpected score count")

    results = [{"image_id": image_path.name, "hps_v2_score": scores[0]}]
    update_eval_results(output_json, results, "hps_v2_score")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument(
        "--hps-version",
        choices=["v2.0", "v2.1"],
        default="v2.1",
    )
    args = parser.parse_args()

    results = evaluate_hps(
        args.image,
        args.prompt_json,
        args.output_json,
        hps_version=args.hps_version,
    )
    print(f"{results[0]['image_id']}: {results[0]['hps_v2_score']:.4f}")


if __name__ == "__main__":
    main()
