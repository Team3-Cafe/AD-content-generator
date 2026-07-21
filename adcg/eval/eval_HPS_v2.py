"""HPS v2 image and prompt preference evaluation."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib
from pathlib import Path
import sys
from types import ModuleType

from .eval_result_store import update_eval_results
from .eval_utils import load_background_prompt


@contextmanager
def _headless_hpsv2_import_compatibility():
    """Avoid HPSv2's unused turtle dependency on headless servers."""
    try:
        importlib.import_module("tkinter")
    except ImportError:
        pass
    else:
        yield
        return

    existing_turtle = sys.modules.get("turtle")
    if existing_turtle is not None:
        yield
        return

    turtle_stub = ModuleType("turtle")

    def unsupported_forward(*_args, **_kwargs):
        raise RuntimeError(
            "HPSv2 unexpectedly tried to use turtle.forward on a "
            "headless server"
        )

    turtle_stub.forward = unsupported_forward
    sys.modules["turtle"] = turtle_stub
    try:
        yield
    finally:
        if sys.modules.get("turtle") is turtle_stub:
            del sys.modules["turtle"]


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

    prompt = load_background_prompt(prompt_json)
    with _headless_hpsv2_import_compatibility():
        try:
            import hpsv2
        except ImportError as exc:
            raise RuntimeError(
                "HPSv2 is not installed. Install it before enabling hps_v2."
            ) from exc

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
                    "The PyPI hpsv2 wheel is missing its tokenizer "
                    "vocabulary. Reinstall hpsv2 from the pinned GitHub "
                    "source in requirements.txt."
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