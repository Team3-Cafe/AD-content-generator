from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from .generator import generate_prompt_layout
from .io import load_ad_copy


def _jsonable(value):
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate one content-aware ad design, refine its draft, and "
            "review the completed advertisement once more with GPT-4o."
        )
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Completed background image.",
    )
    parser.add_argument(
        "--ad-copy",
        required=True,
        help="ad_copy.json path.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for design JSON and rendered images.",
    )
    parser.add_argument(
        "--copy-index",
        type=int,
        default=0,
        help="Index in ad_copy.json copies array.",
    )
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument(
        "--font",
        help="Optional TrueType/OpenType font used for final_ad.png.",
    )
    parser.add_argument(
        "--detail",
        choices=("low", "high", "auto"),
        default="high",
    )
    parser.add_argument("--temperature", type=float, default=0.4)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    ad_copy = load_ad_copy(
        args.ad_copy,
        copy_index=args.copy_index,
    )
    result = generate_prompt_layout(
        image_path=args.image,
        ad_copy=ad_copy,
        output_dir=args.output_dir,
        model=args.model,
        font_path=args.font,
        detail=args.detail,
        temperature=args.temperature,
    )
    print(
        json.dumps(
            {
                key: _jsonable(value)
                for key, value in asdict(result).items()
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
