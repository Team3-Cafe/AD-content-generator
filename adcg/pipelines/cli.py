from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from ..config import EVAL_METRICS, LAYOUT_MODES
from .copy_layout import run_copy_layout_pipeline
from .image import run_image_pipeline


COPY_LENGTHS = ("short", "medium", "long")


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): _jsonable(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _add_image_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "image",
        help="Run preprocessing, image generation, and identity restoration.",
    )
    parser.add_argument("--image", required=True, help="Product image path.")
    parser.add_argument("--info", required=True, help="Product information JSON.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gpt-model", default="gpt-5.4-nano")
    parser.add_argument("--product-focus", type=float, default=1.0)
    parser.add_argument("--brand-focus", type=float, default=0.5)
    parser.add_argument(
        "--layout-mode",
        choices=LAYOUT_MODES,
        default="layout",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument(
        "--eval-metrics",
        nargs="+",
        choices=EVAL_METRICS,
        default=None,
    )
    parser.set_defaults(handler=_run_image_command)


def _add_copy_layout_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "copy-layout",
        help="Generate controlled ad copy and compose it onto a finished image.",
    )
    parser.add_argument(
        "--identity-image",
        required=True,
        help="Identity-restored image produced by the image pipeline.",
    )
    parser.add_argument("--info", required=True, help="Product information JSON.")
    parser.add_argument(
        "--prompt-json",
        required=True,
        help="Scene prompt JSON produced by the image pipeline.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gpt-model", default="gpt-5.4-nano")
    parser.add_argument("--copy-count", type=int, default=1)
    parser.add_argument("--copy-index", type=int, default=0)
    parser.add_argument("--copy-tone", required=True)
    parser.add_argument(
        "--copy-length",
        required=True,
        choices=COPY_LENGTHS,
    )
    parser.add_argument("--layout-model", default="gpt-4o")
    parser.add_argument(
        "--layout-detail",
        choices=("low", "high", "auto"),
        default="high",
    )
    parser.add_argument("--layout-temperature", type=float, default=0.4)
    parser.add_argument("--font", help="Optional layout font path.")
    parser.set_defaults(handler=_run_copy_layout_command)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m adcg.pipelines",
        description=(
            "Run the image stage and the copy-layout stage independently."
        ),
    )
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )
    _add_image_parser(subparsers)
    _add_copy_layout_parser(subparsers)
    return parser


def _run_image_command(args):
    return run_image_pipeline(
        image_path=args.image,
        info_path=args.info,
        output_dir=args.output_dir,
        gpt_model=args.gpt_model,
        product_focus=args.product_focus,
        brand_focus=args.brand_focus,
        layout_mode=args.layout_mode,
        seed=args.seed,
        cpu_offload=args.cpu_offload,
        evaluate=args.evaluate,
        eval_metrics=args.eval_metrics,
    )


def _run_copy_layout_command(args):
    layout_options = {
        "detail": args.layout_detail,
        "temperature": args.layout_temperature,
    }
    if args.font:
        layout_options["font_path"] = args.font

    return run_copy_layout_pipeline(
        identity_image=args.identity_image,
        info_path=args.info,
        prompt_json=args.prompt_json,
        output_dir=args.output_dir,
        gpt_model=args.gpt_model,
        copy_count=args.copy_count,
        copy_tone=args.copy_tone,
        copy_length=args.copy_length,
        copy_index=args.copy_index,
        layout_model=args.layout_model,
        layout_options=layout_options,
    )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = args.handler(args)
    print(
        json.dumps(
            _jsonable(asdict(result)),
            ensure_ascii=False,
            indent=2,
        )
    )
