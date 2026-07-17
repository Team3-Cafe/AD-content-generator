from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


DIRECTIONS = ("product_focus", "brand_focus")
LAYOUT_MODES = ("layout", "preserve")
EVAL_METRICS = (
    "clip_score",
    "aesthetic_score",
    "dino_similarity",
    "hps_v2_score",
)


@dataclass(frozen=True)
class AppConfig:
    image_path: Path
    info_path: Path
    output_dir: Path
    gpt_model: str
    direction: str
    layout_mode: str
    copy_count: int
    evaluate: bool
    eval_metrics: tuple[str, ...] | None
    seed: int
    cpu_offload: bool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Configure the advertisement image and copywriting pipelines."
    )

    common = parser.add_argument_group("common inputs")
    common.add_argument("--image", required=True, help="Product image file or directory")
    common.add_argument(
        "--info",
        help="Existing product/store info JSON. If omitted, one is generated from the fields below.",
    )
    common.add_argument("--output-dir", default="outputs/pipeline")
    common.add_argument("--seed", type=int, default=42)
    common.add_argument("--cpu-offload", action="store_true")

    image_generation = parser.add_argument_group("image generation")
    image_generation.add_argument("--gpt-model", default="gpt-5.4-nano")
    image_generation.add_argument(
        "--direction",
        choices=DIRECTIONS,
        default="product_focus",
    )
    image_generation.add_argument(
        "--layout-mode",
        choices=LAYOUT_MODES,
        default="layout",
    )

    copywriting = parser.add_argument_group("copywriting")
    copywriting.add_argument("--copy-count", type=int, default=9)

    evaluation = parser.add_argument_group("evaluation")
    evaluation.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate final_identity_restored.png after generation.",
    )
    evaluation.add_argument(
        "--eval-metric",
        dest="eval_metrics",
        action="append",
        choices=EVAL_METRICS,
        help="Evaluation metric to run. Repeat to select multiple metrics.",
    )

    info = parser.add_argument_group("product and store information")
    info.add_argument("--product-name")
    info.add_argument("--store-name")
    info.add_argument("--store-type", default="소상공인 매장")
    info.add_argument("--product-category", default="")
    info.add_argument("--product-description", default="")
    info.add_argument(
        "--feature",
        dest="features",
        action="append",
        default=[],
        help="Product feature. Repeat this option for multiple values.",
    )
    info.add_argument("--target-customer", default="")
    info.add_argument("--promotion", default="")
    info.add_argument("--price", default="")
    info.add_argument("--tone", default="친근하고 신뢰감 있는 분위기")
    info.add_argument("--desired-scene", default="")
    info.add_argument("--additional-request", default="")
    info.add_argument(
        "--info-output",
        help="Generated info JSON path. Defaults to <output-dir>/00_config/product_info.json.",
    )

    return parser


def _load_environment() -> None:
    if load_dotenv is not None:
        load_dotenv()
        return

    env_path = Path(".env")
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _validate_api_key(parser: argparse.ArgumentParser) -> None:
    _load_environment()
    if not os.getenv("OPENAI_API_KEY"):
        parser.error(
            "OPENAI_API_KEY가 없습니다. 프로젝트 루트의 .env에 "
            "OPENAI_API_KEY=... 형식으로 설정해주세요."
        )


def _validate_image(parser: argparse.ArgumentParser, image_path: Path) -> None:
    if not image_path.exists():
        parser.error(f"상품 이미지 경로를 찾을 수 없습니다: {image_path}")


def _build_info_data(args: argparse.Namespace) -> dict:
    return {
        "product_name": args.product_name,
        "store_name": args.store_name,
        "store_type": args.store_type,
        "product_category": args.product_category,
        "product_description": args.product_description,
        "features": args.features,
        "target_customer": args.target_customer,
        "promotion": args.promotion,
        "price": args.price,
        "tone": args.tone,
        "desired_scene": args.desired_scene,
        "additional_request": args.additional_request,
    }


def _resolve_info_path(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    output_dir: Path,
) -> Path:
    if args.info:
        info_path = Path(args.info).expanduser()
        if not info_path.exists():
            parser.error(f"info JSON을 찾을 수 없습니다: {info_path}")
        try:
            json.loads(info_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            parser.error(f"info JSON을 읽을 수 없습니다: {error}")
        return info_path.resolve()

    if not args.product_name or not args.store_name:
        parser.error(
            "--info를 생략할 때는 --product-name과 --store-name이 필요합니다."
        )

    info_path = (
        Path(args.info_output).expanduser()
        if args.info_output
        else output_dir / "00_config" / "product_info.json"
    )
    info_path.parent.mkdir(parents=True, exist_ok=True)
    info_path.write_text(
        json.dumps(_build_info_data(args), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return info_path.resolve()


def parse_config(argv: list[str] | None = None) -> AppConfig:
    parser = build_parser()
    args = parser.parse_args(argv)

    _validate_api_key(parser)

    image_path = Path(args.image).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    _validate_image(parser, image_path)

    if args.copy_count < 1:
        parser.error("--copy-count는 1 이상이어야 합니다.")

    output_dir.mkdir(parents=True, exist_ok=True)
    info_path = _resolve_info_path(parser, args, output_dir)

    return AppConfig(
        image_path=image_path,
        info_path=info_path,
        output_dir=output_dir,
        gpt_model=args.gpt_model,
        direction=args.direction,
        layout_mode=args.layout_mode,
        copy_count=args.copy_count,
        evaluate=args.evaluate or bool(args.eval_metrics),
        eval_metrics=(
            tuple(args.eval_metrics)
            if args.eval_metrics
            else None
        ),
        seed=args.seed,
        cpu_offload=args.cpu_offload,
    )


if __name__ == "__main__":
    config = parse_config()
    print(json.dumps({
        "image_path": str(config.image_path),
        "info_path": str(config.info_path),
        "output_dir": str(config.output_dir),
        "gpt_model": config.gpt_model,
        "direction": config.direction,
        "layout_mode": config.layout_mode,
        "copy_count": config.copy_count,
        "evaluate": config.evaluate,
        "eval_metrics": config.eval_metrics,
        "seed": config.seed,
        "cpu_offload": config.cpu_offload,
    }, ensure_ascii=False, indent=2))
