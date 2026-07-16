import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate an advertisement image from a product photo."
    )
    parser.add_argument("--image", required=True)
    parser.add_argument("--info", required=True)
    parser.add_argument("--output-dir", default="outputs/pipeline")
    parser.add_argument("--gpt-model", default="gpt-5.4-nano")
    parser.add_argument(
        "--direction",
        default="product_focus",
        choices=["product_focus", "brand_focus", "sales_focus"],
    )
    parser.add_argument(
        "--layout-mode",
        default="layout",
        choices=["layout", "preserve"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu-offload", action="store_true")
    return parser.parse_args()
