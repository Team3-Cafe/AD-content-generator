import json
from pathlib import Path

import numpy as np
from PIL import Image
from rembg import remove

from adcg.preprocessing.product import get_rembg_session


RESAMPLING = getattr(Image, "Resampling", Image).LANCZOS


def load_metadata(metadata):
    if metadata is None:
        return {}

    if isinstance(metadata, dict):
        return metadata

    metadata_path = Path(metadata)

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"전처리 메타데이터를 찾을 수 없습니다: {metadata_path}"
        )

    return json.loads(
        metadata_path.read_text(encoding="utf-8")
    )


def load_product_cutout(
    image_path,
    mode="auto",
    alpha_threshold=4,
):
    image = Image.open(image_path).convert("RGBA")
    alpha = image.getchannel("A")
    has_transparency = alpha.getextrema()[0] < 250

    use_rembg = (
        mode == "rembg"
        or (mode == "auto" and not has_transparency)
    )

    if use_rembg:
        session = get_rembg_session("u2net")
        image = remove(
            image,
            session=session,
            alpha_matting=False,
        ).convert("RGBA")

    array = np.array(image)
    alpha_array = array[:, :, 3]

    alpha_array[alpha_array <= alpha_threshold] = 0
    array[:, :, 3] = alpha_array

    image = Image.fromarray(array, mode="RGBA")
    bbox = image.getchannel("A").getbbox()

    if bbox is None:
        raise ValueError("상품 알파 영역을 찾지 못했습니다.")

    return image.crop(bbox)


def _layout_geometry(
    product,
    width,
    height,
    product_x,
    product_y,
    product_scale,
):
    target_width = max(1, int(width * product_scale))
    ratio = target_width / product.width

    resized_width = target_width
    resized_height = max(1, int(product.height * ratio))

    max_height = int(height * 0.86)

    if resized_height > max_height:
        ratio = max_height / product.height
        resized_width = max(1, int(product.width * ratio))
        resized_height = max_height

    left = int(width * product_x) - resized_width // 2
    top = int(height * product_y) - resized_height // 2

    left = max(0, min(left, width - resized_width))
    top = max(0, min(top, height - resized_height))

    return left, top, resized_width, resized_height


def _preserve_geometry(product, width, height, metadata):
    original_size = metadata.get("original_size", {})
    bbox = metadata.get("product_bbox", {})

    original_width = int(original_size.get("width", 0))
    original_height = int(original_size.get("height", 0))

    required_bbox_keys = {"left", "top", "right", "bottom"}

    if (
        original_width <= 0
        or original_height <= 0
        or not required_bbox_keys.issubset(bbox)
    ):
        raise ValueError(
            "preserve 모드에는 original_size와 product_bbox가 "
            "포함된 전처리 메타데이터가 필요합니다."
        )

    source_scale = min(
        width / original_width,
        height / original_height,
    )

    canvas_offset_x = (
        width - original_width * source_scale
    ) / 2
    canvas_offset_y = (
        height - original_height * source_scale
    ) / 2

    bbox_width = int(bbox["right"]) - int(bbox["left"])
    bbox_height = int(bbox["bottom"]) - int(bbox["top"])

    resized_width = max(1, int(bbox_width * source_scale))
    resized_height = max(1, int(bbox_height * source_scale))

    left = int(
        canvas_offset_x + int(bbox["left"]) * source_scale
    )
    top = int(
        canvas_offset_y + int(bbox["top"]) * source_scale
    )

    left = max(0, min(left, width - resized_width))
    top = max(0, min(top, height - resized_height))

    return left, top, resized_width, resized_height


def create_condition_canvas(
    product,
    width,
    height,
    layout_mode="layout",
    product_x=0.5,
    product_y=0.7,
    product_scale=0.45,
    metadata=None,
):
    metadata = load_metadata(metadata)

    if layout_mode == "preserve":
        geometry = _preserve_geometry(
            product,
            width,
            height,
            metadata,
        )
    elif layout_mode == "layout":
        geometry = _layout_geometry(
            product,
            width,
            height,
            product_x,
            product_y,
            product_scale,
        )
    else:
        raise ValueError(
            "layout_mode은 layout 또는 preserve여야 합니다."
        )

    left, top, resized_width, resized_height = geometry

    resized_product = product.resize(
        (resized_width, resized_height),
        RESAMPLING,
    )

    product_layer = Image.new(
        "RGBA",
        (width, height),
        (0, 0, 0, 0),
    )
    product_layer.alpha_composite(
        resized_product,
        (left, top),
    )

    neutral_background = Image.new(
        "RGBA",
        (width, height),
        (181, 177, 168, 255),
    )

    condition_canvas = Image.alpha_composite(
        neutral_background,
        product_layer,
    )

    placement = {
        "mode": layout_mode,
        "left": left,
        "top": top,
        "width": resized_width,
        "height": resized_height,
        "center_x": left + resized_width // 2,
        "center_y": top + resized_height // 2,
    }

    return {
        "resized_product": resized_product,
        "product_layer": product_layer,
        "condition_canvas": condition_canvas,
        "product_mask": product_layer.getchannel("A"),
        "placement": placement,
    }
