from PIL import Image

RESAMPLING = getattr(Image, "Resampling", Image).LANCZOS


def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def calculate_auto_layout(
    product,
    canvas_width,
    canvas_height,
    gpt_layout,
    reference_weight=0.35,
):
    """
    누끼 종횡비로 기본 배치를 계산하고 GPT layout을 참고값으로 반영한다.
    """
    aspect_ratio = product.width / max(product.height, 1)

    # place_product에서 scale은 캔버스 너비 대비 상품 너비를 의미함
    if aspect_ratio >= 2.0:
        auto_scale = 0.72
    elif aspect_ratio >= 1.45:
        auto_scale = 0.66
    elif aspect_ratio >= 1.0:
        auto_scale = 0.58
    elif aspect_ratio >= 0.70:
        auto_scale = 0.48
    else:
        auto_scale = 0.40

    # place_product와 동일한 방식으로 예상 resize 크기 계산
    resized_width = max(1, int(canvas_width * auto_scale))
    resize_ratio = resized_width / product.width
    resized_height = max(1, int(product.height * resize_ratio))

    max_height = int(canvas_height * 0.78)

    if resized_height > max_height:
        resize_ratio = max_height / product.height
        resized_width = max(1, int(product.width * resize_ratio))
        resized_height = max_height
        auto_scale = resized_width / canvas_width

    auto_x = 0.50

    # 상품 하단이 캔버스 하단에서 약 7% 떨어지도록 계산
    bottom_margin = canvas_height * 0.07
    auto_center_y = (
        canvas_height
        - bottom_margin
        - resized_height / 2
    )
    auto_y = auto_center_y / canvas_height

    gpt_x = float(gpt_layout.get("product_x", auto_x))
    gpt_y = float(gpt_layout.get("product_y", auto_y))
    gpt_scale = float(
        gpt_layout.get("product_scale", auto_scale)
    )

    weight = clamp(reference_weight, 0.0, 1.0)

    product_x = auto_x * (1.0 - weight) + gpt_x * weight
    product_y = auto_y * (1.0 - weight) + gpt_y * weight
    product_scale = (
        auto_scale * (1.0 - weight)
        + gpt_scale * weight
    )

    return {
        "product_x": clamp(product_x, 0.15, 0.85),
        "product_y": clamp(product_y, 0.25, 0.90),
        "product_scale": clamp(product_scale, 0.30, 0.75),
        "auto_layout": {
            "product_x": auto_x,
            "product_y": auto_y,
            "product_scale": auto_scale,
            "aspect_ratio": aspect_ratio,
        },
        "gpt_layout": {
            "product_x": gpt_x,
            "product_y": gpt_y,
            "product_scale": gpt_scale,
        },
        "reference_weight": weight,
    }


def place_product(
    product,
    width,
    height,
    product_x,
    product_y,
    product_scale,
):
    target_width = max(1, int(width * product_scale))
    resize_ratio = target_width / product.width

    resized_width = target_width
    resized_height = max(1, int(product.height * resize_ratio))

    max_height = int(height * 0.78)

    if resized_height > max_height:
        resize_ratio = max_height / product.height
        resized_width = max(1, int(product.width * resize_ratio))
        resized_height = max_height

    resized_product = product.resize(
        (resized_width, resized_height),
        RESAMPLING,
    )

    center_x = int(width * product_x)
    center_y = int(height * product_y)

    left = center_x - resized_width // 2
    top = center_y - resized_height // 2

    left = max(0, min(left, width - resized_width))
    top = max(0, min(top, height - resized_height))

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

    product_mask = product_layer.getchannel("A")

    placement = {
        "left": left,
        "top": top,
        "width": resized_width,
        "height": resized_height,
        "center_x": center_x,
        "center_y": center_y,
    }

    return (
        resized_product,
        product_layer,
        condition_canvas,
        product_mask,
        placement,
    )


def place_product_preserve(
    product,
    width,
    height,
):
    source_width, source_height = product.size

    resize_ratio = min(
        width / source_width,
        height / source_height,
    )

    resized_width = max(
        1,
        round(source_width * resize_ratio),
    )
    resized_height = max(
        1,
        round(source_height * resize_ratio),
    )

    resized_product = product.resize(
        (resized_width, resized_height),
        RESAMPLING,
    )

    left = (width - resized_width) // 2
    top = (height - resized_height) // 2

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

    product_mask = product_layer.getchannel("A")
    product_bbox = product_mask.getbbox()

    placement = {
        "mode": "preserve",
        "source_canvas_width": source_width,
        "source_canvas_height": source_height,
        "canvas_scale": resize_ratio,
        "canvas_left": left,
        "canvas_top": top,
        "canvas_width": resized_width,
        "canvas_height": resized_height,
        "product_bbox": {
            "left": product_bbox[0],
            "top": product_bbox[1],
            "right": product_bbox[2],
            "bottom": product_bbox[3],
        },
    }

    return (
        resized_product,
        product_layer,
        condition_canvas,
        product_mask,
        placement,
    )
