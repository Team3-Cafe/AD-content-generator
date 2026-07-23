import cv2
import numpy as np
from PIL import Image, ImageFilter


def match_product_luminance(
    product_rgb,
    generated_rgb,
    inner_mask,
):
    """
    상품 색상은 유지하고 밝기만 생성 장면에 맞춤.
    배경의 초록색 등이 상품에 전이되는 현상을 줄이기 위해
    LAB 색 공간의 L 채널만 보정함.
    """
    product_np = np.asarray(
        product_rgb.convert("RGB"),
        dtype=np.uint8,
    )
    generated_np = np.asarray(
        generated_rgb.convert("RGB"),
        dtype=np.uint8,
    )
    mask_np = np.asarray(
        inner_mask.convert("L"),
        dtype=np.uint8,
    ) > 32

    if mask_np.sum() < 10:
        return product_rgb.convert("RGB")

    product_lab = cv2.cvtColor(
        product_np,
        cv2.COLOR_RGB2LAB,
    ).astype(np.float32)

    generated_lab = cv2.cvtColor(
        generated_np,
        cv2.COLOR_RGB2LAB,
    ).astype(np.float32)

    source_l = product_lab[:, :, 0]
    target_l = generated_lab[:, :, 0]

    source_mean = source_l[mask_np].mean()
    source_std = max(source_l[mask_np].std(), 1.0)

    target_mean = target_l[mask_np].mean()
    target_std = max(target_l[mask_np].std(), 1.0)

    scale = np.clip(
        target_std / source_std,
        0.75,
        1.25,
    )

    adjusted_l = (
        (source_l - source_mean) * scale
        + target_mean
    )

    product_lab[:, :, 0] = np.clip(
        adjusted_l,
        0,
        255,
    )

    matched_rgb = cv2.cvtColor(
        product_lab.astype(np.uint8),
        cv2.COLOR_LAB2RGB,
    )

    return Image.fromarray(matched_rgb, mode="RGB")


def restore_product_inner_detail(
    generated_image,
    product_layer,
    product_mask,
    inner_erode,
    inner_feather,
    opacity,
    color_match,
):
    """
    diffusion이 만든 경계와 접촉면은 그대로 유지하고
    상품 내부 영역에만 원본 디테일을 복원함.
    """
    generated_image = generated_image.convert("RGB")
    product_layer = product_layer.convert("RGBA")
    product_mask = product_mask.convert("L")

    if inner_erode > 0:
        erode_size = inner_erode * 2 + 1

        if erode_size % 2 == 0:
            erode_size += 1

        inner_mask = product_mask.filter(
            ImageFilter.MinFilter(erode_size)
        )
    else:
        inner_mask = product_mask.copy()

    if inner_feather > 0:
        inner_mask = inner_mask.filter(
            ImageFilter.GaussianBlur(inner_feather)
        )

    inner_np = (
        np.asarray(inner_mask, dtype=np.float32)
        / 255.0
    )
    alpha_np = (
        np.asarray(
            product_layer.getchannel("A"),
            dtype=np.float32,
        )
        / 255.0
    )

    restore_alpha = (
        inner_np
        * alpha_np
        * np.clip(opacity, 0.0, 1.0)
    )

    restore_mask = Image.fromarray(
        np.clip(
            restore_alpha * 255.0,
            0,
            255,
        ).astype(np.uint8),
        mode="L",
    )

    product_rgb = product_layer.convert("RGB")

    if color_match:
        product_rgb = match_product_luminance(
            product_rgb=product_rgb,
            generated_rgb=generated_image,
            inner_mask=inner_mask,
        )

    final_image = Image.composite(
        product_rgb,
        generated_image,
        restore_mask,
    )

    return final_image, inner_mask, restore_mask


def resize_product_to_mask(product, positioned_mask, threshold):
    binary = np.where(
        positioned_mask >= threshold,
        255,
        0,
    ).astype(np.uint8)

    points = cv2.findNonZero(binary)
    if points is None:
        raise ValueError("Product mask is empty.")

    x, y, width, height = cv2.boundingRect(points)

    resized = product.resize(
        (width, height),
        Image.Resampling.LANCZOS,
    )

    canvas = Image.new(
        "RGBA",
        (positioned_mask.shape[1], positioned_mask.shape[0]),
        (0, 0, 0, 0),
    )
    canvas.alpha_composite(resized, (x, y))

    return np.asarray(canvas), (x, y, width, height)


def create_contact_shadow(
    product_mask,
    bbox,
    offset,
    blur,
    strength,
):
    x, y, width, height = bbox
    mask = product_mask.astype(np.float32) / 255.0

    vertical_weight = np.zeros_like(mask)
    start_y = int(y + height * 0.72)
    end_y = min(y + height, mask.shape[0])

    if end_y > start_y:
        gradient = np.linspace(
            0.0,
            1.0,
            end_y - start_y,
            dtype=np.float32,
        )
        vertical_weight[start_y:end_y, :] = gradient[:, None]

    lower_mask = mask * vertical_weight

    transform = np.float32([
        [1, 0, 0],
        [0, 1, offset],
    ])

    shifted = cv2.warpAffine(
        lower_mask,
        transform,
        (mask.shape[1], mask.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderValue=0,
    )

    shifted = cv2.GaussianBlur(
        shifted,
        (0, 0),
        sigmaX=blur,
        sigmaY=blur,
    )

    # 그림자는 상품 내부가 아닌 외부에만 적용
    shifted *= 1.0 - mask

    return np.clip(shifted * strength, 0.0, 1.0)


def defringe_rgba(image, radius=2):
    rgba = np.array(image)
    rgb = rgba[..., :3]
    alpha = rgba[..., 3]

    foreground = np.where(alpha >= 64, 255, 0).astype(np.uint8)
    inner = cv2.erode(
        foreground,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (radius * 2 + 1, radius * 2 + 1),
        ),
    )

    fringe = cv2.subtract(foreground, inner)
    cleaned_rgb = cv2.inpaint(rgb, fringe, 3, cv2.INPAINT_TELEA)

    result = np.dstack([cleaned_rgb, alpha])
    return Image.fromarray(result.astype(np.uint8), "RGBA")


def trim_transparent_product(product):
    alpha = product.getchannel("A")
    bbox = alpha.getbbox()

    if bbox is None:
        raise ValueError("Product image alpha channel is empty.")

    return product.crop(bbox)


def align_product_to_mask(product_path, binary_mask, canvas_size):
    product = Image.open(product_path).convert("RGBA")
    product = trim_transparent_product(product)

    points = cv2.findNonZero(binary_mask)
    if points is None:
        raise ValueError("Product mask is empty.")

    x, y, width, height = cv2.boundingRect(points)

    product = product.resize(
        (width, height),
        Image.Resampling.LANCZOS,
    )

    canvas = Image.new(
        "RGBA",
        canvas_size,
        (0, 0, 0, 0),
    )
    canvas.alpha_composite(product, (x, y))

    return canvas


def create_identity_weight(
    aligned_alpha,
    binary_mask,
    feather,
    opacity,
):
    effective_alpha = np.minimum(
        aligned_alpha,
        binary_mask,
    ).astype(np.uint8)

    effective_binary = np.where(
        effective_alpha > 0,
        255,
        0,
    ).astype(np.uint8)

    distance = cv2.distanceTransform(
        effective_binary,
        cv2.DIST_L2,
        5,
    )

    # Keep the immediate boundary generated, then restore the core gradually.
    weight = np.clip(
        (distance - 1.0) / max(feather, 1.0),
        0.0,
        1.0,
    )

    weight *= effective_alpha.astype(np.float32) / 255.0
    weight *= opacity

    return np.clip(weight, 0.0, 1.0)


def color_match_product(
    product_rgb,
    target_rgb,
    valid_mask,
    amount,
):
    if amount <= 0 or np.count_nonzero(valid_mask) < 50:
        return product_rgb

    source = product_rgb[valid_mask]
    target = target_rgb[valid_mask]

    source_mean = source.mean(axis=0)
    source_std = source.std(axis=0) + 1e-6
    target_mean = target.mean(axis=0)
    target_std = target.std(axis=0) + 1e-6

    matched = (
        (product_rgb - source_mean)
        * (target_std / source_std)
        + target_mean
    )

    matched = np.clip(matched, 0, 255)

    return (
        product_rgb * (1.0 - amount)
        + matched * amount
    )
