import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageOps


def create_background_inpaint_mask(
    product_mask,
    margin,
    blur,
):
    protected_mask = product_mask

    if margin > 0:
        filter_size = margin * 2 + 1

        if filter_size % 2 == 0:
            filter_size += 1

        protected_mask = protected_mask.filter(
            ImageFilter.MaxFilter(filter_size)
        )

    # 흰색 영역은 생성하고 검은색 상품 영역은 보호함
    background_mask = ImageOps.invert(protected_mask)

    if blur > 0:
        background_mask = background_mask.filter(
            ImageFilter.GaussianBlur(blur)
        )

    return background_mask


def create_canny_control(
    product_layer,
    product_mask,
    low_threshold,
    high_threshold,
):
    del product_layer

    alpha = np.asarray(
        product_mask.convert("L"),
        dtype=np.uint8,
    )

    # 흐릿한 반투명 픽셀을 제거하고 상품 외곽 형태만 사용
    binary_mask = np.where(
        alpha >= 32,
        255,
        0,
    ).astype(np.uint8)

    edges = cv2.Canny(
        binary_mask,
        50,
        150,
    )

    # ControlNet이 외곽선을 안정적으로 인식하도록 1픽셀 확장
    edges = cv2.dilate(
        edges,
        np.ones((3, 3), dtype=np.uint8),
        iterations=1,
    )

    control = np.stack(
        [edges, edges, edges],
        axis=-1,
    )

    return Image.fromarray(control, mode="RGB")


def ellipse_kernel(radius):
    size = radius * 2 + 1
    return cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (size, size),
    )


def blur_mask(mask, sigma):
    if sigma <= 0:
        return mask.astype(np.float32) / 255.0

    blurred = cv2.GaussianBlur(
        mask,
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
    )
    return blurred.astype(np.float32) / 255.0


def create_boundary_masks(
    alpha_mask,
    threshold,
    inner_radius,
    outer_radius,
    blur,
    opacity,
):
    binary = np.where(
        alpha_mask >= threshold,
        255,
        0,
    ).astype(np.uint8)

    if cv2.countNonZero(binary) == 0:
        raise ValueError("Product mask is empty.")

    inside = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    outside = cv2.distanceTransform(255 - binary, cv2.DIST_L2, 5)
    signed = inside - outside

    inpaint_mask = np.where(
        (signed >= -outer_radius)
        & (signed <= inner_radius),
        255,
        0,
    ).astype(np.uint8)

    safe_inner = max(inner_radius, 1.0)
    safe_outer = max(outer_radius, 1.0)

    inside_weight = np.clip(
        1.0 - signed / safe_inner,
        0.0,
        1.0,
    )
    outside_weight = np.clip(
        1.0 + signed / safe_outer,
        0.0,
        1.0,
    )

    blend_weight = np.where(
        signed >= 0,
        inside_weight,
        outside_weight,
    ).astype(np.float32)

    blend_weight *= inpaint_mask.astype(np.float32) / 255.0

    if blur > 0:
        blend_weight = cv2.GaussianBlur(
            blend_weight,
            (0, 0),
            sigmaX=blur,
            sigmaY=blur,
        )

    blend_weight = np.clip(
        blend_weight * opacity,
        0.0,
        1.0,
    )

    return binary, inpaint_mask, blend_weight


def create_control_image(binary):
    canny = cv2.Canny(binary, 50, 150)

    canny = cv2.dilate(
        canny,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (3, 3),
        ),
        iterations=1,
    )

    return canny
