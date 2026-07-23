import cv2
import numpy as np
from PIL import Image, ImageFilter


def create_background_inpaint_mask(
    product_mask,
    boundary_width=4,
    blur=2.0,
    contact_ratio=0.06,
):
    alpha = np.asarray(
        product_mask.convert("L"),
        dtype=np.uint8,
    )

    binary = np.where(
        alpha >= 32,
        255,
        0,
    ).astype(np.uint8)

    if not np.any(binary):
        raise ValueError("인페인팅 마스크에 상품 영역이 없습니다.")

    if boundary_width > 0:
        kernel_size = boundary_width * 2 + 1
        kernel = np.ones(
            (kernel_size, kernel_size),
            dtype=np.uint8,
        )
        protected_core = cv2.erode(
            binary,
            kernel,
            iterations=1,
        )
    else:
        protected_core = binary.copy()

    # 흰색: 생성 영역 / 검정색: 보존할 상품 core
    inpaint_mask = 255 - protected_core

    points = cv2.findNonZero(binary)

    if points is not None and contact_ratio > 0:
        x, y, width, height = cv2.boundingRect(points)
        contact_height = max(
            2,
            int(height * contact_ratio),
        )
        contact_top = max(y, y + height - contact_height)

        inpaint_mask[
            contact_top:y + height + 1,
            x:x + width + 1,
        ] = 255

    mask = Image.fromarray(
        inpaint_mask,
        mode="L",
    )

    if blur > 0:
        mask = mask.filter(
            ImageFilter.GaussianBlur(blur)
        )

    return mask