import cv2
import numpy as np
from PIL import Image


def _foreground_bbox(binary_mask):
    points = cv2.findNonZero(binary_mask)

    if points is None:
        raise ValueError("Canny 생성을 위한 상품 영역이 없습니다.")

    return cv2.boundingRect(points)


def create_canny_control(
    product_layer,
    product_mask,
    low_threshold=100,
    high_threshold=200,
    contact_ratio=0.06,
    truncation=None,
    edge_suppression=3,
):
    rgba = np.asarray(
        product_layer.convert("RGBA"),
        dtype=np.uint8,
    )
    alpha = np.asarray(
        product_mask.convert("L"),
        dtype=np.uint8,
    )

    binary = np.where(
        alpha >= 32,
        255,
        0,
    ).astype(np.uint8)

    rgb = rgba[:, :, :3].copy()
    rgb[alpha < 32] = 0

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    detail_edges = cv2.Canny(
        gray,
        low_threshold,
        high_threshold,
    )
    silhouette_edges = cv2.Canny(
        binary,
        50,
        150,
    )

    detail_edges[binary == 0] = 0
    edges = np.maximum(detail_edges, silhouette_edges)

    edges = cv2.dilate(
        edges,
        np.ones((3, 3), dtype=np.uint8),
        iterations=1,
    )

    x, y, width, height = _foreground_bbox(binary)

    # 하단 접촉면은 배경이 자연스럽게 이어지도록 edge를 약화한다.
    contact_height = max(
        2,
        int(height * max(0.0, contact_ratio)),
    )
    contact_top = max(y, y + height - contact_height)
    edges[
        contact_top:y + height + 1,
        x:x + width + 1,
    ] = 0

    touching_edges = set(
        (truncation or {}).get("touching_edges", [])
    )
    suppression = max(0, int(edge_suppression))

    if suppression > 0:
        if "top" in touching_edges:
            edges[y:y + suppression, x:x + width + 1] = 0
        if "bottom" in touching_edges:
            edges[
                max(y, y + height - suppression):y + height + 1,
                x:x + width + 1,
            ] = 0
        if "left" in touching_edges:
            edges[y:y + height + 1, x:x + suppression] = 0
        if "right" in touching_edges:
            edges[
                y:y + height + 1,
                max(x, x + width - suppression):x + width + 1,
            ] = 0

    return Image.fromarray(edges, mode="L").convert("RGB")
