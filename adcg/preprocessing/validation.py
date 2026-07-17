import warnings

import numpy as np


TRUNCATION_POLICIES = ("allow", "warn", "error")


def _edge_coverage(mask, edge_margin):
    height, width = mask.shape
    margin = max(1, min(int(edge_margin), height, width))

    return {
        "top": float(mask[:margin, :].any(axis=0).mean()),
        "bottom": float(mask[-margin:, :].any(axis=0).mean()),
        "left": float(mask[:, :margin].any(axis=1).mean()),
        "right": float(mask[:, -margin:].any(axis=1).mean()),
    }


def detect_truncation(
    cutout,
    edge_margin=2,
    min_edge_coverage=0.01,
    alpha_threshold=32,
):
    """
    상품 알파 영역이 이미지 경계에 닿았는지 검사한다.

    이미지 경계에 닿은 상품은 촬영 단계에서 일부가 잘렸을
    가능성이 있으므로 원본 정체성 보존이 어려울 수 있다.
    """
    alpha = np.asarray(
        cutout.convert("RGBA").getchannel("A")
    )

    foreground = alpha >= alpha_threshold
    coverage = _edge_coverage(foreground, edge_margin)

    touching_edges = [
        edge
        for edge, ratio in coverage.items()
        if ratio >= min_edge_coverage
    ]

    return {
        "is_truncated": bool(touching_edges),
        "touching_edges": touching_edges,
        "edge_coverage": {
            edge: round(ratio, 6)
            for edge, ratio in coverage.items()
        },
        "edge_margin": int(edge_margin),
        "min_edge_coverage": float(min_edge_coverage),
        "alpha_threshold": int(alpha_threshold),
    }


def handle_truncation(truncation, policy="warn"):
    if policy not in TRUNCATION_POLICIES:
        raise ValueError(
            f"truncation_policy는 "
            f"{TRUNCATION_POLICIES} 중 하나여야 합니다."
        )

    if not truncation["is_truncated"] or policy == "allow":
        return

    touching_edges = ", ".join(
        truncation["touching_edges"]
    )

    message = (
        f"상품이 이미지 경계에 닿아 있습니다: {touching_edges}. "
        "이미 잘린 부분은 원본 정체성을 유지하며 복원할 수 없습니다. "
        "가능하면 상품 전체가 포함된 사진을 사용하세요."
    )

    if policy == "error":
        raise ValueError(message)

    warnings.warn(
        message,
        RuntimeWarning,
        stacklevel=2,
    )