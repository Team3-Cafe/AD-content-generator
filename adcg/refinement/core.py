from adcg.image_utils.masks import blur_mask, ellipse_kernel
from adcg.image_utils.blending import (
    create_contact_shadow, defringe_rgba, resize_product_to_mask,
)

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from argparse import Namespace












def _run_core_refinement(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated = Image.open(
        args.generated_image
    ).convert("RGB")
    generated_array = np.asarray(
        generated,
        dtype=np.uint8,
    )

    product = Image.open(
        args.product_image
    ).convert("RGBA")

    product = defringe_rgba(product, radius=2)

    product_mask_image = Image.open(
        args.product_mask
    ).convert("L")

    if product_mask_image.size != generated.size:
        product_mask_image = product_mask_image.resize(
            generated.size,
            Image.Resampling.BILINEAR,
        )

    product_mask = np.asarray(
        product_mask_image,
        dtype=np.uint8,
    )
    
    # 알파 경계의 작은 돌기와 잔여 픽셀 완화
    product_mask = cv2.medianBlur(product_mask, 3)
    product_mask = product_mask.copy()
    product_mask[product_mask < args.alpha_threshold] = 0
    
    product_canvas, bbox = resize_product_to_mask(
        product,
        product_mask,
        args.alpha_threshold,
    )

    product_rgb = product_canvas[..., :3].astype(np.float32)

    binary_mask = np.where(
        product_mask >= args.alpha_threshold,
        255,
        0,
    ).astype(np.uint8)

    # 1. 상품 중심부 보호 마스크
    core_mask = cv2.erode(
        binary_mask,
        ellipse_kernel(args.core_erode),
        iterations=1,
    )
    core_weight = blur_mask(
        core_mask,
        args.core_feather,
    )
    core_weight *= product_mask.astype(np.float32) / 255.0
    core_weight *= args.core_opacity
    core_weight = np.clip(core_weight, 0.0, 1.0)

    # 2. 상품 외곽 보호 영역 및 경계 링
    outer_mask = cv2.dilate(
        binary_mask,
        ellipse_kernel(args.outer_protection),
        iterations=1,
    )

    boundary_ring = cv2.subtract(
        outer_mask,
        core_mask,
    )

    # 3. 상품 외부 배경에만 약한 bilateral refinement 적용
    refined_background = cv2.bilateralFilter(
        generated_array,
        d=7,
        sigmaColor=28,
        sigmaSpace=28,
    ).astype(np.float32)

    background_mask = 255 - outer_mask
    background_weight = blur_mask(
        background_mask,
        3.0,
    )
    background_weight *= args.background_strength
    background_weight = background_weight[..., None]

    generated_float = generated_array.astype(np.float32)

    result = (
        generated_float * (1.0 - background_weight)
        + refined_background * background_weight
    )

    # 4. 상품 하단에만 약한 접지 그림자 적용
    shadow = create_contact_shadow(
        product_mask,
        bbox,
        args.shadow_offset,
        args.shadow_blur,
        args.shadow_strength,
    )

    result *= 1.0 - shadow[..., None]

    # 5. 상품 중심부 원본 복원
    core_weight_3d = core_weight[..., None]

    result = (
        result * (1.0 - core_weight_3d)
        + product_rgb * core_weight_3d
    )

    result = np.clip(
        result,
        0,
        255,
    ).astype(np.uint8)

    Image.fromarray(result).save(
        output_dir / "final_core_ring_refined.png"
    )
    Image.fromarray(core_mask).save(
        output_dir / "core_mask.png"
    )
    Image.fromarray(boundary_ring).save(
        output_dir / "boundary_ring.png"
    )
    Image.fromarray(background_mask).save(
        output_dir / "background_refine_mask.png"
    )
    Image.fromarray(
        np.clip(shadow * 255, 0, 255).astype(np.uint8)
    ).save(
        output_dir / "contact_shadow_mask.png"
    )

    final_path = output_dir / "final_core_ring_refined.png"
    Image.fromarray(result).save(final_path)

    print("[완료]")
    print("최종 결과:", final_path)

    return final_path


CORE_DEFAULTS = {
    "alpha_threshold": 40,
    "core_erode": 10,
    "core_feather": 7.0,
    "core_opacity": 0.92,
    "outer_protection": 7,
    "background_strength": 0.20,
    "shadow_offset": 3,
    "shadow_blur": 6.0,
    "shadow_strength": 0.12,
}


def run_core_refinement(
    generated_image,
    product_image,
    product_mask,
    output_dir,
    **overrides,
):
    unknown = set(overrides) - set(CORE_DEFAULTS)
    if unknown:
        raise TypeError(f"지원하지 않는 보정 옵션: {sorted(unknown)}")

    config = {
        **CORE_DEFAULTS,
        **overrides,
        "generated_image": str(generated_image),
        "product_image": str(product_image),
        "product_mask": str(product_mask),
        "output_dir": str(output_dir),
    }

    return _run_core_refinement(Namespace(**config))
