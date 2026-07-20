from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from adcg.image_utils.blending import (
    create_contact_shadow,
    defringe_rgba,
    resize_product_to_mask,
)
from adcg.image_utils.masks import blur_mask, ellipse_kernel
from .diagnostics import (
    prepare_output_dir,
    save_diagnostics,
    save_metadata,
)


def run_core_refinement(
    generated_image,
    product_image,
    product_mask,
    output_dir="outputs/refinement/core",
    alpha_threshold=40,
    core_erode=10,
    core_feather=7.0,
    core_opacity=0.92,
    outer_protection=7,
    background_strength=0.20,
    focus_strength=1.0,
    shadow_offset=3,
    shadow_blur=6.0,
    shadow_strength=0.12,
):
    output_dir = prepare_output_dir(output_dir)

    generated = Image.open(generated_image).convert("RGB")
    generated_array = np.asarray(generated, dtype=np.uint8)

    product = Image.open(product_image).convert("RGBA")
    product = defringe_rgba(product, radius=2)

    mask_image = Image.open(product_mask).convert("L")
    mask_image = mask_image.resize(
        generated.size,
        Image.Resampling.BILINEAR,
    )

    mask = cv2.medianBlur(
        np.asarray(mask_image, dtype=np.uint8),
        3,
    )
    mask[mask < alpha_threshold] = 0

    product_canvas, bbox = resize_product_to_mask(
        product,
        mask,
        alpha_threshold,
    )
    product_rgb = product_canvas[..., :3].astype(np.float32)

    binary = np.where(
        mask >= alpha_threshold,
        255,
        0,
    ).astype(np.uint8)

    if cv2.countNonZero(binary) == 0:
        raise ValueError("Product mask is empty.")

    core_mask = cv2.erode(
        binary,
        ellipse_kernel(core_erode),
        iterations=1,
    )
    outer_mask = cv2.dilate(
        binary,
        ellipse_kernel(outer_protection),
        iterations=1,
    )

    core_weight = blur_mask(core_mask, core_feather)
    core_weight *= mask.astype(np.float32) / 255.0
    core_weight = np.clip(
        core_weight * core_opacity,
        0.0,
        1.0,
    )

    focus_strength = float(np.clip(focus_strength, 0.0, 1.0))
    effective_background_strength = (
        background_strength + (1.0 - focus_strength) * 0.40
    )
    blur_radius = 1 + int((1.0 - focus_strength) * 10)

    background_refined = cv2.bilateralFilter(
        generated_array,
        d=7,
        sigmaColor=28,
        sigmaSpace=28,
    )
    if blur_radius > 1:
        ksize = blur_radius * 2 + 1
        background_refined = cv2.GaussianBlur(
            background_refined,
            (ksize, ksize),
            sigmaX=blur_radius,
        )
    background_refined = background_refined.astype(np.float32)

    background_mask = 255 - outer_mask
    background_weight = blur_mask(background_mask, 3.0)
    background_weight *= effective_background_strength

    result = generated_array.astype(np.float32)
    result = (
        result * (1.0 - background_weight[..., None])
        + background_refined * background_weight[..., None]
    )

    shadow = create_contact_shadow(
        product_mask=mask,
        bbox=bbox,
        offset=shadow_offset,
        blur=shadow_blur,
        strength=shadow_strength,
    )
    result *= 1.0 - shadow[..., None]

    result = (
        result * (1.0 - core_weight[..., None])
        + product_rgb * core_weight[..., None]
    )
    result = np.clip(result, 0, 255).astype(np.uint8)

    final_path = output_dir / "final_core_ring_refined.png"
    Image.fromarray(result).save(final_path)

    save_diagnostics(
        output_dir,
        {
            "core_mask.png": core_mask,
            "boundary_ring.png": cv2.subtract(
                outer_mask,
                core_mask,
            ),
            "background_refine_mask.png": background_mask,
            "contact_shadow_mask.png": shadow * 255,
        },
    )

    save_metadata(
        output_dir,
        "core_refinement_result.json",
        {
            "input": generated_image,
            "output": final_path,
            "core_erode": core_erode,
            "core_feather": core_feather,
            "core_opacity": core_opacity,
            "background_strength": background_strength,
            "focus_strength": focus_strength,
            "effective_background_strength": effective_background_strength,
        },
    )

    print(f"[DONE] core refinement: {final_path}")
    return final_path