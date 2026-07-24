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

def _enhance_product_details(product_rgb, product_focus):
    product_focus = float(np.clip(product_focus, 0.0, 1.0))
    mean = np.mean(product_rgb, axis=2, keepdims=True)
    contrast = 1.0 + product_focus * 0.35
    enhanced = mean + (product_rgb - mean) * contrast
    saturation = 1.0 + product_focus * 0.22
    boosted = np.clip(mean + (enhanced - mean) * saturation, 0, 255)
    brightness = 1.0 + product_focus * 0.12
    return np.clip(boosted * brightness, 0, 255)


def _dim_background(background, product_focus):
    product_focus = float(np.clip(product_focus, 0.0, 1.0))
    dim_scale = 1.0 - product_focus * 0.28
    return np.clip(background * dim_scale, 0, 255)

def _background_blur_radius(product_focus):
    """Scale background blur continuously from radius 1 to 52."""
    product_focus = float(np.clip(product_focus, 0.0, 1.0))
    return 1 + int(product_focus * 51.0)


def _blur_background_only(image, protected_mask, blur_radius):
    """Blur using background samples without leaking protected pixels."""
    image = image.astype(np.float32)
    if blur_radius <= 1:
        return image

    background = 1.0 - (
        protected_mask.astype(np.float32) / 255.0
    )
    ksize = blur_radius * 2 + 1
    blurred_weight = cv2.GaussianBlur(
        background,
        (ksize, ksize),
        sigmaX=blur_radius,
    )
    weighted_image = cv2.GaussianBlur(
        image * background[..., None],
        (ksize, ksize),
        sigmaX=blur_radius,
    )
    safe_weight = np.maximum(blurred_weight, 1e-6)
    background_only = weighted_image / safe_weight[..., None]
    return np.where(
        (blurred_weight > 1e-6)[..., None],
        background_only,
        image,
    )


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
    outer_protection=14,
    background_strength=0.20,
    product_focus=1.0,
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
        core_weight * core_opacity * (1.0 + product_focus * 0.10),
        0.0,
        1.0,
    )

    product_focus = float(np.clip(product_focus, 0.0, 1.0))
    effective_background_strength = (
        background_strength + product_focus * 0.60
    )
    blur_radius = _background_blur_radius(product_focus)

    background_refined = cv2.bilateralFilter(
        generated_array,
        d=7,
        sigmaColor=28,
        sigmaSpace=28,
    )
    background_refined = _blur_background_only(
        background_refined,
        protected_mask=outer_mask,
        blur_radius=blur_radius,
    )
    background_refined = background_refined.astype(np.float32)
    background_refined = _dim_background(
        background_refined,
        product_focus,
    )

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
            "product_focus": product_focus,
            "effective_background_strength": effective_background_strength,
            "background_blur_radius": blur_radius,
            "background_blur_guard_px": outer_protection,
        },
    )

    print(f"[DONE] core refinement: {final_path}")
    return final_path