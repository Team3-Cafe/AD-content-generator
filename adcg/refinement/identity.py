from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from adcg.generation.model_loader import (
    load_generation_pipeline,
)
from adcg.prompt_tokens import fit_clip_prompt
from adcg.image_utils.blending import (
    align_product_to_mask,
    color_match_product,
    create_identity_weight,
)
from adcg.image_utils.masks import (
    create_boundary_masks,
    create_control_image,
)
from .diagnostics import (
    prepare_output_dir,
    save_diagnostics,
    save_metadata,
)
from .prompt import (
    NEGATIVE_REQUIRED,
    POSITIVE_REQUIRED,
    load_refinement_prompt,
)


def _dim_boundary(boundary, product_focus):
    product_focus = float(np.clip(product_focus, 0.0, 1.0))
    dim_scale = 1.0 - product_focus * 0.18
    return np.clip(boundary * dim_scale, 0, 255)


def _enhance_identity_product(product_rgb, product_focus):
    product_focus = float(np.clip(product_focus, 0.0, 1.0))
    mean = np.mean(product_rgb, axis=2, keepdims=True)
    contrast = 1.0 + product_focus * 0.28
    enhanced = mean + (product_rgb - mean) * contrast
    saturation = 1.0 + product_focus * 0.15
    enhanced = np.clip(mean + (enhanced - mean) * saturation, 0, 255)
    brightness = 1.0 + product_focus * 0.12
    return np.clip(enhanced * brightness, 0, 255)


def run_identity_restoration(
    input_image,
    product_image,
    product_mask,
    prompt_json,
    output_dir="outputs/refinement/identity",
    base_model="digiplay/majicMIX_realistic_v7",
    controlnet_model="lllyasviel/control_v11p_sd15_canny",
    width=512,
    height=768,
    alpha_threshold=45,
    inner_radius=1.0,
    outer_radius=10.0,
    mask_blur=1.0,
    blend_opacity=0.85,
    identity_feather=7.0,
    identity_opacity=0.92,
    color_match=0.25,
    steps=30,
    strength=0.38,
    guidance_scale=6.5,
    controlnet_scale=0.60,
    seed=42,
    product_focus=1.0,
    cpu_offload=False,
    pipe=None,
):
    output_dir = prepare_output_dir(output_dir)
    size = (width, height)

    base_image = Image.open(input_image).convert("RGB")
    base_image = base_image.resize(
        size,
        Image.Resampling.LANCZOS,
    )

    mask_image = Image.open(product_mask).convert("L")
    mask_image = mask_image.resize(
        size,
        Image.Resampling.BILINEAR,
    )

    alpha_mask = cv2.medianBlur(
        np.asarray(mask_image, dtype=np.uint8),
        3,
    )

    binary, inpaint_mask, boundary_weight = (
        create_boundary_masks(
            alpha_mask=alpha_mask,
            threshold=alpha_threshold,
            inner_radius=inner_radius,
            outer_radius=outer_radius,
            blur=mask_blur,
            opacity=blend_opacity,
        )
    )

    canny = create_control_image(binary)
    control_image = Image.fromarray(
        cv2.cvtColor(canny, cv2.COLOR_GRAY2RGB)
    )

    prompt, negative_prompt = load_refinement_prompt(
        prompt_json
    )

    owns_pipe = pipe is None

    if owns_pipe:
        pipe = load_generation_pipeline(
            base_model=base_model,
            controlnet_model=controlnet_model,
            cpu_offload=cpu_offload,
        )

    prompt = fit_clip_prompt(
        pipe.tokenizer,
        prompt,
        label="identity positive",
        required_prefix=POSITIVE_REQUIRED,
    )
    negative_prompt = fit_clip_prompt(
        pipe.tokenizer,
        negative_prompt,
        label="identity negative",
        required_prefix=NEGATIVE_REQUIRED,
    )
    generator_device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(
        device=generator_device
    ).manual_seed(seed)

    try:
        inpainted = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=base_image,
            mask_image=Image.fromarray(inpaint_mask),
            control_image=control_image,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            strength=strength,
            controlnet_conditioning_scale=controlnet_scale,
            generator=generator,
        ).images[0].convert("RGB")
    finally:
        if owns_pipe:
            del pipe
            torch.cuda.empty_cache()

    base_array = np.asarray(base_image, dtype=np.float32)
    inpainted_array = np.asarray(inpainted, dtype=np.float32)

    boundary_refined = (
        base_array * (1.0 - boundary_weight[..., None])
        + inpainted_array * boundary_weight[..., None]
    )
    boundary_refined = _dim_boundary(boundary_refined, product_focus)

    aligned_product = align_product_to_mask(
        product_path=product_image,
        binary_mask=binary,
        canvas_size=size,
    )
    aligned_array = np.asarray(
        aligned_product,
        dtype=np.float32,
    )

    product_rgb = aligned_array[..., :3]
    product_alpha = aligned_array[..., 3].astype(np.uint8)

    identity_weight = create_identity_weight(
        aligned_alpha=product_alpha,
        binary_mask=binary,
        feather=identity_feather,
        opacity=identity_opacity,
    )

    valid_core = identity_weight >= (
        identity_opacity * 0.85
    )
    product_rgb = color_match_product(
        product_rgb=product_rgb,
        target_rgb=boundary_refined,
        valid_mask=valid_core,
        amount=color_match,
    )
    product_rgb = _enhance_identity_product(
        product_rgb,
        product_focus,
    )

    final = (
        boundary_refined
        * (1.0 - identity_weight[..., None])
        + product_rgb * identity_weight[..., None]
    )
    final = np.clip(final, 0, 255).astype(np.uint8)

    final_path = output_dir / "final_identity_restored.png"
    Image.fromarray(final).save(final_path)

    save_diagnostics(
        output_dir,
        {
            "product_binary_mask.png": binary,
            "boundary_inpaint_mask.png": inpaint_mask,
            "product_canny.png": canny,
            "boundary_blend_weight.png": (
                boundary_weight * 255
            ),
            "identity_restore_weight.png": (
                identity_weight * 255
            ),
            "aligned_original_product.png": aligned_product,
            "boundary_inpaint_raw.png": inpainted,
            "boundary_refined.png": boundary_refined,
        },
    )

    save_metadata(
        output_dir,
        "identity_refinement_result.json",
        {
            "input": input_image,
            "output": final_path,
            "base_model": base_model,
            "controlnet_model": controlnet_model,
            "steps": steps,
            "strength": strength,
            "controlnet_scale": controlnet_scale,
            "identity_opacity": identity_opacity,
            "product_focus": product_focus,
            "seed": seed,
        },
    )

    print(f"[DONE] identity restoration: {final_path}")
    return final_path
