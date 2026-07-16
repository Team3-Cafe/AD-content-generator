from adcg.image_utils.masks import create_boundary_masks, create_control_image
from adcg.image_utils.blending import (
    align_product_to_mask, color_match_product, create_identity_weight,
)
from adcg.generation.model_loader import (
    load_identity_pipeline as load_pipeline,
)

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetInpaintPipeline,
)
from PIL import Image
from argparse import Namespace




def load_prompt(path):
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    generation_prompt = data.get("generation_prompt", {})

    prompt = (
        generation_prompt.get("background_prompt")
        or data.get("background_prompt")
        or ""
    )
    negative_prompt = (
        generation_prompt.get("negative_prompt")
        or data.get("negative_prompt")
        or ""
    )

    prompt += (
        ", seamless product boundary, matched ambient lighting, "
        "realistic contact with the surface, natural edge colors, "
        "commercial product photography"
    )
    negative_prompt += (
        ", halo, black outline, white outline, jagged edge, "
        "pasted cutout, floating product, duplicate product, "
        "deformed product"
    )

    return prompt, negative_prompt
















def _run_identity_restoration(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    size = (args.width, args.height)

    base_image = Image.open(args.input_image).convert("RGB")
    base_image = base_image.resize(size, Image.Resampling.LANCZOS)

    mask_image = Image.open(args.product_mask).convert("L")
    mask_image = mask_image.resize(size, Image.Resampling.BILINEAR)

    alpha_mask = np.array(mask_image, dtype=np.uint8)
    alpha_mask = cv2.medianBlur(alpha_mask, 3)

    binary, inpaint_mask_array, boundary_weight = (
        create_boundary_masks(
            alpha_mask,
            threshold=args.alpha_threshold,
            inner_radius=args.inner_radius,
            outer_radius=args.outer_radius,
            blur=args.mask_blur,
            opacity=args.blend_opacity,
        )
    )

    canny = create_control_image(binary)

    prompt, negative_prompt = load_prompt(args.prompt_json)

    pipe = load_pipeline(args)
    generator = torch.Generator(device="cuda").manual_seed(args.seed)

    print("[1/3] ControlNet + MajicMix boundary refinement")

    inpainted = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=base_image,
        mask_image=Image.fromarray(inpaint_mask_array),
        control_image=Image.fromarray(
            cv2.cvtColor(canny, cv2.COLOR_GRAY2RGB)
        ),
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        strength=args.strength,
        controlnet_conditioning_scale=args.controlnet_scale,
        generator=generator,
    ).images[0].convert("RGB")

    base_array = np.array(base_image, dtype=np.float32)
    inpainted_array = np.array(inpainted, dtype=np.float32)

    boundary_weight_3d = boundary_weight[..., None]

    boundary_refined = (
        base_array * (1.0 - boundary_weight_3d)
        + inpainted_array * boundary_weight_3d
    )

    print("[2/3] Automatic original product alignment")

    aligned_product = align_product_to_mask(
        args.product_image,
        binary,
        size,
    )

    aligned_array = np.array(aligned_product, dtype=np.float32)
    product_rgb = aligned_array[..., :3]
    product_alpha = aligned_array[..., 3].astype(np.uint8)

    identity_weight = create_identity_weight(
        product_alpha,
        binary,
        feather=args.identity_feather,
        opacity=args.identity_opacity,
    )

    valid_core = identity_weight >= (
        args.identity_opacity * 0.85
    )

    product_rgb = color_match_product(
        product_rgb,
        boundary_refined,
        valid_core,
        args.color_match,
    )

    print("[3/3] Identity restoration")

    identity_weight_3d = identity_weight[..., None]

    final_array = (
        boundary_refined * (1.0 - identity_weight_3d)
        + product_rgb * identity_weight_3d
    )

    final_array = np.clip(final_array, 0, 255).astype(np.uint8)
    boundary_refined_uint8 = np.clip(
        boundary_refined,
        0,
        255,
    ).astype(np.uint8)

    # Save diagnostics.
    Image.fromarray(binary).save(
        output_dir / "product_binary_mask.png"
    )
    Image.fromarray(inpaint_mask_array).save(
        output_dir / "boundary_inpaint_mask.png"
    )
    Image.fromarray(canny).save(
        output_dir / "product_canny.png"
    )
    Image.fromarray(
        np.clip(boundary_weight * 255, 0, 255).astype(np.uint8)
    ).save(output_dir / "boundary_blend_weight.png")
    Image.fromarray(
        np.clip(identity_weight * 255, 0, 255).astype(np.uint8)
    ).save(output_dir / "identity_restore_weight.png")

    aligned_product.save(
        output_dir / "aligned_original_product.png"
    )
    inpainted.save(
        output_dir / "boundary_inpaint_raw.png"
    )
    Image.fromarray(boundary_refined_uint8).save(
        output_dir / "boundary_refined.png"
    )
    Image.fromarray(final_array).save(
        output_dir / "final_identity_restored.png"
    )

    final_path = output_dir / "final_identity_restored.png"
    Image.fromarray(final_array).save(final_path)

    print("\n[Done]")
    print("Final:", final_path)

    return final_path


IDENTITY_DEFAULTS = {
    "base_model": "digiplay/majicMIX_realistic_v7",
    "controlnet_model": "lllyasviel/control_v11p_sd15_canny",
    "width": 512,
    "height": 512,
    "alpha_threshold": 45,
    "inner_radius": 1.0,
    "outer_radius": 10.0,
    "mask_blur": 1.0,
    "blend_opacity": 0.85,
    "identity_feather": 7.0,
    "identity_opacity": 0.92,
    "color_match": 0.25,
    "steps": 30,
    "strength": 0.38,
    "guidance_scale": 6.5,
    "controlnet_scale": 0.60,
    "seed": 42,
    "cpu_offload": False,
}


def run_identity_restoration(
    input_image,
    product_image,
    product_mask,
    prompt_json,
    output_dir,
    **overrides,
):
    unknown = set(overrides) - set(IDENTITY_DEFAULTS)
    if unknown:
        raise TypeError(f"지원하지 않는 복원 옵션: {sorted(unknown)}")

    config = {
        **IDENTITY_DEFAULTS,
        **overrides,
        "input_image": str(input_image),
        "product_image": str(product_image),
        "product_mask": str(product_mask),
        "prompt_json": str(prompt_json),
        "output_dir": str(output_dir),
    }

    return _run_identity_restoration(Namespace(**config))
