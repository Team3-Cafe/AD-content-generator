import time

import torch

from adcg.brand_focus import (
    EVERYDAY_BACKGROUND_ANCHOR,
    STUDIO_BACKGROUND_ANCHOR,
    blend_prompt_embeddings,
    brand_blend_weight,
)
from adcg.prompt_tokens import (
    fit_clip_prompt,
    fit_clip_prompt_parts,
)


LEGACY_ENVIRONMENT_FIELD_PRIORITY = (
    "location",
    "surface",
    "camera",
    "lighting",
    "composition",
    "copy_space",
    "atmosphere",
)

PRODUCT_ONLY_ANCHOR = (
    "unattended product-only scene, "
    "empty surroundings, isolated subject"
)


def _prepare_control_images(control_image):
    """
    단일 ControlNet이면 PIL Image를 반환하고,
    Multi-ControlNet이면 PIL Image 목록을 반환한다.
    """
    if isinstance(control_image, (list, tuple)):
        return [
            image.convert("RGB")
            for image in control_image
        ]

    return control_image.convert("RGB")


def _prepare_control_scales(controlnet_scale):
    """
    단일 scale 또는 Multi-ControlNet용 scale 목록을
    float 형태로 정규화한다.
    """
    if isinstance(controlnet_scale, (list, tuple)):
        return [
            float(scale)
            for scale in controlnet_scale
        ]

    return float(controlnet_scale)


def _validate_control_inputs(
    control_images,
    control_scales,
):
    if not isinstance(control_images, list):
        return control_images, control_scales

    if not isinstance(control_scales, list):
        control_scales = [
            float(control_scales)
            for _ in control_images
        ]

    if len(control_images) != len(control_scales):
        raise ValueError(
            "ControlNet 이미지 수와 scale 수가 다릅니다: "
            f"{len(control_images)} != "
            f"{len(control_scales)}"
        )

    return control_images, control_scales


def _fit_background_prompt(
    tokenizer,
    fallback_prompt,
    prompt_parts,
    anchor,
    label,
):
    required_anchor = (
        f"{PRODUCT_ONLY_ANCHOR}, {anchor}"
    )

    if isinstance(prompt_parts, dict):
        parts = [
            prompt_parts.get(field)
            for field
            in LEGACY_ENVIRONMENT_FIELD_PRIORITY
        ]
    elif isinstance(prompt_parts, list):
        parts = prompt_parts
    else:
        parts = None

    if parts is not None:
        return fit_clip_prompt_parts(
            tokenizer,
            parts,
            label=label,
            required_prefix=required_anchor,
        )

    prompt = fit_clip_prompt(
        tokenizer,
        fallback_prompt,
        label=label,
        required_prefix=PRODUCT_ONLY_ANCHOR,
    )

    return prompt, {
        "prompt": prompt,
        "token_count": None,
        "target_tokens": 77,
        "model_limit": 77,
        "raw_token_count": None,
        "selected_clauses": [],
        "dropped_clauses": [],
    }


def _encode_prompt(
    pipe,
    prompt,
    negative_prompt,
    guidance_scale,
):
    device = getattr(
        pipe,
        "_execution_device",
        None,
    )

    if device is None:
        device = getattr(
            pipe,
            "device",
            None,
        )

    return pipe.encode_prompt(
        prompt=prompt,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=(
            guidance_scale > 1.0
        ),
        negative_prompt=negative_prompt,
    )


def run_conditioned_inference(
    pipe,
    prompt,
    negative_prompt,
    condition_canvas,
    inpaint_mask,
    control_image,
    width,
    height,
    steps,
    guidance_scale,
    strength,
    controlnet_scale,
    seed,
    everyday_prompt=None,
    studio_prompt=None,
    everyday_prompt_parts=None,
    studio_prompt_parts=None,
    brand_focus=0.5,
):
    everyday_prompt, everyday_token_data = (
        _fit_background_prompt(
            pipe.tokenizer,
            everyday_prompt or prompt,
            everyday_prompt_parts,
            EVERYDAY_BACKGROUND_ANCHOR,
            "generation everyday positive",
        )
    )

    studio_prompt, studio_token_data = (
        _fit_background_prompt(
            pipe.tokenizer,
            studio_prompt or prompt,
            studio_prompt_parts,
            STUDIO_BACKGROUND_ANCHOR,
            "generation studio positive",
        )
    )

    negative_prompt = fit_clip_prompt(
        pipe.tokenizer,
        negative_prompt,
        label="generation negative",
    )

    blend_weight = brand_blend_weight(
        brand_focus
    )

    if blend_weight <= 0.0:
        prompt_arguments = {
            "prompt": everyday_prompt,
            "negative_prompt": negative_prompt,
        }
        blend_mode = "everyday endpoint"

    elif blend_weight >= 1.0:
        prompt_arguments = {
            "prompt": studio_prompt,
            "negative_prompt": negative_prompt,
        }
        blend_mode = "studio endpoint"

    else:
        everyday_embeds, negative_embeds = (
            _encode_prompt(
                pipe,
                everyday_prompt,
                negative_prompt,
                guidance_scale,
            )
        )

        studio_embeds, _ = _encode_prompt(
            pipe,
            studio_prompt,
            negative_prompt,
            guidance_scale,
        )

        prompt_arguments = {
            "prompt_embeds": (
                blend_prompt_embeddings(
                    everyday_embeds,
                    studio_embeds,
                    brand_focus,
                )
            ),
            "negative_prompt_embeds": (
                negative_embeds
            ),
        }

        blend_mode = "CLIP embedding blend"

    print(
        f"[Brand focus] "
        f"input={float(brand_focus):.2f}, "
        f"studio_weight={blend_weight:.3f}, "
        f"mode={blend_mode}"
    )

    prepared_control_images = (
        _prepare_control_images(
            control_image
        )
    )

    prepared_control_scales = (
        _prepare_control_scales(
            controlnet_scale
        )
    )

    (
        prepared_control_images,
        prepared_control_scales,
    ) = _validate_control_inputs(
        prepared_control_images,
        prepared_control_scales,
    )

    if isinstance(
        prepared_control_images,
        list,
    ):
        print(
            "[ControlNet] multi-control mode: "
            f"{len(prepared_control_images)} controls, "
            f"scales={prepared_control_scales}"
        )
    else:
        print(
            "[ControlNet] single-control mode: "
            f"scale={prepared_control_scales}"
        )

    generator_device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    generator = torch.Generator(
        device=generator_device
    ).manual_seed(int(seed))

    start_time = time.perf_counter()

    image = pipe(
        **prompt_arguments,
        image=condition_canvas.convert("RGB"),
        mask_image=inpaint_mask.convert("L"),
        control_image=prepared_control_images,
        width=int(width),
        height=int(height),
        num_inference_steps=int(steps),
        guidance_scale=float(guidance_scale),
        strength=float(strength),
        controlnet_conditioning_scale=(
            prepared_control_scales
        ),
        generator=generator,
    ).images[0].convert("RGB")

    elapsed = (
        time.perf_counter() - start_time
    )

    return image, elapsed, {
        "everyday": everyday_token_data,
        "studio": studio_token_data,
        "negative_prompt": negative_prompt,
        "brand_focus": float(brand_focus),
        "studio_weight": float(blend_weight),
        "blend_mode": blend_mode,
        "controlnet_scales": (
            prepared_control_scales
        ),
    }