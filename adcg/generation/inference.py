import time

import torch

from adcg.brand_focus import (
    blend_prompt_embeddings,
    brand_blend_weight,
)
from adcg.prompt_tokens import fit_clip_prompt


def _encode_prompt(pipe, prompt, negative_prompt, guidance_scale):
    device = getattr(pipe, "_execution_device", None)
    if device is None:
        device = getattr(pipe, "device", None)

    return pipe.encode_prompt(
        prompt=prompt,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=guidance_scale > 1.0,
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
    brand_focus=0.5,
):
    everyday_prompt = fit_clip_prompt(
        pipe.tokenizer,
        everyday_prompt or prompt,
        label="generation everyday positive",
    )
    studio_prompt = fit_clip_prompt(
        pipe.tokenizer,
        studio_prompt or prompt,
        label="generation studio positive",
    )
    negative_prompt = fit_clip_prompt(
        pipe.tokenizer,
        negative_prompt,
        label="generation negative",
    )

    blend_weight = brand_blend_weight(brand_focus)
    prompt_arguments = {}
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
        everyday_embeds, negative_embeds = _encode_prompt(
            pipe,
            everyday_prompt,
            negative_prompt,
            guidance_scale,
        )
        studio_embeds, _ = _encode_prompt(
            pipe,
            studio_prompt,
            negative_prompt,
            guidance_scale,
        )
        prompt_arguments = {
            "prompt_embeds": blend_prompt_embeddings(
                everyday_embeds,
                studio_embeds,
                brand_focus,
            ),
            "negative_prompt_embeds": negative_embeds,
        }
        blend_mode = "CLIP embedding blend"

    print(
        f"[Brand focus] input={float(brand_focus):.2f}, "
        f"studio_weight={blend_weight:.3f}, mode={blend_mode}"
    )

    generator_device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(
        device=generator_device
    ).manual_seed(seed)

    start_time = time.perf_counter()

    image = pipe(
        **prompt_arguments,
        image=condition_canvas.convert("RGB"),
        mask_image=inpaint_mask.convert("L"),
        control_image=control_image.convert("RGB"),
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        strength=strength,
        controlnet_conditioning_scale=controlnet_scale,
        generator=generator,
    ).images[0].convert("RGB")

    elapsed = time.perf_counter() - start_time

    return image, elapsed