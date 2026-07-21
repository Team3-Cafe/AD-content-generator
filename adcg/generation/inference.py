import time

import torch

from adcg.prompt_tokens import fit_clip_prompt


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
):
    prompt = fit_clip_prompt(
        pipe.tokenizer,
        prompt,
        label="generation positive",
    )
    negative_prompt = fit_clip_prompt(
        pipe.tokenizer,
        negative_prompt,
        label="generation negative",
    )

    generator_device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(
        device=generator_device
    ).manual_seed(seed)

    start_time = time.perf_counter()

    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
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