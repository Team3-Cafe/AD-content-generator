import time

import torch


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
    generator = torch.Generator(
        device="cuda"
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