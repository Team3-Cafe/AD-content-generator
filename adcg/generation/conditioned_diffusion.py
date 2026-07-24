import json
from pathlib import Path

import torch

from adcg.brand_focus import brand_blend_weight, select_background_prompt

from .canvas import (
    create_condition_canvas,
    load_metadata,
    load_product_cutout,
)
from .control import create_canny_control
from .inference import run_conditioned_inference
from .inpaint_mask import create_background_inpaint_mask
from .model_loader import load_generation_pipeline
from .result import save_generation_result


GENERATION_DEFAULTS = {
    "base_model": "digiplay/majicMIX_realistic_v7",
    "controlnet_model": "lllyasviel/control_v11p_sd15_canny",
    "width": 576,
    "height": 1024,
    "steps": 35,
    "guidance_scale": 8.0,
    "strength": 1.0,
    "controlnet_scale": 0.7,
    "layout_mode": "layout",
    "product_x": None,
    "product_y": None,
    "product_scale": None,
    "mask_margin": 4,
    "mask_blur": 2.0,
    "contact_ratio": 0.06,
    "canny_low": 100,
    "canny_high": 200,
    "edge_suppression": 3,
    "cutout_mode": "auto",
    "alpha_threshold": 4,
    "preprocess_metadata": None,
    "seed": 42,
    "cpu_offload": False,
}


def load_prompt_data(prompt_json):
    prompt_path = Path(prompt_json)

    if not prompt_path.exists():
        raise FileNotFoundError(
            f"프롬프트 JSON을 찾을 수 없습니다: {prompt_path}"
        )

    data = json.loads(
        prompt_path.read_text(encoding="utf-8")
    )

    generation_prompt = data.get(
        "generation_prompt",
        {},
    )

    background_prompt = str(
        generation_prompt.get("background_prompt")
        or data.get("background_prompt")
        or ""
    ).strip()
    everyday_prompt = str(
        generation_prompt.get("everyday_background_prompt")
        or background_prompt
    ).strip()
    studio_prompt = str(
        generation_prompt.get("studio_background_prompt")
        or background_prompt
    ).strip()
    everyday_prompt_parts = generation_prompt.get(
        "everyday_prompt_parts"
    )
    studio_prompt_parts = generation_prompt.get(
        "studio_prompt_parts"
    )
    # Backward compatibility for prompt JSON created by the previous schema.
    if everyday_prompt_parts is None:
        everyday_prompt_parts = generation_prompt.get(
            "everyday_environment"
        )
    if studio_prompt_parts is None:
        studio_prompt_parts = generation_prompt.get(
            "studio_environment"
        )

    negative_prompt = str(
        generation_prompt.get("negative_prompt")
        or data.get("negative_prompt")
        or ""
    ).strip()
    controls = data.get("controls") or {}
    brand_focus = float(controls.get("brand_focus", 0.5))

    if not everyday_prompt or not studio_prompt:
        raise ValueError("background prompt endpoint is empty.")

    return (
        data,
        background_prompt or everyday_prompt,
        everyday_prompt,
        studio_prompt,
        everyday_prompt_parts,
        studio_prompt_parts,
        negative_prompt,
        brand_focus,
    )


def _resolve_layout(config, prompt_data):
    layout = prompt_data.get("layout", {})

    def resolve(name, default):
        value = config.get(name)

        if value is not None:
            return float(value)

        return float(layout.get(name, default))

    return {
        "product_x": resolve("product_x", 0.50),
        "product_y": resolve("product_y", 0.70),
        # None selects aspect-aware automatic sizing. Numeric values are
        # interpreted against the canvas short axis, never the VLM JSON.
        "product_scale": (
            None
            if config.get("product_scale") is None
            else float(config["product_scale"])
        ),
    }


def _run_generation(config, pipe=None):
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    (
        prompt_data,
        background_prompt,
        everyday_prompt,
        studio_prompt,
        everyday_prompt_parts,
        studio_prompt_parts,
        negative_prompt,
        brand_focus,
    ) = load_prompt_data(config["prompt_json"])

    metadata = load_metadata(
        config.get("preprocess_metadata")
    )
    truncation = metadata.get("truncation", {})
    layout = _resolve_layout(config, prompt_data)

    print("[상품 누끼 로드]")
    product = load_product_cutout(
        image_path=config["product_image"],
        mode=config["cutout_mode"],
        alpha_threshold=config["alpha_threshold"],
    )

    canvas_result = create_condition_canvas(
        product=product,
        width=config["width"],
        height=config["height"],
        layout_mode=config["layout_mode"],
        product_x=layout["product_x"],
        product_y=layout["product_y"],
        product_scale=layout["product_scale"],
        metadata=metadata,
    )

    placement = canvas_result["placement"]
    layout["requested_product_scale"] = layout["product_scale"]
    layout["product_scale"] = placement["effective_product_scale"]
    print(
        f"[Layout] product_scale="
        f"{placement['requested_product_scale']!r} "
        f"-> effective={placement['effective_product_scale']:.4f} "
        f"({placement['sizing_mode']})"
    )

    inpaint_mask = create_background_inpaint_mask(
        product_mask=canvas_result["product_mask"],
        boundary_width=config["mask_margin"],
        blur=config["mask_blur"],
        contact_ratio=config["contact_ratio"],
    )

    control_image = create_canny_control(
        product_layer=canvas_result["product_layer"],
        product_mask=canvas_result["product_mask"],
        low_threshold=config["canny_low"],
        high_threshold=config["canny_high"],
        contact_ratio=config["contact_ratio"],
        truncation=truncation,
        edge_suppression=config["edge_suppression"],
    )

    owns_pipe = pipe is None

    if owns_pipe:
        pipe = load_generation_pipeline(
            base_model=config["base_model"],
            controlnet_model=config["controlnet_model"],
            cpu_offload=config["cpu_offload"],
        )

    print("[조건부 이미지 생성 시작]")
    try:
        generated_image, generation_time, prompt_token_data = (
            run_conditioned_inference(
                pipe=pipe,
                prompt=background_prompt,
                everyday_prompt=everyday_prompt,
                studio_prompt=studio_prompt,
                everyday_prompt_parts=everyday_prompt_parts,
                studio_prompt_parts=studio_prompt_parts,
                brand_focus=brand_focus,
                negative_prompt=negative_prompt,
                condition_canvas=canvas_result[
                    "condition_canvas"
                ],
                inpaint_mask=inpaint_mask,
                control_image=control_image,
                width=config["width"],
                height=config["height"],
                steps=config["steps"],
                guidance_scale=config["guidance_scale"],
                strength=config["strength"],
                controlnet_scale=config["controlnet_scale"],
                seed=config["seed"],
            )
        )
    finally:
        if owns_pipe:
            del pipe
            torch.cuda.empty_cache()

    experiment_data = {
        "base_model": config["base_model"],
        "controlnet_model": config["controlnet_model"],
        "product_image": config["product_image"],
        "prompt_json": config["prompt_json"],
        "background_prompt": background_prompt,
        "everyday_background_prompt": everyday_prompt,
        "studio_background_prompt": studio_prompt,
        "everyday_prompt_parts": everyday_prompt_parts,
        "studio_prompt_parts": studio_prompt_parts,
        "effective_everyday_background_prompt": prompt_token_data[
            "everyday"
        ]["prompt"],
        "effective_studio_background_prompt": prompt_token_data[
            "studio"
        ]["prompt"],
        "effective_background_prompt": select_background_prompt(
            prompt_token_data["everyday"]["prompt"],
            prompt_token_data["studio"]["prompt"],
            brand_focus,
        ),
        "prompt_tokenization": prompt_token_data,
        "brand_focus": brand_focus,
        "brand_blend_weight": brand_blend_weight(brand_focus),
        "negative_prompt": negative_prompt,
        "layout": {
            **layout,
            "layout_mode": config["layout_mode"],
            "placement_pixels": canvas_result["placement"],
        },
        "truncation": truncation,
        "generation": {
            "width": config["width"],
            "height": config["height"],
            "steps": config["steps"],
            "guidance_scale": config["guidance_scale"],
            "strength": config["strength"],
            "controlnet_scale": config["controlnet_scale"],
            "mask_margin": config["mask_margin"],
            "mask_blur": config["mask_blur"],
            "contact_ratio": config["contact_ratio"],
            "seed": config["seed"],
            "generation_time_sec": generation_time,
        },
    }

    outputs = save_generation_result(
        output_dir=output_dir,
        product=product,
        canvas_result=canvas_result,
        inpaint_mask=inpaint_mask,
        control_image=control_image,
        generated_image=generated_image,
        experiment_data=experiment_data,
    )

    print(f"[DONE] generated: {outputs['image']}")
    print(f"[TIME] {generation_time:.2f}초")

    return outputs


def run_generation(
    product_image,
    prompt_json,
    output_dir,
    pipe=None,
    **overrides,
):
    unknown = set(overrides) - set(GENERATION_DEFAULTS)

    if unknown:
        raise TypeError(
            f"지원하지 않는 생성 옵션: {sorted(unknown)}"
        )

    config = {
        **GENERATION_DEFAULTS,
        **overrides,
        "product_image": str(product_image),
        "prompt_json": str(prompt_json),
        "output_dir": str(output_dir),
    }

    return _run_generation(config, pipe=pipe)
