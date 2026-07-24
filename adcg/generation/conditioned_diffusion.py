import json
import time
from pathlib import Path

import torch
from PIL import Image

from adcg.prompting.schema import normalize_negative_prompt

from .canvas import create_condition_canvas
from .control import (
    create_canny_control,
    create_depth_control,
)
from .inference import run_conditioned_inference
from .inpaint_mask import (
    create_background_inpaint_mask,
)
from .model_loader import load_generation_pipeline
from .result import save_generation_result


GENERATION_DEFAULTS = {
    "base_model": "digiplay/majicMIX_realistic_v7",
    "controlnet_model": (
        "lllyasviel/control_v11p_sd15_canny"
    ),
    "depth_controlnet_model": (
        "lllyasviel/control_v11f1p_sd15_depth"
    ),
    "depth_estimator_model": (
        "Intel/dpt-hybrid-midas"
    ),
    "depth_device": "cpu",
    "width": 512,
    "height": 512,
    "steps": 35,
    "guidance_scale": 6.5,
    "strength": 0.80,
    "controlnet_scale": 0.30,
    "depth_controlnet_scale": 0.30,
    "brand_focus": 0.50,
    "everyday_prompt": None,
    "studio_prompt": None,
    "everyday_prompt_parts": None,
    "studio_prompt_parts": None,
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


def _load_prompt_json(prompt_json):
    prompt_json = Path(prompt_json)

    if not prompt_json.exists():
        raise FileNotFoundError(
            f"Prompt JSON not found: {prompt_json}"
        )

    return json.loads(
        prompt_json.read_text(encoding="utf-8")
    )


def _load_product(product_image):
    product_image = Path(product_image)

    if not product_image.exists():
        raise FileNotFoundError(
            f"Product image not found: {product_image}"
        )

    return Image.open(
        product_image
    ).convert("RGBA")


def _resolve_number(
    config,
    layout,
    key,
    default,
):
    override = config.get(key)

    if override is not None:
        return float(override)

    layout_value = layout.get(key)

    if layout_value is not None:
        return float(layout_value)

    return float(default)


def _resolve_layout(
    config,
    prompt_data,
    product,
):
    layout = dict(
        prompt_data.get("layout") or {}
    )

    if config["layout_mode"] == "preserve":
        product_scale = min(
            product.width / int(config["width"]),
            product.height / int(config["height"]),
            1.0,
        )

        return {
            "product_x": 0.50,
            "product_y": 0.50,
            "product_scale": product_scale,
        }

    return {
        "product_x": _resolve_number(
            config,
            layout,
            "product_x",
            0.50,
        ),
        "product_y": _resolve_number(
            config,
            layout,
            "product_y",
            0.70,
        ),
        "product_scale": _resolve_number(
            config,
            layout,
            "product_scale",
            0.45,
        ),
    }


def _pipe_control_count(pipe):
    controlnet = getattr(
        pipe,
        "controlnet",
        None,
    )

    networks = getattr(
        controlnet,
        "nets",
        None,
    )

    if networks is not None:
        return len(networks)

    if controlnet is not None:
        return 1

    return 0


def _build_config(options):
    config = dict(GENERATION_DEFAULTS)

    unknown_options = (
        set(options) - set(config)
    )

    if unknown_options:
        raise TypeError(
            "지원하지 않는 생성 옵션: "
            f"{sorted(unknown_options)}"
        )

    config.update(options)

    return config


def _first_value(*values):
    for value in values:
        if value is not None:
            return value

    return None


def _resolve_brand_prompts(
    config,
    generation_prompt,
):
    everyday_prompt = _first_value(
        config.get("everyday_prompt"),
        generation_prompt.get(
            "everyday_prompt"
        ),
        generation_prompt.get(
            "everyday_background_prompt"
        ),
    )

    studio_prompt = _first_value(
        config.get("studio_prompt"),
        generation_prompt.get(
            "studio_prompt"
        ),
        generation_prompt.get(
            "studio_background_prompt"
        ),
    )

    everyday_prompt_parts = _first_value(
        config.get("everyday_prompt_parts"),
        generation_prompt.get(
            "everyday_prompt_parts"
        ),
        generation_prompt.get(
            "everyday_background_prompt_parts"
        ),
    )

    studio_prompt_parts = _first_value(
        config.get("studio_prompt_parts"),
        generation_prompt.get(
            "studio_prompt_parts"
        ),
        generation_prompt.get(
            "studio_background_prompt_parts"
        ),
    )

    return {
        "everyday_prompt": everyday_prompt,
        "studio_prompt": studio_prompt,
        "everyday_prompt_parts": (
            everyday_prompt_parts
        ),
        "studio_prompt_parts": (
            studio_prompt_parts
        ),
    }


def _run_generation(config):
    total_started_at = time.perf_counter()

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    prompt_data = _load_prompt_json(
        config["prompt_json"]
    )

    generation_prompt = dict(
        prompt_data.get("generation_prompt") or {}
    )

    prompt = str(
        generation_prompt.get(
            "background_prompt"
        ) or ""
    ).strip()

    negative_prompt = normalize_negative_prompt(
        generation_prompt.get("negative_prompt")
    )

    brand_prompts = _resolve_brand_prompts(
        config=config,
        generation_prompt=generation_prompt,
    )

    brand_focus = float(
        config.get("brand_focus", 0.50)
    )
    brand_focus = max(
        0.0,
        min(1.0, brand_focus),
    )

    product = _load_product(
        config["product_image"]
    )

    layout = _resolve_layout(
        config=config,
        prompt_data=prompt_data,
        product=product,
    )

    print("[Create condition canvas]")

    canvas_result = create_condition_canvas(
        product=product,
        width=int(config["width"]),
        height=int(config["height"]),
        product_x=layout["product_x"],
        product_y=layout["product_y"],
        product_scale=layout["product_scale"],
        alpha_threshold=int(
            config["alpha_threshold"]
        ),
    )

    inpaint_mask = (
        create_background_inpaint_mask(
            product_mask=canvas_result[
                "product_mask"
            ],
            margin=int(config["mask_margin"]),
            blur=float(config["mask_blur"]),
            contact_ratio=float(
                config["contact_ratio"]
            ),
        )
    )

    canny_control = create_canny_control(
        product_layer=canvas_result[
            "product_layer"
        ],
        alpha_mask=canvas_result[
            "product_mask"
        ],
        low_threshold=int(
            config["canny_low"]
        ),
        high_threshold=int(
            config["canny_high"]
        ),
        suppress_radius=int(
            config["edge_suppression"]
        ),
    )

    control_images = [canny_control]
    control_scales = [
        float(config["controlnet_scale"])
    ]

    depth_control = None
    depth_controlnet_model = config.get(
        "depth_controlnet_model"
    )

    if depth_controlnet_model:
        print("[Create depth control]")

        depth_control = create_depth_control(
            image=canvas_result[
                "condition_canvas"
            ],
            model_id=config[
                "depth_estimator_model"
            ],
            device=config["depth_device"],
        )

        control_images.append(
            depth_control
        )
        control_scales.append(
            float(
                config[
                    "depth_controlnet_scale"
                ]
            )
        )

    pipe = config.get("pipe")
    owns_pipe = pipe is None

    required_control_count = len(
        control_images
    )

    if (
        pipe is not None
        and _pipe_control_count(pipe)
        != required_control_count
    ):
        print(
            "[WARN] Preloaded pipeline has a "
            "different ControlNet configuration. "
            "Reloading the generation pipeline."
        )

        pipe = None
        owns_pipe = True

    if pipe is None:
        pipe = load_generation_pipeline(
            base_model=config["base_model"],
            controlnet_model=config[
                "controlnet_model"
            ],
            depth_controlnet_model=(
                depth_controlnet_model
            ),
            cpu_offload=bool(
                config["cpu_offload"]
            ),
        )

    if len(control_images) == 1:
        inference_control = (
            control_images[0]
        )
        inference_scales = (
            control_scales[0]
        )
    else:
        inference_control = control_images
        inference_scales = control_scales

    (
        generated_image,
        inference_elapsed,
        prompt_token_data,
    ) = run_conditioned_inference(
        pipe=pipe,
        prompt=prompt,
        negative_prompt=negative_prompt,
        condition_canvas=canvas_result[
            "condition_canvas"
        ],
        inpaint_mask=inpaint_mask,
        control_image=inference_control,
        width=int(config["width"]),
        height=int(config["height"]),
        steps=int(config["steps"]),
        guidance_scale=float(
            config["guidance_scale"]
        ),
        strength=float(config["strength"]),
        controlnet_scale=inference_scales,
        seed=int(config["seed"]),
        everyday_prompt=brand_prompts[
            "everyday_prompt"
        ],
        studio_prompt=brand_prompts[
            "studio_prompt"
        ],
        everyday_prompt_parts=brand_prompts[
            "everyday_prompt_parts"
        ],
        studio_prompt_parts=brand_prompts[
            "studio_prompt_parts"
        ],
        brand_focus=brand_focus,
    )

    total_elapsed = (
        time.perf_counter()
        - total_started_at
    )

    experiment_data = {
        "base_model": config["base_model"],
        "controlnet_model": config[
            "controlnet_model"
        ],
        "depth_controlnet_model": (
            depth_controlnet_model
        ),
        "depth_estimator_model": (
            config["depth_estimator_model"]
            if depth_controlnet_model
            else None
        ),
        "input_product": str(
            Path(config["product_image"])
        ),
        "prompt_json": str(
            Path(config["prompt_json"])
        ),
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "brand_focus": brand_focus,
        "brand_prompts": brand_prompts,
        "prompt_token_data": (
            prompt_token_data
        ),
        "width": int(config["width"]),
        "height": int(config["height"]),
        "steps": int(config["steps"]),
        "guidance_scale": float(
            config["guidance_scale"]
        ),
        "strength": float(
            config["strength"]
        ),
        "controlnet_scale": float(
            config["controlnet_scale"]
        ),
        "depth_controlnet_scale": (
            float(
                config[
                    "depth_controlnet_scale"
                ]
            )
            if depth_controlnet_model
            else None
        ),
        "layout_mode": config[
            "layout_mode"
        ],
        "layout": layout,
        "seed": int(config["seed"]),
        "inference_elapsed_seconds": round(
            inference_elapsed,
            3,
        ),
        "total_elapsed_seconds": round(
            total_elapsed,
            3,
        ),
    }

    result = save_generation_result(
        output_dir=output_dir,
        product=product,
        canvas_result=canvas_result,
        inpaint_mask=inpaint_mask,
        control_image=canny_control,
        depth_control_image=depth_control,
        generated_image=generated_image,
        experiment_data=experiment_data,
    )

    if owns_pipe:
        del pipe

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(
        "[TIME] "
        f"inference={inference_elapsed:.2f}s, "
        f"total={total_elapsed:.2f}s"
    )

    return result


def run_generation(
    product_image,
    prompt_json,
    output_dir,
    pipe=None,
    **options,
):
    config = _build_config(options)

    config.update(
        {
            "product_image": product_image,
            "prompt_json": prompt_json,
            "output_dir": output_dir,
            "pipe": pipe,
        }
    )

    return _run_generation(config)