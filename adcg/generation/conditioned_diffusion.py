from adcg.image_utils.layout import (
    calculate_auto_layout, place_product, place_product_preserve,
)
from adcg.image_utils.masks import (
    create_background_inpaint_mask, create_canny_control,
)
from adcg.image_utils.blending import restore_product_inner_detail
from .model_loader import load_generation_pipeline as load_pipeline

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from diffusers import (
    ControlNetModel,
    EulerAncestralDiscreteScheduler,
    StableDiffusionControlNetInpaintPipeline,
)
from PIL import Image, ImageFilter, ImageOps
from argparse import Namespace


RESAMPLING = getattr(Image, "Resampling", Image).LANCZOS




def load_prompt_data(prompt_json):
    with open(prompt_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    generation_prompt = data.get("generation_prompt", {})

    background_prompt = (
        generation_prompt.get("background_prompt")
        or data.get("background_prompt")
        or ""
    )

    negative_prompt = (
        generation_prompt.get("negative_prompt")
        or data.get("negative_prompt")
        or ""
    )

    layout = data.get("layout", {})

    return data, background_prompt, negative_prompt, layout


def load_product_cutout(
    image_path,
    mode,
    alpha_threshold,
    trim=True,
):
    image = Image.open(image_path).convert("RGBA")
    alpha = image.getchannel("A")

    has_transparency = alpha.getextrema()[0] < 250

    should_use_rembg = (
        mode == "rembg"
        or (mode == "auto" and not has_transparency)
    )

    if should_use_rembg:
        try:
            from rembg import remove
        except ImportError as exc:
            raise RuntimeError(
                "rembg가 필요합니다: python -m pip install rembg"
            ) from exc

        image = remove(image).convert("RGBA")

    alpha_np = np.asarray(
        image.getchannel("A"),
        dtype=np.uint8,
    )

    alpha_np = np.where(
        alpha_np <= alpha_threshold,
        0,
        alpha_np,
    ).astype(np.uint8)

    image.putalpha(Image.fromarray(alpha_np, mode="L"))

    bbox = image.getchannel("A").getbbox()

    if bbox is None:
        raise ValueError("상품 alpha 영역을 찾지 못했습니다.")

    return image.crop(bbox) if trim else image




















def _run_generation(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    (
        prompt_data,
        background_prompt,
        negative_prompt,
        layout,
    ) = load_prompt_data(args.prompt_json)


    background_prompt = background_prompt.strip()
    negative_prompt = negative_prompt.strip()
    

    product = load_product_cutout(
        image_path=args.product_image,
        mode=args.cutout_mode,
        alpha_threshold=args.alpha_threshold,
        trim=args.layout_mode != "preserve",
    )
    
    if args.layout_mode == "preserve":
        product_x = None
        product_y = None
        product_scale = None
        auto_layout = None
    
        (
            resized_product,
            product_layer,
            condition_canvas,
            product_mask,
            placement,
        ) = place_product_preserve(
            product=product,
            width=args.width,
            height=args.height,
        )
    
        print("[레이아웃] 입력 이미지 위치 보존")
        print(placement)
    
    else:
        auto_layout = calculate_auto_layout(
            product=product,
            canvas_width=args.width,
            canvas_height=args.height,
            gpt_layout=layout,
            reference_weight=args.layout_reference_weight,
        )
    
        product_x = (
            args.product_x
            if args.product_x is not None
            else auto_layout["product_x"]
        )
        product_y = (
            args.product_y
            if args.product_y is not None
            else auto_layout["product_y"]
        )
        product_scale = (
            args.product_scale
            if args.product_scale is not None
            else auto_layout["product_scale"]
        )
    
        (
            resized_product,
            product_layer,
            condition_canvas,
            product_mask,
            placement,
        ) = place_product(
            product=product,
            width=args.width,
            height=args.height,
            product_x=product_x,
            product_y=product_y,
            product_scale=product_scale,
        )
    
        print("[레이아웃] 기존 자동/GPT 배치")
        print(f"GPT 참고값: {auto_layout['gpt_layout']}")
        print(f"자동 계산값: {auto_layout['auto_layout']}")

    resized_product.save(
        output_dir / "product_resized.png"
    )
    product_layer.save(
        output_dir / "product_layer.png"
    )
    condition_canvas.save(
        output_dir / "condition_canvas.png"
    )
    product_mask.save(
        output_dir / "product_alpha_mask.png"
    )

    background_inpaint_mask = (
        create_background_inpaint_mask(
            product_mask=product_mask,
            margin=args.mask_margin,
            blur=args.mask_blur,
        )
    )
    background_inpaint_mask.save(
        output_dir / "background_inpaint_mask.png"
    )

    canny_control = create_canny_control(
        product_layer=product_layer,
        product_mask=product_mask,
        low_threshold=args.canny_low,
        high_threshold=args.canny_high,
    )
    canny_control.save(
        output_dir / "product_canny_control.png"
    )

    pipe = load_pipeline(args)

    generator = torch.Generator(
        device="cuda"
    ).manual_seed(args.seed)

    print("[Cutout condition diffusion 시작]")

    start_time = time.perf_counter()

    generated_image = pipe(
        prompt=background_prompt,
        negative_prompt=negative_prompt,
        image=condition_canvas.convert("RGB"),
        mask_image=background_inpaint_mask,
        control_image=canny_control,
        width=args.width,
        height=args.height,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        strength=args.strength,
        controlnet_conditioning_scale=args.controlnet_scale,
        generator=generator,
    ).images[0].convert("RGB")

    generation_time = time.perf_counter() - start_time

    generated_path = (
        output_dir
        / "generated_with_cutout_condition.png"
    )
    generated_image.save(generated_path)

    if args.disable_inner_restore:
        final_image = generated_image
        inner_mask = Image.new(
            "L",
            (args.width, args.height),
            0,
        )
        restore_mask = inner_mask.copy()
    else:
        print("[원본 상품 내부 디테일 복원]")

        final_image, inner_mask, restore_mask = (
            restore_product_inner_detail(
                generated_image=generated_image,
                product_layer=product_layer,
                product_mask=product_mask,
                inner_erode=args.inner_erode,
                inner_feather=args.inner_feather,
                opacity=args.inner_restore_opacity,
                color_match=not args.disable_color_match,
            )
        )

    inner_mask.save(
        output_dir / "product_inner_mask.png"
    )
    restore_mask.save(
        output_dir / "product_restore_mask.png"
    )

    final_restored_path = (
        output_dir / "final_inner_restored.png"
    )
    final_image.save(final_restored_path)
    final_image.save(output_dir / "final.png")

    result = {
        "base_model": args.base_model,
        "controlnet_model": args.controlnet_model,
        "product_image": args.product_image,
        "prompt_json": args.prompt_json,
        "background_prompt": background_prompt,
        "negative_prompt": negative_prompt,
        "layout": {
            "mode": args.layout_mode,
            "product_x": product_x,
            "product_y": product_y,
            "product_scale": product_scale,
            "placement_pixels": placement,
            "auto_layout": (
                auto_layout["auto_layout"]
                if auto_layout is not None
                else None
            ),
            "gpt_layout": (
                auto_layout["gpt_layout"]
                if auto_layout is not None
                else None
            ),
            "gpt_reference_weight": (
                auto_layout["reference_weight"]
                if auto_layout is not None
                else None
            ),
        },
        "generation": {
            "width": args.width,
            "height": args.height,
            "steps": args.steps,
            "guidance_scale": args.guidance_scale,
            "strength": args.strength,
            "controlnet_scale": args.controlnet_scale,
            "mask_margin": args.mask_margin,
            "mask_blur": args.mask_blur,
            "seed": args.seed,
            "generation_time_sec": generation_time,
        },
        "inner_restoration": {
            "enabled": not args.disable_inner_restore,
            "inner_erode": args.inner_erode,
            "inner_feather": args.inner_feather,
            "opacity": args.inner_restore_opacity,
            "color_match": not args.disable_color_match,
        },
        "outputs": {
            "generated": str(generated_path),
            "final_inner_restored": str(
                final_restored_path
            ),
            "final": str(output_dir / "final.png"),
        },
    }

    with open(
        output_dir / "experiment_result.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            result,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n[완료]")
    print(f"생성 결과: {generated_path}")
    print(f"내부 복원: {final_restored_path}")
    print(f"최종 결과: {output_dir / 'final.png'}")
    print(f"생성 시간: {generation_time:.2f}초")

    return {
        "image": final_restored_path,
        "generated_image": generated_path,
        "product_mask": output_dir / "product_alpha_mask.png",
        "condition_canvas": output_dir / "condition_canvas.png",
        "canny_control": output_dir / "product_canny_control.png",
        "background_mask": output_dir / "background_inpaint_mask.png",
        "result_json": output_dir / "experiment_result.json",
    }


GENERATION_DEFAULTS = {
    "base_model": "digiplay/majicMIX_realistic_v7",
    "controlnet_model": "lllyasviel/control_v11p_sd15_canny",
    "width": 512,
    "height": 512,
    "steps": 35,
    "guidance_scale": 8.0,
    "strength": 1.0,
    "controlnet_scale": 0.7,
    "layout_mode": "layout",
    "product_x": None,
    "product_y": None,
    "product_scale": None,
    "layout_reference_weight": 0.35,
    "mask_margin": 4,
    "mask_blur": 12.0,
    "canny_low": 100,
    "canny_high": 200,
    "inner_erode": 12,
    "inner_feather": 4.0,
    "inner_restore_opacity": 0.75,
    "disable_inner_restore": False,
    "disable_color_match": False,
    "cutout_mode": "auto",
    "alpha_threshold": 4,
    "seed": 42,
    "cpu_offload": False,
}


def run_generation(product_image, prompt_json, output_dir, layout_mode, **overrides):
    unknown = set(overrides) - set(GENERATION_DEFAULTS)
    if unknown:
        raise TypeError(f"지원하지 않는 생성 옵션: {sorted(unknown)}")

    config = {
        **GENERATION_DEFAULTS,
        **overrides,
        "product_image": str(product_image),
        "prompt_json": str(prompt_json),
        "output_dir": str(output_dir),
        "layout_mode": str(layout_mode),
    }

    return _run_generation(Namespace(**config))
