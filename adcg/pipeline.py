import json
from dataclasses import dataclass
from pathlib import Path

from .eval import run_evaluation
from .generation import run_generation
from .preprocessing import run_preprocess
from .prompt_layout import generate_prompt_layout, load_ad_copy
from .prompting import generate_ad_copy, run_prompt_generation
from .refinement import (
    run_core_refinement,
    run_identity_restoration,
)


@dataclass(frozen=True)
class PipelineResult:
    output_dir: Path
    prompt_json: Path
    copy_json: Path
    generated_image: Path
    core_refined_image: Path
    identity_restored_image: Path
    layout_plan_json: Path
    layout_json: Path
    layout_preview_html: Path
    final_image: Path
    eval_json: Path | None


def run_pipeline(
    image_path,
    info_path,
    output_dir="outputs/pipeline",
    gpt_model="gpt-5.4-nano",
    copy_count=1,
    direction="product_focus",
    layout_mode="layout",
    layout_model="gpt-4o",
    layout_font=None,
    seed=42,
    cpu_offload=False,
    generation_options=None,
    core_refinement_options=None,
    identity_options=None,
    prompt_layout_options=None,
    evaluate=False,
    eval_metrics=None,
    eval_options=None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generation_options = dict(generation_options or {})
    core_refinement_options = dict(
        core_refinement_options or {}
    )
    identity_options = dict(identity_options or {})
    prompt_layout_options = dict(prompt_layout_options or {})
    eval_options = dict(eval_options or {})

    print("[1/8] Product preprocessing")

    preprocessed = run_preprocess(
        image_path=image_path,
        output_dir=output_dir / "01_preprocessed",
    )

    print("[2/8] Scene prompt generation")

    prompt_json = run_prompt_generation(
        image_path=image_path,
        info_path=info_path,
        output_path=(
            output_dir / "02_prompt" / "ad_prompt.json"
        ),
        model=gpt_model,
        direction=direction,
    )

    print("[3/8] Advertisement copy generation")

    product_info = json.loads(
        Path(info_path).read_text(encoding="utf-8")
    )
    prompt_data = json.loads(
        Path(prompt_json).read_text(encoding="utf-8")
    )

    generation_prompt = prompt_data.get(
        "generation_prompt",
        {},
    )
    background_prompt = generation_prompt.get(
        "background_prompt",
        "",
    )

    ad_copies = [
        generate_ad_copy(
            product_info=product_info,
            background_prompt=background_prompt,
            model=gpt_model,
        )
        for _ in range(copy_count)
    ]

    copy_json = output_dir / "02_prompt" / "ad_copy.json"
    copy_json.write_text(
        json.dumps(
            {
                "model": gpt_model,
                "copy_count": copy_count,
                "copies": ad_copies,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # preserve는 원본 캔버스의 상품 위치와 크기를 유지한다.
    if layout_mode == "preserve":
        generation_product = preprocessed["full_cutout"]
    else:
        generation_product = preprocessed["trimmed_cutout"]

    generation_kwargs = {
        "layout_mode": layout_mode,
        "seed": seed,
        "cpu_offload": cpu_offload,
    }
    generation_kwargs.update(generation_options)

    print("[4/8] Conditioned diffusion generation")

    generated = run_generation(
        product_image=generation_product,
        prompt_json=prompt_json,
        output_dir=output_dir / "03_generated",
        **generation_kwargs,
    )

    # Refinement 및 상품 보존 평가에는 원본 비율의 trimmed cutout 사용
    refinement_product = preprocessed["trimmed_cutout"]

    print("[5/8] Core product refinement")

    core_refined = run_core_refinement(
        generated_image=generated["image"],
        product_image=refinement_product,
        product_mask=generated["product_mask"],
        output_dir=output_dir / "04_core_refined",
        **core_refinement_options,
    )

    identity_kwargs = {
        "seed": seed,
        "cpu_offload": cpu_offload,
    }
    identity_kwargs.update(identity_options)

    print("[6/8] Boundary and identity restoration")

    identity_restored_image = run_identity_restoration(
        input_image=core_refined,
        product_image=refinement_product,
        product_mask=generated["product_mask"],
        prompt_json=prompt_json,
        output_dir=output_dir / "05_final",
        **identity_kwargs,
    )

    eval_json = None

    if evaluate:
        print("[7/8] Quantitative evaluation")

        eval_dir = output_dir / "06_eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        eval_json = eval_dir / "eval_results.json"

        eval_kwargs = {
            "metric_options": eval_options,
        }

        if eval_metrics is not None:
            eval_kwargs["metrics"] = tuple(eval_metrics)

        run_evaluation(
            final_image=identity_restored_image,
            prompt_json=prompt_json,
            product_image=refinement_product,
            product_mask=generated["product_mask"],
            output_json=eval_json,
            **eval_kwargs,
        )
    else:
        print("[7/8] Evaluation skipped")

    print("[8/8] Content-aware advertisement copy layout")

    layout_kwargs = {
        "model": layout_model,
        "font_path": layout_font,
    }
    layout_kwargs.update(prompt_layout_options)
    layout_result = generate_prompt_layout(
        image_path=identity_restored_image,
        ad_copy=load_ad_copy(copy_json),
        output_dir=output_dir / "07_prompt_layout",
        **layout_kwargs,
    )

    return PipelineResult(
        output_dir=output_dir,
        prompt_json=Path(prompt_json),
        copy_json=copy_json,
        generated_image=Path(generated["image"]),
        core_refined_image=Path(core_refined),
        identity_restored_image=Path(identity_restored_image),
        layout_plan_json=layout_result.plan_json,
        layout_json=layout_result.layout_json,
        layout_preview_html=layout_result.preview_html,
        final_image=layout_result.rendered_image,
        eval_json=eval_json,
    )
