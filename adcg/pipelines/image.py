from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..generation import run_generation
from ..preprocessing import run_preprocess
from ..prompting import run_prompt_generation
from ..refinement import (
    run_core_refinement,
    run_identity_restoration,
)


@dataclass(frozen=True)
class ImagePipelineResult:
    """Artifacts produced before advertisement copy is requested."""

    output_dir: Path
    info_path: Path
    prompt_json: Path
    generated_image: Path
    core_refined_image: Path
    identity_restored_image: Path
    eval_json: Path | None


def run_image_pipeline(
    image_path,
    info_path,
    output_dir="outputs/pipeline",
    gpt_model="gpt-5.4-nano",
    product_focus=1.0,
    brand_focus=0.5,
    layout_mode="layout",
    seed=42,
    cpu_offload=False,
    diffusion_pipe=None,
    evaluate=False,
    eval_metrics=None,
    eval_options=None,
):
    """Run the pipeline through identity restoration, without ad copy."""

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    info_path = Path(info_path)
    eval_options = dict(eval_options or {})
    total_steps = 6 if evaluate else 5

    print(
        f"[image 1/{total_steps}] "
        "Product preprocessing"
    )

    preprocessed = run_preprocess(
        image_path=image_path,
        output_dir=(
            output_dir / "01_preprocessed"
        ),
    )

    print(
        f"[image 2/{total_steps}] "
        "Scene prompt generation"
    )

    prompt_json = run_prompt_generation(
        image_path=image_path,
        info_path=info_path,
        output_path=(
            output_dir
            / "02_prompt"
            / "ad_prompt.json"
        ),
        model=gpt_model,
        product_focus=product_focus,
        brand_focus=brand_focus,
    )

    if layout_mode == "preserve":
        generation_product = (
            preprocessed["full_cutout"]
        )
    else:
        generation_product = (
            preprocessed["trimmed_cutout"]
        )

    generation_kwargs = {
        "layout_mode": layout_mode,
        "brand_focus": brand_focus,
        "seed": seed,
        "cpu_offload": cpu_offload,
    }

    print(
        f"[image 3/{total_steps}] "
        "Conditioned diffusion generation"
    )

    generated = run_generation(
        product_image=generation_product,
        prompt_json=prompt_json,
        output_dir=(
            output_dir / "03_generated"
        ),
        pipe=diffusion_pipe,
        **generation_kwargs,
    )

    refinement_product = (
        preprocessed["trimmed_cutout"]
    )

    print(
        f"[image 4/{total_steps}] "
        "Core product refinement"
    )

    core_refined = run_core_refinement(
        generated_image=generated["image"],
        product_image=refinement_product,
        product_mask=generated["product_mask"],
        output_dir=(
            output_dir / "04_core_refined"
        ),
        product_focus=product_focus,
    )

    print(
        f"[image 5/{total_steps}] "
        "Boundary and identity restoration"
    )

    identity_restored_image = (
        run_identity_restoration(
            input_image=core_refined,
            product_image=refinement_product,
            product_mask=generated[
                "product_mask"
            ],
            prompt_json=prompt_json,
            generation_result_json=generated.get(
                "metadata"
            ),
            output_dir=(
                output_dir / "05_final"
            ),
            seed=seed,
            product_focus=product_focus,
            cpu_offload=cpu_offload,
            pipe=diffusion_pipe,
        )
    )

    eval_json = None

    if evaluate:
        from ..eval import run_evaluation

        print(
            f"[image 6/{total_steps}] "
            "Quantitative evaluation"
        )

        eval_dir = output_dir / "06_eval"
        eval_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        eval_json = (
            eval_dir / "eval_results.json"
        )

        eval_kwargs = {
            "metric_options": eval_options,
        }

        if eval_metrics is not None:
            eval_kwargs["metrics"] = tuple(
                eval_metrics
            )

        run_evaluation(
            final_image=(
                identity_restored_image
            ),
            prompt_json=prompt_json,
            product_image=refinement_product,
            product_mask=generated[
                "product_mask"
            ],
            output_json=eval_json,
            **eval_kwargs,
        )

    return ImagePipelineResult(
        output_dir=output_dir,
        info_path=info_path,
        prompt_json=Path(prompt_json),
        generated_image=Path(
            generated["image"]
        ),
        core_refined_image=Path(
            core_refined
        ),
        identity_restored_image=Path(
            identity_restored_image
        ),
        eval_json=eval_json,
    )