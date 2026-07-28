from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Semaphore
from typing import Any

from ..artifacts import keep_only, remove_pipeline_stage
from ..generation import run_generation
from ..pipelines.image import (
    ImagePipelineResult,
    _aspect_aware_generation_size,
)
from ..preprocessing import run_preprocess
from ..prompting import run_prompt_generation
from ..refinement import (
    run_core_refinement,
    run_identity_restoration,
)


# A cached rembg ONNX session is reused by every preparation thread.
# Serialize only rembg preprocessing; OpenAI prompt generation can overlap
# with another job's preprocessing or GPU stage.
_PREPROCESS_SEMAPHORE = Semaphore(1)


@dataclass(frozen=True)
class PreparedImageJob:
    """Non-diffusion artifacts prepared without changing model inputs."""

    output_dir: Path
    info_path: Path
    prompt_json: Path
    preprocessed: dict[str, Any]
    generation_product: Path
    generation_width: int
    generation_height: int
    product_focus: float
    product_scale: float | None
    brand_focus: float
    layout_mode: str
    seed: int
    cpu_offload: bool
    evaluate: bool
    eval_metrics: tuple[str, ...] | None
    eval_options: dict[str, Any]


def prepare_image_job(
    image_path,
    info_path,
    output_dir="outputs/pipeline",
    gpt_model="gpt-5.4-nano",
    product_focus=1.0,
    product_scale=None,
    width=None,
    height=None,
    brand_focus=0.5,
    layout_mode="layout",
    seed=42,
    cpu_offload=False,
    evaluate=False,
    eval_metrics=None,
    eval_options=None,
):
    """Prepare the exact inputs used by the integration image pipeline."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    info_path = Path(info_path)
    eval_options = dict(eval_options or {})
    total_steps = 6 if evaluate else 5

    print(f"[image 1/{total_steps}] Product preprocessing")

    with _PREPROCESS_SEMAPHORE:
        preprocessed = run_preprocess(
            image_path=image_path,
            output_dir=output_dir / "01_preprocessed",
        )

    generation_width, generation_height = _aspect_aware_generation_size(
        preprocessed["original_size"],
        width=width,
        height=height,
    )

    print(f"[image 2/{total_steps}] Scene prompt generation")

    # Keep the integration branch's original VLM input. In particular, do
    # not replace the source image with the rembg cutout.
    prompt_json = run_prompt_generation(
        image_path=image_path,
        info_path=info_path,
        output_path=output_dir / "02_prompt" / "ad_prompt.json",
        model=gpt_model,
        product_focus=product_focus,
        brand_focus=brand_focus,
    )

    generation_product = (
        preprocessed["full_cutout"]
        if layout_mode == "preserve"
        else preprocessed["trimmed_cutout"]
    )

    return PreparedImageJob(
        output_dir=output_dir,
        info_path=info_path,
        prompt_json=Path(prompt_json),
        preprocessed=preprocessed,
        generation_product=Path(generation_product),
        generation_width=int(generation_width),
        generation_height=int(generation_height),
        product_focus=float(product_focus),
        product_scale=(
            None if product_scale is None else float(product_scale)
        ),
        brand_focus=float(brand_focus),
        layout_mode=str(layout_mode),
        seed=int(seed),
        cpu_offload=bool(cpu_offload),
        evaluate=bool(evaluate),
        eval_metrics=(
            tuple(eval_metrics) if eval_metrics is not None else None
        ),
        eval_options=eval_options,
    )


def run_prepared_image_job(
    prepared: PreparedImageJob,
) -> ImagePipelineResult:
    """Run the unchanged integration generation and refinement functions."""

    output_dir = prepared.output_dir
    preprocessed = prepared.preprocessed
    total_steps = 6 if prepared.evaluate else 5

    print(f"[image 3/{total_steps}] Conditioned diffusion generation")

    # pipe=None deliberately preserves the integration branch lifecycle:
    # generation loads its Dual ControlNet pipeline, then releases it.
    generated = run_generation(
        product_image=prepared.generation_product,
        prompt_json=prepared.prompt_json,
        output_dir=output_dir / "03_generated",
        pipe=None,
        width=prepared.generation_width,
        height=prepared.generation_height,
        layout_mode=prepared.layout_mode,
        brand_focus=prepared.brand_focus,
        product_scale=prepared.product_scale,
        preprocess_metadata=preprocessed.get("metadata"),
        seed=prepared.seed,
        cpu_offload=prepared.cpu_offload,
    )

    refinement_product = preprocessed["trimmed_cutout"]

    print(f"[image 4/{total_steps}] Core product refinement")

    core_refined = run_core_refinement(
        generated_image=generated["image"],
        product_image=refinement_product,
        product_mask=generated["product_mask"],
        output_dir=output_dir / "04_core_refined",
        product_focus=prepared.product_focus,
    )

    print(f"[image 5/{total_steps}] Boundary and identity restoration")

    # pipe=None preserves the integration branch's Canny-only identity stage.
    identity_restored_image = run_identity_restoration(
        input_image=core_refined,
        product_image=refinement_product,
        product_mask=generated["product_mask"],
        prompt_json=prepared.prompt_json,
        generation_result_json=generated.get("metadata"),
        output_dir=output_dir / "05_final",
        seed=prepared.seed,
        product_focus=prepared.product_focus,
        cpu_offload=prepared.cpu_offload,
        pipe=None,
    )

    eval_json = None
    if prepared.evaluate:
        from ..eval import run_evaluation

        print(f"[image 6/{total_steps}] Quantitative evaluation")
        eval_dir = output_dir / "06_eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        eval_json = eval_dir / "eval_results.json"
        eval_kwargs = {"metric_options": prepared.eval_options}
        if prepared.eval_metrics is not None:
            eval_kwargs["metrics"] = prepared.eval_metrics

        run_evaluation(
            final_image=identity_restored_image,
            prompt_json=prepared.prompt_json,
            product_image=refinement_product,
            product_mask=generated["product_mask"],
            output_json=eval_json,
            **eval_kwargs,
        )

    for stage_name in ("01_preprocessed", "03_generated", "04_core_refined"):
        remove_pipeline_stage(output_dir, stage_name)
    keep_only(output_dir / "05_final", (identity_restored_image,))
    if eval_json is not None:
        keep_only(output_dir / "06_eval", (eval_json,))

    return ImagePipelineResult(
        output_dir=output_dir,
        info_path=prepared.info_path,
        prompt_json=prepared.prompt_json,
        identity_restored_image=Path(identity_restored_image),
        eval_json=eval_json,
    )


__all__ = [
    "PreparedImageJob",
    "prepare_image_job",
    "run_prepared_image_job",
]
