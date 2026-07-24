from dataclasses import dataclass
from pathlib import Path

from .pipelines import (
    run_copy_layout_pipeline,
    run_image_pipeline,
)


@dataclass(frozen=True)
class PipelineResult:
    output_dir: Path
    prompt_json: Path
    copy_json: Path
    identity_restored_image: Path
    layout_json: Path
    final_image: Path
    eval_json: Path | None


def run_pipeline(
    image_path,
    info_path,
    output_dir="outputs/pipeline",
    gpt_model="gpt-5.4-nano",
    copy_count=1,
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
    copy_tone=None,
    copy_length=None,
    copy_index=0,
    layout_model="gpt-4o",
    layout_options=None,
):
    """Run both stages continuously for backward-compatible callers."""
    image_result = run_image_pipeline(
        image_path=image_path,
        info_path=info_path,
        output_dir=output_dir,
        gpt_model=gpt_model,
        product_focus=product_focus,
        product_scale=product_scale,
        width=width,
        height=height,
        brand_focus=brand_focus,
        layout_mode=layout_mode,
        seed=seed,
        cpu_offload=cpu_offload,
        evaluate=evaluate,
        eval_metrics=eval_metrics,
        eval_options=eval_options,
    )

    copy_result = run_copy_layout_pipeline(
        identity_image=image_result.identity_restored_image,
        info_path=image_result.info_path,
        prompt_json=image_result.prompt_json,
        output_dir=image_result.output_dir,
        gpt_model=gpt_model,
        copy_count=copy_count,
        copy_tone=copy_tone,
        copy_length=copy_length,
        copy_index=copy_index,
        layout_model=layout_model,
        layout_options=layout_options,
    )

    return PipelineResult(
        output_dir=image_result.output_dir,
        prompt_json=image_result.prompt_json,
        copy_json=copy_result.copy_json,
        identity_restored_image=image_result.identity_restored_image,
        layout_json=copy_result.layout_json,
        final_image=copy_result.final_image,
        eval_json=image_result.eval_json,
    )


__all__ = [
    "PipelineResult",
    "run_copy_layout_pipeline",
    "run_image_pipeline",
    "run_pipeline",
]
