from dataclasses import dataclass
from pathlib import Path

from .generation import run_generation
from .preprocessing import run_preprocess
from .prompting import run_prompt_generation
from .refinement import run_core_refinement, run_identity_restoration


@dataclass(frozen=True)
class PipelineResult:
    output_dir: Path
    prompt_json: Path
    generated_image: Path
    core_refined_image: Path
    final_image: Path


def run_pipeline(
    image_path,
    info_path,
    output_dir="outputs/pipeline",
    gpt_model="gpt-5.4-nano",
    direction="product_focus",
    layout_mode="layout",
    seed=42,
    cpu_offload=False,
    generation_options=None,
    core_refinement_options=None,
    identity_options=None,
):
    """Run preprocessing, prompting, generation, and refinement in sequence."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generation_options = dict(generation_options or {})
    core_refinement_options = dict(core_refinement_options or {})
    identity_options = dict(identity_options or {})

    preprocessed = run_preprocess(
        image_path=image_path,
        output_dir=output_dir / "01_preprocessed",
    )

    prompt_json = run_prompt_generation(
        image_path=image_path,
        info_path=info_path,
        output_path=output_dir / "02_prompt" / "ad_prompt.json",
        model=gpt_model,
        direction=direction,
    )

    generated = run_generation(
        product_image=preprocessed["cutout"],
        prompt_json=prompt_json,
        output_dir=output_dir / "03_generated",
        layout_mode=layout_mode,
        seed=seed,
        cpu_offload=cpu_offload,
        **generation_options,
    )

    product_image = generated.get(
        "product_layer",
        preprocessed["trimmed_cutout"],
    )

    core_refined = run_core_refinement(
        generated_image=generated["image"],
        product_image=product_image,
        product_mask=generated["product_mask"],
        output_dir=output_dir / "04_core_refined",
        **core_refinement_options,
    )

    final_image = run_identity_restoration(
        input_image=core_refined,
        product_image=product_image,
        product_mask=generated["product_mask"],
        prompt_json=prompt_json,
        output_dir=output_dir / "05_final",
        seed=seed,
        cpu_offload=cpu_offload,
        **identity_options,
    )

    return PipelineResult(
        output_dir=output_dir,
        prompt_json=Path(prompt_json),
        generated_image=Path(generated["image"]),
        core_refined_image=Path(core_refined),
        final_image=Path(final_image),
    )
