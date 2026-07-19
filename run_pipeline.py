from adcg.config import parse_config
from adcg.pipeline import run_pipeline


def main():
    config = parse_config()

    result = run_pipeline(
        image_path=config.image_path,
        info_path=config.info_path,
        output_dir=config.output_dir,
        gpt_model=config.gpt_model,
        copy_count=config.copy_count,
        direction=config.direction,
        layout_mode=config.layout_mode,
        layout_model=config.layout_model,
        layout_font=config.layout_font,
        seed=config.seed,
        cpu_offload=config.cpu_offload,
        evaluate=config.evaluate,
        eval_metrics=config.eval_metrics,
    )

    print("\n[PIPELINE DONE]")
    print(f"Prompt JSON : {result.prompt_json}")
    print(f"Copy JSON   : {result.copy_json}")
    print(f"Generated   : {result.generated_image}")
    print(f"Core refined: {result.core_refined_image}")
    print(f"Identity img: {result.identity_restored_image}")
    print(f"Design info : {result.design_analysis_json}")
    print(f"Design spec : {result.design_spec_json}")
    print(f"Revision    : {result.design_revision_json}")
    print(f"Final review: {result.final_review_json}")
    print(f"Layout JSON : {result.layout_json}")
    print(f"Design draft: {result.layout_draft_image}")
    print(f"Review input: {result.final_review_input_image}")
    print(f"Final image : {result.final_image}")

    if result.eval_json is not None:
        print(f"Evaluation  : {result.eval_json}")


if __name__ == "__main__":
    main()
