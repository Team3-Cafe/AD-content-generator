from adcg.config import parse_config
from adcg.pipeline import run_pipeline


def main():
    config = parse_config()

    result = run_pipeline(
        image_path=config.image_path,
        info_path=config.info_path,
        output_dir=config.output_dir,
        gpt_model=config.gpt_model,
        copy_count=getattr(config, "copy_count", 9),
        direction=config.direction,
        layout_mode=config.layout_mode,
        seed=config.seed,
        cpu_offload=config.cpu_offload,
        evaluate=getattr(config, "evaluate", False),
        eval_metrics=getattr(config, "eval_metrics", None),
    )

    print("\n[PIPELINE DONE]")
    print(f"Prompt JSON : {result.prompt_json}")
    print(f"Copy JSON   : {result.copy_json}")
    print(f"Generated   : {result.generated_image}")
    print(f"Core refined: {result.core_refined_image}")
    print(f"Final image : {result.final_image}")

    if result.eval_json is not None:
        print(f"Evaluation  : {result.eval_json}")


if __name__ == "__main__":
    main()