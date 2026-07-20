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
    )

    print("\n[PIPELINE DONE]")
    print(f"Prompt JSON : {result.prompt_json}")
    print(f"Copy JSON   : {result.copy_json}")


if __name__ == "__main__":
    main()
