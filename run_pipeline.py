from adcg.preprocessing.config import parse_config


def main():
    config = parse_config()
    from adcg.pipeline import run_pipeline

    result = run_pipeline(
        image_path=config.image_path,
        info_path=config.info_path,
        output_dir=config.output_dir,
        gpt_model=config.gpt_model,
        copy_count=config.copy_count,
        direction=config.direction,
        layout_mode=config.layout_mode,
        seed=config.seed,
        cpu_offload=config.cpu_offload,
    )

    print("\n[PIPELINE DONE]")
    print(f"Ad copy: {result.copy_json}")
    print(f"Final image: {result.final_image}")


if __name__ == "__main__":
    main()
