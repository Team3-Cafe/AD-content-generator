from adcg.cli import parse_args


def main():
    args = parse_args()
    from adcg.pipeline import run_pipeline

    result = run_pipeline(
        image_path=args.image,
        info_path=args.info,
        output_dir=args.output_dir,
        gpt_model=args.gpt_model,
        direction=args.direction,
        layout_mode=args.layout_mode,
        seed=args.seed,
        cpu_offload=args.cpu_offload,
    )

    print("\n[PIPELINE DONE]")
    print(f"Final image: {result.final_image}")


if __name__ == "__main__":
    main()
