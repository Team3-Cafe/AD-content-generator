import json
from dataclasses import dataclass
from pathlib import Path

from .prompting import generate_ad_copy, run_prompt_generation


@dataclass(frozen=True)
class PipelineResult:
    output_dir: Path
    prompt_json: Path
    copy_json: Path


def run_pipeline(
    image_path,
    info_path,
    output_dir="outputs/pipeline",
    gpt_model="gpt-5.4-nano",
    copy_count=1,
    direction="product_focus",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[1/2] Scene prompt generation")

    prompt_json = run_prompt_generation(
        image_path=image_path,
        info_path=info_path,
        output_path=output_dir / "02_prompt" / "ad_prompt.json",
        model=gpt_model,
        direction=direction,
    )

    print("[2/2] Advertisement copy generation")

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

    print("[DONE] Copy generation complete")

    return PipelineResult(
        output_dir=output_dir,
        prompt_json=Path(prompt_json),
        copy_json=copy_json,
    )
