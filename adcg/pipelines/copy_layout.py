from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..prompt_layout import generate_prompt_layout, load_ad_copy
from ..prompting import generate_ad_copy


@dataclass(frozen=True)
class CopyLayoutResult:
    """Artifacts produced after copy controls have been supplied."""

    output_dir: Path
    copy_json: Path
    layout_json: Path
    final_review_json: Path
    final_image: Path


def _load_json(path: str | Path, label: str) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{label} JSON을 찾을 수 없습니다: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{label} JSON은 object여야 합니다.")
    return data


def run_copy_layout_pipeline(
    identity_image,
    info_path,
    prompt_json,
    output_dir="outputs/pipeline",
    gpt_model="gpt-5.4-nano",
    copy_count=1,
    copy_tone=None,
    copy_length=None,
    copy_index=0,
    layout_model="gpt-4o",
    layout_options=None,
):
    """Generate controlled ad copy, then compose it onto a finished image."""
    copy_count = int(copy_count)
    if copy_count < 1:
        raise ValueError("copy_count는 1 이상이어야 합니다.")
    if not 0 <= int(copy_index) < copy_count:
        raise ValueError("copy_index는 생성할 카피 범위 안에 있어야 합니다.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    product_info = _load_json(info_path, "상품 정보")
    prompt_data = _load_json(prompt_json, "장면 프롬프트")
    background_prompt = prompt_data.get(
        "generation_prompt",
        {},
    ).get("background_prompt", "")

    print("[copy 1/2] Advertisement copy generation")

    ad_copies = [
        generate_ad_copy(
            product_info=product_info,
            background_prompt=background_prompt,
            model=gpt_model,
            copy_tone=copy_tone,
            copy_length=copy_length,
        )
        for _ in range(copy_count)
    ]

    copy_json = output_dir / "02_prompt" / "ad_copy.json"
    copy_json.parent.mkdir(parents=True, exist_ok=True)
    copy_json.write_text(
        json.dumps(
            {
                "model": gpt_model,
                "copy_count": copy_count,
                "controls": {
                    "copy_tone": copy_tone,
                    "copy_length": copy_length,
                },
                "copies": ad_copies,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("[copy 2/2] Content-aware advertisement copy layout")

    layout_kwargs = dict(layout_options or {})
    layout_kwargs.setdefault("model", layout_model)
    layout_result = generate_prompt_layout(
        image_path=identity_image,
        ad_copy=load_ad_copy(copy_json, copy_index=int(copy_index)),
        output_dir=output_dir / "07_prompt_layout",
        **layout_kwargs,
    )

    return CopyLayoutResult(
        output_dir=output_dir,
        copy_json=copy_json,
        layout_json=Path(layout_result.layout_json),
        final_review_json=Path(layout_result.final_review_json),
        final_image=Path(layout_result.rendered_image),
    )
