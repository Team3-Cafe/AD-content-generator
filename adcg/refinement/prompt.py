import json
from pathlib import Path


POSITIVE_REQUIRED = (
    "seamless product boundary, matched ambient lighting, "
    "realistic contact with the supporting surface, "
    "natural edge colors, commercial product photography"
)


def load_refinement_prompt(prompt_json, generation_result_json=None):
    if generation_result_json is not None:
        result_path = Path(generation_result_json)
        if result_path.exists():
            result_data = json.loads(
                result_path.read_text(encoding="utf-8")
            )
            effective_prompt = str(
                result_data.get("effective_background_prompt") or ""
            ).strip()
            if effective_prompt:
                return effective_prompt

    data = json.loads(
        Path(prompt_json).read_text(encoding="utf-8")
    )

    generation = data.get("generation_prompt", {})

    prompt = generation.get(
        "background_prompt",
        data.get("background_prompt", ""),
    )
    return prompt.strip()
