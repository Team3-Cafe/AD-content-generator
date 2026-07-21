import json
from pathlib import Path


POSITIVE_REQUIRED = (
    "seamless product boundary, matched ambient lighting, "
    "realistic contact with the supporting surface, "
    "natural edge colors, commercial product photography"
)

NEGATIVE_REQUIRED = (
    "halo, black outline, white outline, jagged edge, "
    "pasted cutout, floating product, duplicate product, "
    "deformed product"
)


def load_refinement_prompt(prompt_json):
    data = json.loads(
        Path(prompt_json).read_text(encoding="utf-8")
    )

    generation = data.get("generation_prompt", {})

    prompt = generation.get(
        "background_prompt",
        data.get("background_prompt", ""),
    )
    negative = generation.get(
        "negative_prompt",
        data.get("negative_prompt", ""),
    )

    return prompt.strip(), negative.strip()
