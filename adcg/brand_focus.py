EVERYDAY_BACKGROUND_ANCHOR = (
    "ordinary lived-in everyday environment, candid unstyled documentary "
    "realism, visible practical wear"
)
STUDIO_BACKGROUND_ANCHOR = (
    "exclusive purpose-built advertising studio, immaculate premium set, "
    "dramatic hero presentation"
)


def _prepend_anchor(prompt: str, anchor: str) -> str:
    prompt = str(prompt or "").strip().strip(",")
    if anchor.casefold() in prompt.casefold():
        return prompt
    return f"{anchor}, {prompt}" if prompt else anchor


def anchor_background_prompts(
    everyday_prompt: str,
    studio_prompt: str,
) -> tuple[str, str]:
    """Add domain-neutral endpoint character without choosing a location."""
    return (
        _prepend_anchor(
            everyday_prompt,
            EVERYDAY_BACKGROUND_ANCHOR,
        ),
        _prepend_anchor(
            studio_prompt,
            STUDIO_BACKGROUND_ANCHOR,
        ),
    )

def brand_blend_weight(brand_focus: float) -> float:
    """Map brand focus to a contrast-enhanced continuous blend weight."""
    value = max(0.0, min(1.0, float(brand_focus)))
    everyday = (1.0 - value) ** 2
    studio = value**2
    denominator = everyday + studio
    return studio / denominator if denominator else 0.5


def select_background_prompt(
    everyday_prompt: str,
    studio_prompt: str,
    brand_focus: float,
) -> str:
    """Select the dominant endpoint for text-only downstream consumers."""
    weight = brand_blend_weight(brand_focus)
    return studio_prompt if weight >= 0.5 else everyday_prompt


def blend_prompt_embeddings(everyday, studio, brand_focus: float):
    """Continuously interpolate two compatible CLIP embedding tensors."""
    weight = brand_blend_weight(brand_focus)
    return everyday * (1.0 - weight) + studio * weight
