from __future__ import annotations


PRESELECTION_WEIGHTS = {
    "aesthetic": 0.35,
    "structure": 0.65,
}

FINAL_WEIGHTS = {
    "vlm_design": 0.45,
    "aesthetic": 0.35,
    "structure": 0.20,
}

VLM_DIMENSION_WEIGHTS = {
    "hierarchy": 0.30,
    "readability": 0.30,
    "balance": 0.20,
    "commercial_finish": 0.20,
}


def normalize_scores(values: list[float]) -> list[float]:
    """Min-max normalize candidate-relative scores to the 0-100 range."""
    if not values:
        return []
    low = min(values)
    high = max(values)
    if high == low:
        return [50.0 for _ in values]
    return [
        round((value - low) / (high - low) * 100.0, 4)
        for value in values
    ]


def vlm_design_score(rating: dict) -> float:
    score = sum(
        float(rating[name]) * weight
        for name, weight in VLM_DIMENSION_WEIGHTS.items()
    )
    return round(score, 4)


def preselection_score(*, aesthetic: float, structure: float) -> float:
    return round(
        aesthetic * PRESELECTION_WEIGHTS["aesthetic"]
        + structure * PRESELECTION_WEIGHTS["structure"],
        4,
    )


def final_score(
    *,
    vlm_design: float,
    aesthetic: float,
    structure: float,
) -> float:
    return round(
        vlm_design * FINAL_WEIGHTS["vlm_design"]
        + aesthetic * FINAL_WEIGHTS["aesthetic"]
        + structure * FINAL_WEIGHTS["structure"],
        4,
    )


__all__ = [
    "FINAL_WEIGHTS",
    "PRESELECTION_WEIGHTS",
    "VLM_DIMENSION_WEIGHTS",
    "final_score",
    "normalize_scores",
    "preselection_score",
    "vlm_design_score",
]
