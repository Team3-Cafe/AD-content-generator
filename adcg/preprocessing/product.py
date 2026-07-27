import json
import warnings
from functools import lru_cache
from pathlib import Path
from threading import Semaphore

import onnxruntime as ort
from PIL import Image
from rembg import new_session, remove

from .alpha import clean_alpha, neutralize_transparent_pixels
from .geometry import crop_to_product
from .masks import make_masks
from .preview import save_preview
from .validation import detect_truncation, handle_truncation


CPU_PROVIDER = "CPUExecutionProvider"
CUDA_PROVIDER = "CUDAExecutionProvider"
_REMBG_INFERENCE_SEMAPHORE = Semaphore(1)


def get_rembg_providers():
    """Prefer ONNX CUDA when installed and usable, otherwise use CPU."""
    try:
        available = set(ort.get_available_providers())
    except Exception:
        available = {CPU_PROVIDER}

    if CUDA_PROVIDER in available:
        return [CUDA_PROVIDER, CPU_PROVIDER]
    return [CPU_PROVIDER]


@lru_cache(maxsize=2)
def get_rembg_session(model="u2net"):
    """Load each rembg model once and reuse its ONNX session."""
    providers = get_rembg_providers()
    try:
        session = new_session(model, providers=providers)
        print(f"[rembg] provider: {providers[0]}")
        return session
    except Exception as error:
        if providers[0] != CUDA_PROVIDER:
            raise

        warnings.warn(
            "rembg CUDA initialization failed; falling back to CPU: "
            f"{error}",
            RuntimeWarning,
            stacklevel=2,
        )
        session = new_session(model, providers=[CPU_PROVIDER])
        print(f"[rembg] provider: {CPU_PROVIDER} (CUDA fallback)")
        return session


def _remove_background(image, model="u2net"):
    """Reuse the cached session while limiting rembg to one inference."""
    with _REMBG_INFERENCE_SEMAPHORE:
        session = get_rembg_session(model)
        return remove(
            image,
            session=session,
            alpha_matting=False,
        )


def run_preprocess(
    image_path,
    output_dir="outputs/preprocessed",
    alpha_threshold=8,
    padding=10,
    model="u2net",
    truncation_policy="warn",
    edge_margin=2,
    min_edge_coverage=0.01,
    truncation_alpha_threshold=32,
):
    image_path = Path(image_path)
    output_dir = Path(output_dir)

    if not image_path.exists():
        raise FileNotFoundError(
            f"이미지를 찾을 수 없습니다: {image_path}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    original = Image.open(image_path).convert("RGBA")

    cutout = _remove_background(original, model=model)

    cutout = clean_alpha(
        cutout,
        threshold=alpha_threshold,
    )
    cutout = neutralize_transparent_pixels(cutout)

    truncation = detect_truncation(
        cutout=cutout,
        edge_margin=edge_margin,
        min_edge_coverage=min_edge_coverage,
        alpha_threshold=truncation_alpha_threshold,
    )

    handle_truncation(
        truncation=truncation,
        policy=truncation_policy,
    )

    product_mask, background_mask = make_masks(cutout)

    trimmed_cutout, bbox = crop_to_product(
        image=cutout,
        padding=padding,
    )

    paths = {
        "full_cutout": output_dir / "product_cutout.png",
        "trimmed_cutout": (
            output_dir / "product_cutout_trimmed.png"
        ),
        "product_mask": output_dir / "product_mask.png",
        "background_mask": output_dir / "background_mask.png",
        "preview": output_dir / "cutout_preview.png",
        "metadata": output_dir / "preprocess_metadata.json",
    }

    cutout.save(paths["full_cutout"])
    trimmed_cutout.save(paths["trimmed_cutout"])
    product_mask.save(paths["product_mask"])
    background_mask.save(paths["background_mask"])
    save_preview(cutout, paths["preview"])

    metadata = {
        "input_image": str(image_path),
        "original_size": {
            "width": original.width,
            "height": original.height,
        },
        "product_bbox": {
            "left": bbox[0],
            "top": bbox[1],
            "right": bbox[2],
            "bottom": bbox[3],
        },
        "trimmed_size": {
            "width": trimmed_cutout.width,
            "height": trimmed_cutout.height,
        },
        "alpha_threshold": alpha_threshold,
        "rembg_model": model,
        "truncation_policy": truncation_policy,
        "truncation": truncation,
        "outputs": {
            key: str(value)
            for key, value in paths.items()
            if key != "metadata"
        },
    }

    paths["metadata"].write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"[DONE] preprocessing: "
        f"{paths['trimmed_cutout']}"
    )

    if truncation["is_truncated"]:
        print(
            "[WARNING] 상품 경계 접촉:",
            ", ".join(truncation["touching_edges"]),
        )

    return {
        # 기존 파이프라인 호환성을 위해 cutout은 trimmed 경로로 유지
        "cutout": paths["trimmed_cutout"],
        "full_cutout": paths["full_cutout"],
        "trimmed_cutout": paths["trimmed_cutout"],
        "product_mask": paths["product_mask"],
        "background_mask": paths["background_mask"],
        "preview": paths["preview"],
        "metadata": paths["metadata"],
        "original_size": original.size,
        "bbox": bbox,
        "truncation": truncation,
    }
