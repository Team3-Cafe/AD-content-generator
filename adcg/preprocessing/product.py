import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from rembg import new_session, remove


def clean_alpha(image, threshold):
    image = image.convert("RGBA")
    array = np.array(image)
    alpha = array[:, :, 3]

    alpha[alpha < threshold] = 0
    array[:, :, 3] = alpha

    return Image.fromarray(array, "RGBA")


def neutralize_transparent_pixels(image):
    """투명 영역의 RGB를 흰색으로 바꿔 검정 테두리/박스를 방지한다."""
    image = image.convert("RGBA")
    array = np.array(image)

    transparent = array[:, :, 3] == 0
    array[transparent, 0] = 255
    array[transparent, 1] = 255
    array[transparent, 2] = 255

    return Image.fromarray(array, "RGBA")


def make_masks(cutout):
    alpha = cutout.getchannel("A")

    # 흰색: 상품, 검정색: 배경
    product_mask = alpha

    # 흰색: 생성할 배경, 검정색: 보존할 상품
    background_mask = Image.eval(alpha, lambda value: 255 - value)

    return product_mask, background_mask


def crop_to_product(image, padding=10):
    alpha = image.getchannel("A")
    bbox = alpha.getbbox()

    if bbox is None:
        raise RuntimeError("상품 영역을 찾지 못했습니다.")

    left, top, right, bottom = bbox

    left = max(0, left - padding)
    top = max(0, top - padding)
    right = min(image.width, right + padding)
    bottom = min(image.height, bottom + padding)

    return image.crop((left, top, right, bottom)), (left, top, right, bottom)


def save_preview(cutout, output_path):
    background = Image.new("RGBA", cutout.size, (255, 255, 255, 255))
    preview = Image.alpha_composite(background, cutout)
    preview.convert("RGB").save(output_path)


def run_preprocess(
    image_path,
    output_dir="outputs/preprocessed",
    alpha_threshold=8,
    padding=10,
    model="u2net",
):
    image_path = Path(image_path)
    output_dir = Path(output_dir)

    if not image_path.exists():
        raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {image_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    original = Image.open(image_path).convert("RGBA")

    session = new_session(
        model,
        providers=["CPUExecutionProvider"],
    )

    cutout = remove(
        original,
        session=session,
        alpha_matting=False,
    )

    cutout = clean_alpha(cutout, alpha_threshold)
    cutout = neutralize_transparent_pixels(cutout)

    product_mask, background_mask = make_masks(cutout)
    trimmed_cutout, bbox = crop_to_product(cutout, padding)

    cutout_path = output_dir / "product_cutout.png"
    trimmed_path = output_dir / "product_cutout_trimmed.png"
    product_mask_path = output_dir / "product_mask.png"
    background_mask_path = output_dir / "background_mask.png"
    preview_path = output_dir / "cutout_preview.png"
    metadata_path = output_dir / "preprocess_metadata.json"

    cutout.save(cutout_path)
    trimmed_cutout.save(trimmed_path)
    product_mask.save(product_mask_path)
    background_mask.save(background_mask_path)
    save_preview(cutout, preview_path)

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
        "outputs": {
            "cutout": str(cutout_path),
            "trimmed_cutout": str(trimmed_path),
            "product_mask": str(product_mask_path),
            "background_mask": str(background_mask_path),
            "preview": str(preview_path),
        },
    }

    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[DONE] preprocessing: {trimmed_path}")

    return {
        # 후속 생성 단계에는 여백을 제거한 누끼를 전달
        "cutout": trimmed_path,
        "full_cutout": cutout_path,
        "trimmed_cutout": trimmed_path,
        "product_mask": product_mask_path,
        "background_mask": background_mask_path,
        "preview": preview_path,
        "metadata": metadata_path,
        "bbox": bbox,
    }
