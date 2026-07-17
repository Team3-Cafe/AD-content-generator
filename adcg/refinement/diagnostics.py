import json
from pathlib import Path

import numpy as np
from PIL import Image


def prepare_output_dir(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_array(array, path, scale_float=False):
    array = np.asarray(array)

    if scale_float:
        array = array * 255.0

    array = np.clip(array, 0, 255).astype(np.uint8)
    Image.fromarray(array).save(path)


def save_diagnostics(output_dir, images):
    output_dir = prepare_output_dir(output_dir)

    for filename, image in images.items():
        path = output_dir / filename

        if isinstance(image, Image.Image):
            image.save(path)
        else:
            save_array(image, path)


def save_metadata(output_dir, filename, metadata):
    output_dir = prepare_output_dir(output_dir)

    path = output_dir / filename
    path.write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    return path