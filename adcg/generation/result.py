import json
from pathlib import Path


def _json_safe(value):
    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {
            key: _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]

    return value


def save_generation_result(
    output_dir,
    product,
    canvas_result,
    inpaint_mask,
    control_image,
    generated_image,
    experiment_data,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "product_cutout": (
            output_dir / "product_cutout_trimmed.png"
        ),
        "product_resized": (
            output_dir / "product_resized.png"
        ),
        "product_layer": (
            output_dir / "product_layer.png"
        ),
        "condition_canvas": (
            output_dir / "condition_canvas.png"
        ),
        "product_mask": (
            output_dir / "product_alpha_mask.png"
        ),
        "background_mask": (
            output_dir / "background_inpaint_mask.png"
        ),
        "canny_control": (
            output_dir / "product_canny_control.png"
        ),
        "generated": (
            output_dir / "generated_with_cutout_condition.png"
        ),
        "final": output_dir / "final.png",
        "result_json": (
            output_dir / "experiment_result.json"
        ),
    }

    product.save(paths["product_cutout"])
    canvas_result["resized_product"].save(
        paths["product_resized"]
    )
    canvas_result["product_layer"].save(
        paths["product_layer"]
    )
    canvas_result["condition_canvas"].save(
        paths["condition_canvas"]
    )
    canvas_result["product_mask"].save(
        paths["product_mask"]
    )
    inpaint_mask.save(paths["background_mask"])
    control_image.save(paths["canny_control"])
    generated_image.save(paths["generated"])
    generated_image.save(paths["final"])

    experiment_data["outputs"] = {
        key: str(path)
        for key, path in paths.items()
        if key != "result_json"
    }

    paths["result_json"].write_text(
        json.dumps(
            _json_safe(experiment_data),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "image": paths["final"],
        "generated_image": paths["generated"],
        "product_mask": paths["product_mask"],
        "condition_canvas": paths["condition_canvas"],
        "canny_control": paths["canny_control"],
        "background_mask": paths["background_mask"],
        "result_json": paths["result_json"],
    }