import json
from pathlib import Path


def save_generation_result(
    output_dir,
    product,
    canvas_result,
    inpaint_mask,
    control_image,
    generated_image,
    experiment_data,
    depth_control_image=None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    product_path = (
        output_dir / "product_cutout_trimmed.png"
    )
    resized_path = (
        output_dir / "product_resized.png"
    )
    product_layer_path = (
        output_dir / "product_layer.png"
    )
    condition_canvas_path = (
        output_dir / "condition_canvas.png"
    )
    product_mask_path = (
        output_dir / "product_alpha_mask.png"
    )
    inpaint_mask_path = (
        output_dir / "background_inpaint_mask.png"
    )
    canny_path = (
        output_dir / "product_canny_control.png"
    )
    depth_path = (
        output_dir / "product_depth_control.png"
    )
    generated_path = (
        output_dir
        / "generated_with_cutout_condition.png"
    )
    final_path = output_dir / "final.png"
    metadata_path = (
        output_dir / "experiment_result.json"
    )

    product.convert("RGBA").save(product_path)

    canvas_result["resized_product"].save(
        resized_path
    )
    canvas_result["product_layer"].save(
        product_layer_path
    )
    canvas_result["condition_canvas"].save(
        condition_canvas_path
    )
    canvas_result["product_mask"].save(
        product_mask_path
    )

    inpaint_mask.convert("L").save(
        inpaint_mask_path
    )
    control_image.convert("RGB").save(
        canny_path
    )

    if depth_control_image is not None:
        depth_control_image.convert("RGB").save(
            depth_path
        )
        saved_depth_path = depth_path
    else:
        saved_depth_path = None

    generated_image.convert("RGB").save(
        generated_path
    )
    generated_image.convert("RGB").save(
        final_path
    )

    experiment_data = dict(experiment_data)

    experiment_data["outputs"] = {
        "product": str(product_path),
        "resized_product": str(resized_path),
        "product_layer": str(product_layer_path),
        "condition_canvas": str(
            condition_canvas_path
        ),
        "product_mask": str(product_mask_path),
        "inpaint_mask": str(inpaint_mask_path),
        "canny_control": str(canny_path),
        "depth_control": (
            str(saved_depth_path)
            if saved_depth_path is not None
            else None
        ),
        "generated": str(generated_path),
        "final": str(final_path),
    }

    metadata_path.write_text(
        json.dumps(
            experiment_data,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[DONE] generated: {final_path}")

    return {
        "image": final_path,
        "generated_image": generated_path,
        "product_layer": product_layer_path,
        "product_mask": product_mask_path,
        "condition_canvas": condition_canvas_path,
        "inpaint_mask": inpaint_mask_path,
        "canny_control": canny_path,
        "depth_control": saved_depth_path,
        "metadata": metadata_path,
    }