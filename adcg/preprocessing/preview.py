from PIL import Image


def save_preview(cutout, output_path):
    cutout = cutout.convert("RGBA")

    background = Image.new(
        "RGBA",
        cutout.size,
        (255, 255, 255, 255),
    )

    preview = Image.alpha_composite(
        background,
        cutout,
    )

    preview.convert("RGB").save(output_path)