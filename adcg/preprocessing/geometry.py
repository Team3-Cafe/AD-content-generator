def crop_to_product(image, padding=10):
    image = image.convert("RGBA")
    alpha = image.getchannel("A")
    bbox = alpha.getbbox()

    if bbox is None:
        raise RuntimeError("상품 영역을 찾지 못했습니다.")

    left, top, right, bottom = bbox

    left = max(0, left - padding)
    top = max(0, top - padding)
    right = min(image.width, right + padding)
    bottom = min(image.height, bottom + padding)

    crop_bbox = (left, top, right, bottom)

    return image.crop(crop_bbox), crop_bbox