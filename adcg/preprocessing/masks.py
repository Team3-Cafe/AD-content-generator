from PIL import Image


def make_masks(cutout):
    alpha = cutout.convert("RGBA").getchannel("A")

    # 흰색: 상품 / 검정색: 배경
    product_mask = alpha

    # 흰색: 생성할 배경 / 검정색: 보존할 상품
    background_mask = Image.eval(
        alpha,
        lambda value: 255 - value,
    )

    return product_mask, background_mask