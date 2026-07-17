import numpy as np
from PIL import Image


def clean_alpha(image, threshold=8):
    image = image.convert("RGBA")
    array = np.array(image)
    alpha = array[:, :, 3]

    alpha[alpha < threshold] = 0
    array[:, :, 3] = alpha

    return Image.fromarray(array, mode="RGBA")


def neutralize_transparent_pixels(image):
    """투명 영역의 RGB를 흰색으로 변경해 검정 테두리를 방지한다."""
    image = image.convert("RGBA")
    array = np.array(image)

    transparent = array[:, :, 3] == 0
    array[transparent, :3] = 255

    return Image.fromarray(array, mode="RGBA")