import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from threading import Semaphore
from transformers import DPTForDepthEstimation, DPTImageProcessor


_DEPTH_PROCESSOR = None
_DEPTH_MODEL = None
_DEPTH_MODEL_ID = None
_DEPTH_DEVICE = None
_DEPTH_INFERENCE_SEMAPHORE = Semaphore(1)


def create_canny_control(
    product_layer,
    alpha_mask,
    low_threshold=100,
    high_threshold=200,
    suppress_radius=3,
):
    product_array = np.array(
        product_layer.convert("RGB")
    )
    alpha_array = np.array(
        alpha_mask.convert("L")
    )

    edges = cv2.Canny(
        product_array,
        int(low_threshold),
        int(high_threshold),
    )

    edges[alpha_array == 0] = 0

    if suppress_radius > 0:
        kernel_size = suppress_radius * 2 + 1
        kernel = np.ones(
            (kernel_size, kernel_size),
            dtype=np.uint8,
        )

        alpha_binary = (
            (alpha_array > 0).astype(np.uint8) * 255
        )

        inner_mask = cv2.erode(
            alpha_binary,
            kernel,
            iterations=1,
        )

        edges[inner_mask == 0] = 0

    control_array = np.stack(
        [edges, edges, edges],
        axis=-1,
    )

    return Image.fromarray(
        control_array,
        mode="RGB",
    )


def _resolve_depth_device(device):
    device = str(device or "cpu")

    if device == "cuda" and not torch.cuda.is_available():
        return "cpu"

    return device


def _load_depth_estimator(
    model_id="Intel/dpt-hybrid-midas",
    device="cpu",
):
    global _DEPTH_PROCESSOR
    global _DEPTH_MODEL
    global _DEPTH_MODEL_ID
    global _DEPTH_DEVICE

    device = _resolve_depth_device(device)

    should_reload = (
        _DEPTH_PROCESSOR is None
        or _DEPTH_MODEL is None
        or _DEPTH_MODEL_ID != model_id
    )

    if should_reload:
        print(f"[Depth estimator load] {model_id}")

        _DEPTH_PROCESSOR = (
            DPTImageProcessor.from_pretrained(model_id)
        )
        _DEPTH_MODEL = (
            DPTForDepthEstimation.from_pretrained(model_id)
        )
        _DEPTH_MODEL_ID = model_id
        _DEPTH_DEVICE = None

    if _DEPTH_DEVICE != device:
        _DEPTH_MODEL.to(device)
        _DEPTH_DEVICE = device

    _DEPTH_MODEL.eval()

    return (
        _DEPTH_PROCESSOR,
        _DEPTH_MODEL,
        device,
    )


def _create_depth_control(
    image,
    model_id="Intel/dpt-hybrid-midas",
    device="cpu",
):
    image = image.convert("RGB")

    processor, model, device = _load_depth_estimator(
        model_id=model_id,
        device=device,
    )

    inputs = processor(
        images=image,
        return_tensors="pt",
    )

    pixel_values = inputs["pixel_values"].to(device)

    with torch.inference_mode():
        predicted_depth = model(
            pixel_values=pixel_values
        ).predicted_depth

    predicted_depth = F.interpolate(
        predicted_depth.unsqueeze(1),
        size=(image.height, image.width),
        mode="bicubic",
        align_corners=False,
    ).squeeze()

    depth_array = (
        predicted_depth
        .detach()
        .float()
        .cpu()
        .numpy()
    )

    minimum = float(depth_array.min())
    maximum = float(depth_array.max())
    difference = maximum - minimum

    if difference < 1e-8:
        normalized = np.zeros_like(
            depth_array,
            dtype=np.uint8,
        )
    else:
        normalized = (
            (depth_array - minimum)
            / difference
            * 255.0
        ).clip(0, 255).astype(np.uint8)

    return Image.fromarray(
        normalized,
        mode="L",
    ).convert("RGB")


def create_depth_control(
    image,
    model_id="Intel/dpt-hybrid-midas",
    device="cpu",
):
    """Run one cached depth-estimator inference at a time."""
    with _DEPTH_INFERENCE_SEMAPHORE:
        return _create_depth_control(
            image=image,
            model_id=model_id,
            device=device,
        )
