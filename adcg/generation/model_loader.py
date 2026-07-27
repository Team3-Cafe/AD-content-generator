import torch
from diffusers import (
    ControlNetModel,
    EulerAncestralDiscreteScheduler,
    StableDiffusionControlNetInpaintPipeline,
)


def _get_torch_dtype():
    if torch.cuda.is_available():
        return torch.float16

    return torch.float32


def load_controlnet(
    model_id,
    torch_dtype=None,
):
    if torch_dtype is None:
        torch_dtype = _get_torch_dtype()

    print(f"[ControlNet load] {model_id}")

    return ControlNetModel.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
    )


def pipeline_controlnet_count(pipe):
    """Return the number of ControlNets attached to a diffusion pipeline."""
    controlnet = getattr(pipe, "controlnet", None)
    networks = getattr(controlnet, "nets", None)

    if networks is not None:
        return len(networks)
    if controlnet is not None:
        return 1
    return 0


def load_generation_pipeline(
    base_model,
    controlnet_model,
    depth_controlnet_model=None,
    cpu_offload=False,
):
    torch_dtype = _get_torch_dtype()

    canny_controlnet = load_controlnet(
        controlnet_model,
        torch_dtype=torch_dtype,
    )

    if depth_controlnet_model:
        depth_controlnet = load_controlnet(
            depth_controlnet_model,
            torch_dtype=torch_dtype,
        )

        controlnet = [
            canny_controlnet,
            depth_controlnet,
        ]
    else:
        controlnet = canny_controlnet

    print(f"[Generation model load] {base_model}")

    pipe = (
        StableDiffusionControlNetInpaintPipeline
        .from_pretrained(
            base_model,
            controlnet=controlnet,
            torch_dtype=torch_dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )
    )

    pipe.scheduler = (
        EulerAncestralDiscreteScheduler.from_config(
            pipe.scheduler.config
        )
    )

    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()

    if torch.cuda.is_available():
        if cpu_offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")
    else:
        pipe.to("cpu")

    return pipe


def load_controlnet_inpaint_pipeline(
    base_model,
    controlnet_model,
    cpu_offload=False,
):
    """기존 refinement 코드와의 호환용 loader."""
    return load_generation_pipeline(
        base_model=base_model,
        controlnet_model=controlnet_model,
        depth_controlnet_model=None,
        cpu_offload=cpu_offload,
    )
