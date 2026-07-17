import torch
from diffusers import (
    ControlNetModel,
    EulerAncestralDiscreteScheduler,
    StableDiffusionControlNetInpaintPipeline,
)


def load_generation_pipeline(
    base_model,
    controlnet_model,
    cpu_offload=False,
):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU를 찾을 수 없습니다.")

    dtype = torch.float16

    print(f"[ControlNet 로드] {controlnet_model}")
    controlnet = ControlNetModel.from_pretrained(
        controlnet_model,
        torch_dtype=dtype,
    )

    print(f"[생성 모델 로드] {base_model}")
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        base_model,
        controlnet=controlnet,
        torch_dtype=dtype,
        safety_checker=None,
    )

    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipe.scheduler.config
    )

    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()

    if cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")

    return pipe