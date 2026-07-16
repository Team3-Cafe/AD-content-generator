import torch
from diffusers import (
    ControlNetModel, EulerAncestralDiscreteScheduler,
    StableDiffusionControlNetInpaintPipeline,
)


def load_generation_pipeline(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU를 찾지 못했습니다.")

    dtype = torch.float16

    print("[ControlNet 로드]")
    print(args.controlnet_model)

    controlnet = ControlNetModel.from_pretrained(
        args.controlnet_model,
        torch_dtype=dtype,
    )

    print("[생성 모델 로드]")
    print(args.base_model)

    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        args.base_model,
        controlnet=controlnet,
        torch_dtype=dtype,
        safety_checker=None,
    )

    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
        pipe.scheduler.config
    )

    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()

    if args.cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")

    return pipe


def load_identity_pipeline(args):
    controlnet = ControlNetModel.from_pretrained(
        args.controlnet_model,
        torch_dtype=torch.float16,
    )

    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        args.base_model,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None,
    )

    if args.cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")

    return pipe
