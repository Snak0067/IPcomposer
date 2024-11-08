from fastcomposer.transforms import get_object_transforms
from fastcomposer.data import DemoDataset
from fastcomposer.model import FastComposerModel
from diffusers import StableDiffusionPipeline, AutoPipelineForText2Image
from diffusers.utils import load_image
from transformers import CLIPTokenizer
from accelerate.utils import set_seed
from ipcomposer.utils import parse_args
from accelerate import Accelerator
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import os
from tqdm.auto import tqdm
from fastcomposer.pipeline import (
    stable_diffusion_call_with_references_delayed_conditioning,
)
import types
import itertools
import os
from safetensors import safe_open

def load_model(model, args):
    model_path = Path(args.finetuned_model_path)
    
    if model_path.is_dir():
        # 检查是否为 `.bin` 或 `.safetensors` 文件
        bin_path = model_path / "pytorch_model.bin"
        safetensor_path = model_path / "model.safetensors"

        if bin_path.exists():
            # 以 .bin 文件加载模型
            print("Loading pretrained model (unet, text-encoder, image-encoder) from .bin file...")
            model.load_state_dict(torch.load(bin_path, map_location="cpu"), strict=False)

        elif safetensor_path.exists():
            # 以 .safetensors 文件加载模型
            print("Loading pretrained model (unet, text-encoder, image-encoder) from .safetensors file...")
            with safe_open(safetensor_path, framework="pt", device="cpu") as f:
                main_model_state = {k: f.get_tensor(k) for k in f.keys()}
            model.load_state_dict(main_model_state, strict=False)

        else:
            print("No compatible checkpoint file found in the specified path.")
    else:
        print(f"Specified model path {model_path} is invalid.")

@torch.no_grad()
def main():
    args = parse_args()
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
    )

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # 加载和配置生成管道
    pipe = AutoPipelineForText2Image.from_pretrained(
        args.pretrained_model_name_or_path, torch_dtype=weight_dtype
    )
    pipe.load_ip_adapter("/home/capg_bind/96/zfd/0.hug/h94/IP-Adapter/", subfolder="models", weight_name="ip-adapter_sd15.bin")
    pipe.set_ip_adapter_scale(0.6)
    
    image = load_image(args.test_ip_adapter_image)
    

    # 加载和配置 FastComposerModel
    model = FastComposerModel.from_pretrained(args)

    load_model(model, args)

    model = model.to(device=accelerator.device, dtype=weight_dtype)

    # 将预训练权重替换到实际SD权重
    pipe.unet = model.unet

    if args.enable_xformers_memory_efficient_attention:
        pipe.unet.enable_xformers_memory_efficient_attention()
        
    # 替换训练的 text encoder（原文是冻结的）
    pipe.text_encoder = model.text_encoder
    # image_encoder 是被描述为 optional component，意思是它是一个可选组件，并不一定会在所有管道中使用。
    # 例如，某些自定义的管道可能需要 image_encoder，用于将图像特征输入到生成模型中，以实现图像-文本多模态的融合。
    pipe.image_encoder = model.image_encoder

    pipe.postfuse_module = model.postfuse_module

    pipe.inference = types.MethodType(
        stable_diffusion_call_with_references_delayed_conditioning, pipe
    )

    del model

    pipe = pipe.to(accelerator.device)

    # Set up the dataset
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )
    # 数据增强方式
    object_transforms = get_object_transforms(args)

    demo_dataset = DemoDataset(
        test_caption=args.test_caption,
        test_reference_folder=args.test_reference_folder,
        tokenizer=tokenizer,
        object_transforms=object_transforms,
        device=accelerator.device,
        max_num_objects=args.max_num_objects,
    )

    image_ids = os.listdir(args.test_reference_folder)
    print(f"Image IDs: {image_ids}")
    demo_dataset.set_image_ids(image_ids)

    unique_token = "<|image|>"

    prompt = args.test_caption
    prompt_text_only = prompt.replace(unique_token, "")

    os.makedirs(args.output_dir, exist_ok=True)

    batch = demo_dataset.get_data()

    input_ids = batch["input_ids"].to(accelerator.device)
    text = tokenizer.batch_decode(input_ids)[0]
    print(prompt)
    # print(input_ids)
    image_token_mask = batch["image_token_mask"].to(accelerator.device)

    # print(image_token_mask)
    all_object_pixel_values = (
        batch["object_pixel_values"].unsqueeze(0).to(accelerator.device)
    )
    num_objects = batch["num_objects"].unsqueeze(0).to(accelerator.device)

    all_object_pixel_values = all_object_pixel_values.to(
        dtype=weight_dtype, device=accelerator.device
    )

    object_pixel_values = all_object_pixel_values  # [:, 0, :, :, :]
    if pipe.image_encoder is not None:
        object_embeds = pipe.image_encoder(object_pixel_values)
    else:
        object_embeds = None

    encoder_hidden_states = pipe.text_encoder(
        input_ids, image_token_mask, object_embeds, num_objects
    )[0]

    encoder_hidden_states_text_only = pipe.encode_prompt(
        prompt_text_only,
        accelerator.device,
        args.num_images_per_prompt,
        do_classifier_free_guidance=False,
    )[0]

    encoder_hidden_states = pipe.postfuse_module(
        encoder_hidden_states,
        object_embeds,
        image_token_mask,
        num_objects,
    )

    cross_attention_kwargs = {}

    images = pipe.inference(
        prompt_embeds=encoder_hidden_states,
        num_inference_steps=args.inference_steps,
        height=args.generate_height,
        width=args.generate_width,
        guidance_scale=args.guidance_scale,
        num_images_per_prompt=args.num_images_per_prompt,
        cross_attention_kwargs=cross_attention_kwargs,
        prompt_embeds_text_only=encoder_hidden_states_text_only,
        start_merge_step=args.start_merge_step,
        ip_adapter_image=image,
    ).images

    for instance_id in range(args.num_images_per_prompt):
        images[instance_id].save(
            os.path.join(
                args.output_dir,
                f"output_{instance_id}.png",
            )
        )


if __name__ == "__main__":
    main()
