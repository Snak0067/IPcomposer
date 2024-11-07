# Part of this script are derived from the official example script of diffusers
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");


import logging
import math
import os
import shutil
import torch
import torch.utils.checkpoint
import sys
import datasets
import diffusers
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from torch.utils.data import Subset
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils.import_utils import is_xformers_available
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from torchvision import transforms
import time
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from ipcomposer.utils import parse_args

from ipcomposer.model import IpComposerModel

from fastcomposer.transforms import (
    get_train_transforms_with_segmap,
    get_object_transforms,
    get_object_processor,
)

from ipcomposer.transforms import get_ip_transform_processor

from ipcomposer.data import get_data_loader, IpComposerDataset
from ip_adapter.ip_adapter import ImageProjModel
from ip_adapter.utils import is_torch2_available
if is_torch2_available():
    from ip_adapter.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor
else:
    from ip_adapter.attention_processor import IPAttnProcessor, AttnProcessor

import itertools
import torch.multiprocessing as mp
import pickle
from safetensors import safe_open
import numpy as np

logger = get_logger(__name__)

# 设置可见设备
os.environ["WANDB_MODE"]="offline"

mp.set_start_method("spawn", force=True)

def save_ipadapter_checkpoint(model, global_step, output_dir, accelerator):
    """
    Save the model checkpoint.
    """
    if accelerator.is_main_process:
        save_path = os.path.join(output_dir, f"checkpoint-{global_step}_ip_adapter.bin")
        state_dict = accelerator.get_state_dict(model)

        # Separate the state dict into different components
        image_proj_sd = {}
        ip_sd = {}
        for k in state_dict:
            if k.startswith("image_proj_model"):
                image_proj_sd[k.replace("image_proj_model.", "")] = state_dict[k]
            elif k.startswith("adapter_modules"):
                ip_sd[k.replace("adapter_modules.", "")] = state_dict[k]
        
        torch.save({"image_proj": image_proj_sd, "ip_adapter": ip_sd}, save_path)
        logger.info(f"ip-adapter model saved at {save_path}")


def resume_checkpoint_by_hand(model, optimizer, scheduler, accelerator, checkpoint_dir):
    # Load main model weights from a safetensors file
    model_path = Path(checkpoint_dir) / "model.safetensors"
    if model_path.exists():
        with safe_open(model_path, framework="pt", device="cpu") as f:
            main_model_state = {k: f.get_tensor(k) for k in f.keys()}
        model.load_state_dict(main_model_state, strict=False)
        logger.info(f"Loaded main model from {model_path}")
    else:
        logger.info(f"Main model checkpoint not found at {model_path}")

    # Load optimizer state
    optimizer_path = Path(checkpoint_dir) / "optimizer.bin"
    if optimizer_path.exists():
        optimizer_state = torch.load(optimizer_path, map_location="cpu")
        optimizer.load_state_dict(optimizer_state)
        logger.info(f"Loaded optimizer state from {optimizer_path}")
    else:
        logger.info(f"Optimizer state not found at {optimizer_path}")

    # Load scheduler state
    scheduler_path = Path(checkpoint_dir) / "scheduler.bin"
    if scheduler_path.exists():
        scheduler_state = torch.load(scheduler_path, map_location="cpu")
        scheduler.load_state_dict(scheduler_state)
        logger.info(f"Loaded scheduler state from {scheduler_path}")
    else:
        logger.info(f"Scheduler state not found at {scheduler_path}")

    # 加载随机状态
    random_states_path = Path(checkpoint_dir) / "random_states_0.pkl"
    if random_states_path.exists():
        random_states = torch.load(random_states_path, map_location="cpu")
        
        # 恢复完整的随机状态
        if "torch_manual_seed" in random_states:
            torch.set_rng_state(random_states["torch_manual_seed"])
            logger.info("Restored torch RNG state.")

        if "torch_cuda_manual_seed" in random_states:
            torch.cuda.set_rng_state_all(random_states["torch_cuda_manual_seed"])
            logger.info("Restored CUDA RNG state.")
        
        if "numpy_random_seed" in random_states:
            np.random.set_state(random_states["numpy_random_seed"])
            logger.info("Restored numpy random state.")
        
        logger.info(f"成功加载并恢复随机状态文件：{random_states_path}")
    else:
        logger.info(f"随机状态文件未找到：{random_states_path}")

    # 加载 scaler 状态用于混合精度训练（如果适用）
    scaler_path = Path(checkpoint_dir) / "scaler.pt"
    if scaler_path.exists():
        scaler_state = torch.load(scaler_path, map_location="cpu")
        accelerator.scaler.load_state_dict(scaler_state)
        logger.info(f"Loaded scaler state from {scaler_path}")

    # 准备模型、优化器和调度器
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    return model, optimizer, scheduler
    

def train():
    args = parse_args()

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_dir=args.logging_dir,
    )
    logger.info(f"Process rank: {accelerator.process_index}, using device: {accelerator.device}")

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
        if args.logging_dir is not None:
            os.makedirs(args.logging_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Make one log on every process with the configuration for debugging.
    t = time.localtime()
    str_m_d_y_h_m_s = time.strftime("%m-%d-%Y_%H-%M-%S", t)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                os.path.join(args.logging_dir, f"{str_m_d_y_h_m_s}.log")
            ),
        ]
        if accelerator.is_main_process
        else [],
    )
    # 用于写入loss损失，防止wandb不能连接 损失消失
    loss_log_path = os.path.join(args.logging_dir, f"{str_m_d_y_h_m_s}_loss_log.txt")
    if accelerator.is_main_process:
        with open(loss_log_path, "a") as loss_log_file:
            loss_log_file.write("Step\tTrain Loss\tDenoise Loss\tLocalization Loss\n")
    
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
    )
    # 从本地路径SD加载预训练权重，同时定义IP-adapter 投影层、替换unet中的cross-Attn
    model = IpComposerModel.from_pretrained(args) 

    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    else:
        weight_dtype = torch.float32

    # freeze all params in the model
    for param in model.parameters():
        param.requires_grad = False
        param.data = param.data.to(weight_dtype)

    if args.load_model is not None:
        model_path = Path(args.load_model) / "model.safetensors"
        if model_path.exists():
            with safe_open(model_path, framework="pt", device="cpu") as f:
                main_model_state = {k: f.get_tensor(k) for k in f.keys()}
            model.load_state_dict(main_model_state, strict=False)
            logger.info(f"Loaded main model from {model_path}")
        else:
            logger.info(f"Main model checkpoint not found at {model_path}")
        logger.info(f" checkpoint模型读取成功.... ")
            
    # 将 UNet 模块的所有参数设置为冻结
    model.unet.requires_grad_(False)
    model.unet.to(torch.float32)
    model.postfuse_module.requires_grad_(False)
    model.text_encoder.requires_grad_(False)
    model.image_encoder.requires_grad_(False)
    # Postfuse 模块（后融合模块）处理文本和图像特征融合的模块
    
    model.image_encoder.to(torch.float32)
    model.postfuse_module.to(torch.float32)
    model.text_encoder.to(torch.float32)
    
    if args.train_ip_adapter:
        # 训练 ip_adapter 的 image_proj_model 参数
        model.image_proj_model.requires_grad_(True) 
        model.image_proj_model.to(torch.float32)
        # 训练 ip_adapter 的 adapter_modules 参数
        model.adapter_modules.requires_grad_(True) 
        model.adapter_modules.to(torch.float32)
    

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            model.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )


    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    parameters = itertools.chain(model.image_proj_model.parameters(), model.adapter_modules.parameters())
    
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=args.adam_weight_decay)
    

    # 对图像和分割图（segmentation map）同时进行处理
    train_transforms = get_train_transforms_with_segmap(args)
    # 数据增强和变换流程，并返回一个包含所有变换的序列
    object_transforms = get_object_transforms(args)
    # 如果 args.object_background_processor 为 "random"，则生成随机背景
    object_processor = get_object_processor(args)

    if args.object_types is None or args.object_types == "all":
        object_types = None  # all object types
    else:
        object_types = args.object_types.split("_")
        logger.info(f"Using object types: {object_types}")

    train_dataset = IpComposerDataset(
        args.dataset_name,
        tokenizer,
        train_transforms,
        object_transforms,
        object_processor,
        device=accelerator.device,
        max_num_objects=args.max_num_objects,
        num_image_tokens=args.num_image_tokens,
        object_appear_prob=args.object_appear_prob,
        uncondition_prob=args.uncondition_prob,
        text_only_prob=args.text_only_prob,
        object_types=None,  # 设置不带任何类别的条件过滤
        split="train",
        min_num_objects=args.min_num_objects,
        balance_num_objects=args.balance_num_objects,
    )

    train_dataloader = get_data_loader(train_dataset, args.train_batch_size, num_workers=4)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    if args.use_ema:
        accelerator.register_for_checkpointing(model.module.ema_param)
        model.module.ema_param.to(accelerator.device)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("IpComposer", config=vars(args))

    # Train!
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(f"Trainable parameter: {name} with shape {param.shape}")

    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        logger.info(f" 恢复断点模型")
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            # accelerator.load_state(os.path.join(args.output_dir, path))
            # replace by load checkpoint by hand
            checkpoint_path = os.path.join(args.output_dir, path)
            model, optimizer, lr_scheduler = resume_checkpoint_by_hand(model, optimizer, lr_scheduler, accelerator, checkpoint_path)
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (
                num_update_steps_per_epoch * args.gradient_accumulation_steps
            )

            # move all the state to the correct device
            model.to(accelerator.device)
            if args.use_ema:
                model.module.ema_param.to(accelerator.device)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(global_step, args.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        logger.info(f" 进入新的epoch.... ")
        model.train()
        train_loss = 0.0
        denoise_loss = 0.0
        localization_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            progress_bar.set_description("Global step: {}".format(global_step))

            with torch.backends.cuda.sdp_kernel(enable_flash=True):  # Explicitly enable Flash Attention
                
                return_dict = model(batch, noise_scheduler)
                loss = return_dict["loss"]

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                avg_denoise_loss = accelerator.gather(
                    return_dict["denoise_loss"].repeat(args.train_batch_size)
                ).mean()
                denoise_loss += (
                    avg_denoise_loss.item() / args.gradient_accumulation_steps
                )

                if "localization_loss" in return_dict:
                    avg_localization_loss = accelerator.gather(
                        return_dict["localization_loss"].repeat(args.train_batch_size)
                    ).mean()
                    localization_loss += (
                        avg_localization_loss.item() / args.gradient_accumulation_steps
                    )

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(parameters, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    model.module.ema_param.step(model.module.unet.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log(
                    {
                        "train_loss": train_loss,
                        "denoise_loss": denoise_loss,
                        "localization_loss": localization_loss,
                    },
                    step=global_step,
                )
                
                # 写入损失到文本文件
                if accelerator.is_main_process:
                    with open(loss_log_path, "a") as loss_log_file:
                        loss_log_file.write(
                            f"{global_step}\t{train_loss:.6f}\t{denoise_loss:.6f}\t{localization_loss:.6f}\n"
                        )
    
                train_loss = 0.0
                denoise_loss = 0.0
                localization_loss = 0.0

                if (global_step % args.checkpointing_steps == 0 and accelerator.is_local_main_process):
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    # accelerator.save_state(save_path)
                    save_ipadapter_checkpoint(model, global_step, save_path, accelerator)
                    logger.info(f"Saved state to {save_path}")
                    # if args.keep_only_last_checkpoint:
                    #     # Remove all other checkpoints
                    #     for file in os.listdir(args.output_dir):
                    #         if file.startswith("checkpoint") and file != os.path.basename(save_path):
                    #             ckpt_num = int(file.split("-")[1])
                    #             if (args.keep_interval is None or ckpt_num % args.keep_interval != 0):
                    #                 logger.info(f"Removing {file}")
                    #                 shutil.rmtree(os.path.join(args.output_dir, file))

            logs = {
                "l_noise": return_dict["denoise_loss"].detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }

            if "localization_loss" in return_dict:
                logs["l_loc"] = return_dict["localization_loss"].detach().item()

            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        model = accelerator.unwrap_model(model)
        if args.use_ema:
            model.ema_param.copy_to(model.unet.parameters())

        pipeline = model.to_pipeline()
        pipeline.save_pretrained(args.output_dir)

    accelerator.end_training()


if __name__ == "__main__":
    train()
