import torch
import torch.nn as nn
from diffusers import AutoencoderKL, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.pipelines.controlnet import MultiControlNetModel
from transformers import CLIPTextModel
import torch.nn.functional as F
from typing import Any, Optional, Tuple, Union, Dict, List
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.clip.modeling_clip import (
    _expand_mask,
    CLIPTextTransformer,
    CLIPPreTrainedModel,
    CLIPModel,
)

import types
import torchvision.transforms as T
import gc
import numpy as np

import os
from safetensors import safe_open
from ip_adapter.attention_processor import IPAttnProcessor
from ip_adapter.utils import is_torch2_available, get_generator
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from PIL import Image
if is_torch2_available():
    from ip_adapter.attention_processor import (
        AttnProcessor2_0 as AttnProcessor,
    )
    from ip_adapter.attention_processor import (
        CNAttnProcessor2_0 as CNAttnProcessor,
    )
    from ip_adapter.attention_processor import (
        IPAttnProcessor2_0 as IPAttnProcessor,
    )
else:
    from ip_adapter.attention_processor import AttnProcessor, CNAttnProcessor, IPAttnProcessor
from ip_adapter.resampler import Resampler
from ip_adapter.ip_adapter import ImageProjModel

inference_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, use_residual=True):
        super().__init__()
        if use_residual:
            assert in_dim == out_dim
        self.layernorm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.use_residual = use_residual
        self.act_fn = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.layernorm(x)
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.fc2(x)
        if self.use_residual:
            x = x + residual
        return x

class IpImageProjModel(torch.nn.Module):
    """Projection Model"""

    def __init__(self, cross_attention_dim=1024, clip_embeddings_dim=1024, clip_extra_context_tokens=4):
        super().__init__()

        self.generator = None
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim)
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        embeds = image_embeds
        clip_extra_context_tokens = self.proj(embeds).reshape(
            -1, self.clip_extra_context_tokens, self.cross_attention_dim
        )
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens

class IpComposerCLIPImageEncoder(CLIPPreTrainedModel):
    @staticmethod
    def from_pretrained(
        global_model_name_or_path,
    ):
        model = CLIPModel.from_pretrained(global_model_name_or_path)
        vision_model = model.vision_model
        visual_projection = model.visual_projection
        vision_processor = T.Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711),
        )
        return IpComposerCLIPImageEncoder(
            vision_model,
            visual_projection,
            vision_processor,
        )

    def __init__(
        self,
        vision_model,
        visual_projection,
        vision_processor,
    ):
        super().__init__(vision_model.config)
        self.vision_model = vision_model
        self.visual_projection = visual_projection
        self.vision_processor = vision_processor

        self.image_size = vision_model.config.image_size

    def forward(self, object_pixel_values):
        b, num_objects, c, h, w = object_pixel_values.shape

        object_pixel_values = object_pixel_values.view(b * num_objects, c, h, w)

        if h != self.image_size or w != self.image_size:
            h, w = self.image_size, self.image_size
            object_pixel_values = F.interpolate(
                object_pixel_values, (h, w), mode="bilinear", antialias=True
            )

        object_pixel_values = self.vision_processor(object_pixel_values)
        object_embeds = self.vision_model(object_pixel_values)[1]
        object_embeds = self.visual_projection(object_embeds)
        object_embeds = object_embeds.view(b, num_objects, 1, -1)
        return object_embeds


def scatter_object_embeddings(
    inputs_embeds,
    image_token_mask,
    object_embeds,
    num_objects,
    image_embedding_transform=None,
):
    object_embeds = object_embeds.to(inputs_embeds.dtype)

    batch_size, max_num_objects = object_embeds.shape[:2]
    seq_length = inputs_embeds.shape[1]
    flat_object_embeds = object_embeds.view(
        -1, object_embeds.shape[-2], object_embeds.shape[-1]
    )

    valid_object_mask = (
        torch.arange(max_num_objects, device=flat_object_embeds.device)[None, :]
        < num_objects[:, None]
    )

    valid_object_embeds = flat_object_embeds[valid_object_mask.flatten()]

    if image_embedding_transform is not None:
        valid_object_embeds = image_embedding_transform(valid_object_embeds)

    inputs_embeds = inputs_embeds.view(-1, inputs_embeds.shape[-1])
    image_token_mask = image_token_mask.view(-1)
    valid_object_embeds = valid_object_embeds.view(-1, valid_object_embeds.shape[-1])

    inputs_embeds.masked_scatter_(image_token_mask[:, None], valid_object_embeds)
    inputs_embeds = inputs_embeds.view(batch_size, seq_length, -1)
    return inputs_embeds


def fuse_object_embeddings(
    inputs_embeds,
    image_token_mask,
    object_embeds,
    num_objects,
    fuse_fn=torch.add,
):
    object_embeds = object_embeds.to(inputs_embeds.dtype)

    batch_size, max_num_objects = object_embeds.shape[:2]
    seq_length = inputs_embeds.shape[1]
    flat_object_embeds = object_embeds.view(
        -1, object_embeds.shape[-2], object_embeds.shape[-1]
    )

    valid_object_mask = (
        torch.arange(max_num_objects, device=flat_object_embeds.device)[None, :]
        < num_objects[:, None]
    )

    valid_object_embeds = flat_object_embeds[valid_object_mask.flatten()]

    inputs_embeds = inputs_embeds.view(-1, inputs_embeds.shape[-1])
    image_token_mask = image_token_mask.view(-1)
    valid_object_embeds = valid_object_embeds.view(-1, valid_object_embeds.shape[-1])

    # slice out the image token embeddings
    image_token_embeds = inputs_embeds[image_token_mask]
    valid_object_embeds = fuse_fn(image_token_embeds, valid_object_embeds)

    inputs_embeds.masked_scatter_(image_token_mask[:, None], valid_object_embeds)
    inputs_embeds = inputs_embeds.view(batch_size, seq_length, -1)
    return inputs_embeds


class IpComposerPostfuseModule(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.mlp1 = MLP(embed_dim * 2, embed_dim, embed_dim, use_residual=False)
        self.mlp2 = MLP(embed_dim, embed_dim, embed_dim, use_residual=True)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def fuse_fn(self, text_embeds, object_embeds):
        text_object_embeds = torch.cat([text_embeds, object_embeds], dim=-1)
        text_object_embeds = self.mlp1(text_object_embeds) + text_embeds
        text_object_embeds = self.mlp2(text_object_embeds)
        text_object_embeds = self.layer_norm(text_object_embeds)
        return text_object_embeds

    def forward(
        self,
        text_embeds,
        object_embeds,
        image_token_mask,
        num_objects,
    ) -> torch.Tensor:
        text_object_embeds = fuse_object_embeddings(
            text_embeds, image_token_mask, object_embeds, num_objects, self.fuse_fn
        )

        return text_object_embeds


class IpComposerTextEncoder(CLIPPreTrainedModel):
    _build_causal_attention_mask = CLIPTextTransformer._build_causal_attention_mask

    @staticmethod
    def from_pretrained(model_name_or_path, **kwargs):
        model = CLIPTextModel.from_pretrained(model_name_or_path, **kwargs)
        text_model = model.text_model
        return IpComposerTextEncoder(text_model)

    def __init__(self, text_model):
        super().__init__(text_model.config)
        self.config = text_model.config
        self.final_layer_norm = text_model.final_layer_norm
        self.embeddings = text_model.embeddings
        self.encoder = text_model.encoder

    def forward(
        self,
        input_ids,
        image_token_mask=None,
        object_embeds=None,
        num_objects=None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        hidden_states = self.embeddings(input_ids)

        bsz, seq_len = input_shape
        causal_attention_mask = self._build_causal_attention_mask(
            bsz, seq_len, hidden_states.dtype
        ).to(hidden_states.device)

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, hidden_states.dtype)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        # text_embeds.shape = [batch_size, sequence_length, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
        pooled_output = last_hidden_state[
            torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
            input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(
                dim=-1
            ),
        ]

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


def unet_store_cross_attention_scores(unet, attention_scores, layers=5):
    from diffusers.models.attention_processor import (
        Attention,
        AttnProcessor,
        AttnProcessor2_0,
    )
    """
    参数解释
        unet: 要处理的 UNet 模型。
        attention_scores: 一个空的字典，用于存储指定层的交叉注意力分数，结构为 {层名称: 注意力分数}。
        layers: 指定需要存储交叉注意力分数的层的数量，默认为 5
    """
    # UNet 中的关键层名称，包括下采样（down_blocks）、中间层（mid_block）和上采样（up_blocks）
    UNET_LAYER_NAMES = [
        "down_blocks.0",
        "down_blocks.1",
        "down_blocks.2",
        "mid_block",
        "up_blocks.1",
        "up_blocks.2",
        "up_blocks.3",
    ]
    start_layer = (len(UNET_LAYER_NAMES) - layers) // 2
    end_layer = start_layer + layers
    applicable_layers = UNET_LAYER_NAMES[start_layer:end_layer]

    def make_new_get_attention_scores_fn(name):
        
        def new_get_attention_scores(module, query, key, attention_mask=None):

            attention_probs = module.old_get_attention_scores(query, key, attention_mask)
            
            # get attention map
            if isinstance(module.processor, IPAttnProcessor):
                attention_scores[name] = attention_probs  
            elif isinstance(module.processor, AttnProcessor):
                attention_scores[name] = attention_probs

            return attention_probs

        return new_get_attention_scores

    for name, module in unet.named_modules():
        # 如果模块是 Attention 实例并且名称包含 "attn2"（通常表示交叉注意力层）
        if isinstance(module, Attention) and "attn2" in name:
            if not any(layer in name for layer in applicable_layers):
                continue
            
            if isinstance(module.processor, AttnProcessor2_0) and not isinstance(module.processor, IPAttnProcessor):
                module.set_processor(AttnProcessor())
                print(f"Set AttnProcessor for layer: {name}")
            elif isinstance(module.processor, IPAttnProcessor):
                print(f"Set IPAttnProcessor for layer: {name}")
                
            module.old_get_attention_scores = module.get_attention_scores
            module.get_attention_scores = types.MethodType(
                make_new_get_attention_scores_fn(name), module
            )

    return unet



class BalancedL1Loss(nn.Module):
    def __init__(self, threshold=1.0, normalize=False):
        super().__init__()
        self.threshold = threshold
        self.normalize = normalize

    def forward(self, object_token_attn_prob, object_segmaps):
        if self.normalize:
            object_token_attn_prob = object_token_attn_prob / (
                object_token_attn_prob.max(dim=2, keepdim=True)[0] + 1e-5
            )
        background_segmaps = 1 - object_segmaps
        background_segmaps_sum = background_segmaps.sum(dim=2) + 1e-5
        object_segmaps_sum = object_segmaps.sum(dim=2) + 1e-5

        background_loss = (object_token_attn_prob * background_segmaps).sum(
            dim=2
        ) / background_segmaps_sum

        object_loss = (object_token_attn_prob * object_segmaps).sum(
            dim=2
        ) / object_segmaps_sum

        return background_loss - object_loss


def get_object_localization_loss_for_one_layer(
    cross_attention_scores,
    object_segmaps,
    object_token_idx,
    object_token_idx_mask,
    loss_fn,
):
    bxh, num_noise_latents, num_text_tokens = cross_attention_scores.shape
    b, max_num_objects, _, _ = object_segmaps.shape
    size = int(num_noise_latents**0.5)

    # Resize the object segmentation maps to the size of the cross attention scores
    object_segmaps = F.interpolate(
        object_segmaps, size=(size, size), mode="bilinear", antialias=True
    )  # (b, max_num_objects, size, size)

    object_segmaps = object_segmaps.view(
        b, max_num_objects, -1
    )  # (b, max_num_objects, num_noise_latents)

    num_heads = bxh // b

    cross_attention_scores = cross_attention_scores.view(
        b, num_heads, num_noise_latents, num_text_tokens
    )

    # Gather object_token_attn_prob
    object_token_attn_prob = torch.gather(
        cross_attention_scores,
        dim=3,
        index=object_token_idx.view(b, 1, 1, max_num_objects).expand(
            b, num_heads, num_noise_latents, max_num_objects
        ),
    )  # (b, num_heads, num_noise_latents, max_num_objects)

    object_segmaps = (
        object_segmaps.permute(0, 2, 1)
        .unsqueeze(1)
        .expand(b, num_heads, num_noise_latents, max_num_objects)
    )

    loss = loss_fn(object_token_attn_prob, object_segmaps)

    loss = loss * object_token_idx_mask.view(b, 1, max_num_objects)
    object_token_cnt = object_token_idx_mask.sum(dim=1).view(b, 1) + 1e-5
    loss = (loss.sum(dim=2) / object_token_cnt).mean()

    return loss


def get_object_localization_loss(
    cross_attention_scores,
    object_segmaps,
    image_token_idx,
    image_token_idx_mask,
    loss_fn,
):
    num_layers = len(cross_attention_scores)
    loss = 0
    for k, v in cross_attention_scores.items():
        layer_loss = get_object_localization_loss_for_one_layer(
            v, object_segmaps, image_token_idx, image_token_idx_mask, loss_fn
        )
        loss += layer_loss
    return loss / num_layers


class IpComposerModel(nn.Module):
    def __init__(self, text_encoder, image_encoder, vae, unet, image_proj_model, adapter_modules, ip_image_encoder, args):
        super().__init__()
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.vae = vae
        self.unet = unet
        self.use_ema = False
        self.ema_param = None
        self.pretrained_model_name_or_path = args.pretrained_model_name_or_path
        self.revision = args.revision
        self.non_ema_revision = args.non_ema_revision
        self.object_localization = args.object_localization
        self.object_localization_weight = args.object_localization_weight
        self.localization_layers = args.localization_layers
        self.mask_loss = args.mask_loss
        self.mask_loss_prob = args.mask_loss_prob

        embed_dim = text_encoder.config.hidden_size
        # self.postfuse_module: 一个用于融合文本和图像嵌入的模块
        self.postfuse_module = IpComposerPostfuseModule(embed_dim)
        
        # TODO:cross_attention_scores目前数量为0，取不出来
        if self.object_localization:
            self.cross_attention_scores = {}
            self.unet = unet_store_cross_attention_scores(
                self.unet, self.cross_attention_scores, self.localization_layers # 5
            )
            self.object_localization_loss_fn = BalancedL1Loss(
                args.object_localization_threshold,
                args.object_localization_normalize,
            )
    
    # ************************ ip adapter code below ************************ #
        self.image_proj_model = image_proj_model
        self.adapter_modules = adapter_modules
        self.ip_image_encoder = ip_image_encoder
        if args.pretrained_ip_adapter_path is not None:
            self.load_from_checkpoint(args.pretrained_ip_adapter_path)
            
    def load_from_checkpoint(self, ckpt_path: str):
        # Calculate original checksums
        orig_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        orig_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        state_dict = torch.load(ckpt_path, map_location="cpu")

        # Load state dict for image_proj_model and adapter_modules
        self.image_proj_model.load_state_dict(state_dict["image_proj"], strict=True)
        self.adapter_modules.load_state_dict(state_dict["ip_adapter"], strict=True)

        # Calculate new checksums
        new_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        new_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        # Verify if the weights have changed
        assert orig_ip_proj_sum != new_ip_proj_sum, "Weights of image_proj_model did not change!"
        assert orig_adapter_sum != new_adapter_sum, "Weights of adapter_modules did not change!"

        print(f"Successfully loaded ip-adapter weights from checkpoint {ckpt_path}")
    
    # ******** TODO: 下面代码基本上是推理使用，所以要针对fastcomposer推理管线进行修改 ******* #
    
    #     self.device = device
    #     self.ip_image_encoder_path = ip_image_encoder_path
    #     self.ip_ckpt = ip_ckpt
    #     self.num_tokens = num_tokens
        
    #     self.pipe = sd_pipe.to(self.device)
    #     self.set_ip_adapter()
        
    #     # load image encoder
    #     self.ip_image_encoder = CLIPVisionModelWithProjection.from_pretrained(self.ip_image_encoder_path).to(
    #         self.device, dtype=torch.float16
    #     )
    #     self.ip_clip_image_processor = CLIPImageProcessor()
    #     # image proj model
    #     self.ip_image_proj_model = self.init_proj()

    #     self.load_ip_adapter()
    
    # def init_proj(self):
    #     image_proj_model = IpImageProjModel(
    #         cross_attention_dim=self.pipe.unet.config.cross_attention_dim,
    #         clip_embeddings_dim=self.image_encoder.config.projection_dim,
    #         clip_extra_context_tokens=self.num_tokens,
    #     ).to(self.device, dtype=torch.float16)
    #     return image_proj_model
        
    # def set_ip_adapter(self):
    #     unet = self.pipe.unet
    #     attn_procs = {}
    #     for name in unet.attn_processors.keys():
    #         cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
    #         if name.startswith("mid_block"):
    #             hidden_size = unet.config.block_out_channels[-1]
    #         elif name.startswith("up_blocks"):
    #             block_id = int(name[len("up_blocks.")])
    #             hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
    #         elif name.startswith("down_blocks"):
    #             block_id = int(name[len("down_blocks.")])
    #             hidden_size = unet.config.block_out_channels[block_id]
    #         if cross_attention_dim is None:
    #             attn_procs[name] = AttnProcessor()
    #         else:
    #             attn_procs[name] = IPAttnProcessor(
    #                 hidden_size=hidden_size,
    #                 cross_attention_dim=cross_attention_dim,
    #                 scale=1.0,
    #                 num_tokens=self.num_tokens,
    #             ).to(self.device, dtype=torch.float16)
                
    #     unet.set_attn_processor(attn_procs)
    #     if hasattr(self.pipe, "controlnet"):
    #         if isinstance(self.pipe.controlnet, MultiControlNetModel):
    #             for controlnet in self.pipe.controlnet.nets:
    #                 controlnet.set_attn_processor(CNAttnProcessor(num_tokens=self.num_tokens))
    #         else:
    #             self.pipe.controlnet.set_attn_processor(CNAttnProcessor(num_tokens=self.num_tokens))
                
    # def load_ip_adapter(self):
    #     if os.path.splitext(self.ip_ckpt)[-1] == ".safetensors":
    #         state_dict = {"image_proj": {}, "ip_adapter": {}}
    #         with safe_open(self.ip_ckpt, framework="pt", device="cpu") as f:
    #             for key in f.keys():
    #                 if key.startswith("image_proj."):
    #                     state_dict["image_proj"][key.replace("image_proj.", "")] = f.get_tensor(key)
    #                 elif key.startswith("ip_adapter."):
    #                     state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = f.get_tensor(key)
    #     else:
    #         state_dict = torch.load(self.ip_ckpt, map_location="cpu")
    #     self.ip_image_proj_model.load_state_dict(state_dict["image_proj"])
    #     ip_layers = torch.nn.ModuleList(self.pipe.unet.attn_processors.values())
    #     ip_layers.load_state_dict(state_dict["ip_adapter"])
    
    # @torch.inference_mode()
    # def get_image_embeds(self, pil_image=None, clip_image_embeds=None):
    #     if pil_image is not None:
    #         if isinstance(pil_image, Image.Image):
    #             pil_image = [pil_image]
    #         clip_image = self.ip_clip_image_processor(images=pil_image, return_tensors="pt").pixel_values
    #         clip_image_embeds = self.image_encoder(clip_image.to(self.device, dtype=torch.float16)).image_embeds
    #     else:
    #         clip_image_embeds = clip_image_embeds.to(self.device, dtype=torch.float16)
    #     image_prompt_embeds = self.ip_image_proj_model(clip_image_embeds)
    #     uncond_image_prompt_embeds = self.ip_image_proj_model(torch.zeros_like(clip_image_embeds))
    #     return image_prompt_embeds, uncond_image_prompt_embeds

    # def set_scale(self, scale):
    #     for attn_processor in self.pipe.unet.attn_processors.values():
    #         if isinstance(attn_processor, IPAttnProcessor):
    #             attn_processor.scale = scale
    
    # ************************ ip adapter code above ************************ #

    def _clear_cross_attention_scores(self):
        if hasattr(self, "cross_attention_scores"):
            keys = list(self.cross_attention_scores.keys())
            for k in keys:
                del self.cross_attention_scores[k]

        gc.collect()

    
    @staticmethod
    def from_pretrained(args):
        text_encoder = IpComposerTextEncoder.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="text_encoder",
            revision=args.revision,
        )
        vae = AutoencoderKL.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision
        )
        unet = UNet2DConditionModel.from_pretrained(
            args.pretrained_model_name_or_path,
            subfolder="unet",
            revision=args.non_ema_revision,
        )
        image_encoder = IpComposerCLIPImageEncoder.from_pretrained(
            args.image_encoder_name_or_path,
        )
        
        ip_image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.image_encoder_path)
        
        #ip-adapter
        image_proj_model = ImageProjModel(
            cross_attention_dim=unet.config.cross_attention_dim,
            clip_embeddings_dim=ip_image_encoder.config.projection_dim,
            clip_extra_context_tokens=4,
        )
        # init adapter modules
        attn_procs = {}
        unet_sd = unet.state_dict()
        # 遍历 UNet 中所有的注意力处理模块名称，处理每个模块并决定是否替换为 IPAttnProcessor。
        for name in unet.attn_processors.keys():
            # 模块名称以 attn1.processor 结尾是自注意力层（self-attention），将 cross_attention_dim 设置为 None。
            # 否则，将其设置为 unet.config.cross_attention_dim，表示交叉注意力层（cross-attention）使用的维度。
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            
            # 根据模块名称决定 hidden_size，这表示该模块所在层的隐藏特征维度
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]
            
            # 如果 cross_attention_dim 是 None（即自注意力层），创建一个标准的 AttnProcessor。
            if cross_attention_dim is None:
                attn_procs[name] = AttnProcessor()
            # 如果是交叉注意力层，创建一个 IPAttnProcessor，并从 unet_sd 中提取权重，用于初始化新的处理器。
            else:
                layer_name = name.split(".processor")[0]
                weights = {
                    "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                    "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
                }
                attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
                attn_procs[name].load_state_dict(weights)
                
        unet.set_attn_processor(attn_procs)
        adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
        
        return IpComposerModel(text_encoder, image_encoder, vae, unet, image_proj_model, adapter_modules, ip_image_encoder, args)
        

    def to_pipeline(self):
        pipe = StableDiffusionPipeline.from_pretrained(
            self.pretrained_model_name_or_path,
            revision=self.revision,
            non_ema_revision=self.non_ema_revision,
            text_encoder=self.text_encoder,
            vae=self.vae,
            unet=self.unet,
        )
        pipe.safety_checker = None

        pipe.image_encoder = self.image_encoder

        pipe.postfuse_module = self.postfuse_module
        pipe.image_proj_model = self.image_proj_model
        pipe.adapter_modules = self.adapter_modules
        pipe.ip_image_encoder = self.ip_image_encoder

        return pipe

    def forward(self, batch, noise_scheduler):
        pixel_values = batch["pixel_values"]
        input_ids = batch["input_ids"]
        image_token_mask = batch["image_token_mask"]
        object_pixel_values = batch["object_pixel_values"]
        num_objects = batch["num_objects"]

        vae_dtype = self.vae.parameters().__next__().dtype
        vae_input = pixel_values.to(vae_dtype)

        latents = self.vae.encode(vae_input).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device
        )
        timesteps = timesteps.long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # (bsz, max_num_objects, num_image_tokens, dim)
        object_embeds = self.image_encoder(object_pixel_values)

        encoder_hidden_states = self.text_encoder(
            input_ids, image_token_mask, object_embeds, num_objects
        )[
            0
        ]  # (bsz, seq_len, dim)

        encoder_hidden_states = self.postfuse_module(
            encoder_hidden_states,
            object_embeds,
            image_token_mask,
            num_objects,
        )

        # Get the target for loss depending on the prediction type
        if noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(
                f"Unknown prediction type {noise_scheduler.config.prediction_type}"
            )
        
        # TODO: ip_adapter operation 加入ip-adapter的前向计算 global_image_feature
        with torch.no_grad():
            image_embeds = self.ip_image_encoder(batch["clip_images"].to(latents.device, dtype=vae_dtype)).image_embeds
        image_embeds_ = []
        for image_embed, drop_image_embed in zip(image_embeds, batch["drop_image_embeds"]):
            if drop_image_embed == 1:
                image_embeds_.append(torch.zeros_like(image_embed))
            else:
                image_embeds_.append(image_embed)
        image_embeds = torch.stack(image_embeds_)
        ip_tokens = self.image_proj_model(image_embeds)
        encoder_hidden_states = torch.cat([encoder_hidden_states, ip_tokens], dim=1)
        # Predict the noise residual
        pred = self.unet(noisy_latents, timesteps, encoder_hidden_states).sample
        
        if self.mask_loss and torch.rand(1) < self.mask_loss_prob:
            object_segmaps = batch["object_segmaps"]
            mask = (object_segmaps.sum(dim=1) > 0).float()
            mask = F.interpolate(
                mask.unsqueeze(1),
                size=(pred.shape[-2], pred.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )
            pred = pred * mask
            target = target * mask

        denoise_loss = F.mse_loss(pred.float(), target.float(), reduction="mean")

        return_dict = {"denoise_loss": denoise_loss}

        if self.object_localization:
            object_segmaps = batch["object_segmaps"]
            image_token_idx = batch["image_token_idx"]
            image_token_idx_mask = batch["image_token_idx_mask"]
            localization_loss = get_object_localization_loss(
                self.cross_attention_scores,
                object_segmaps,
                image_token_idx,
                image_token_idx_mask,
                self.object_localization_loss_fn,
            )
            return_dict["localization_loss"] = localization_loss
            loss = self.object_localization_weight * localization_loss + denoise_loss
            self._clear_cross_attention_scores()
        else:
            loss = denoise_loss

        return_dict["loss"] = loss
        return return_dict
