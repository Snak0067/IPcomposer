import os
import torch
from torchvision.io import read_image, ImageReadMode
import glob
import json
import numpy as np
import random
from copy import deepcopy
from transformers import CLIPImageProcessor



def prepare_image_token_idx(image_token_mask, max_num_objects):
    """
    输入值: 
        image_token_mask: 一个布尔张量, 指示名词短语的结束位置。由 _tokenize_and_mask_noun_phrases_ends 方法生成, True 值表示标记的位置
        max_num_objects: 最大对象数量, 用于确定返回的 image_token_idx 和 image_token_idx_mask 的固定长度。
    返回值:
        image_token_idx: 一个形状为 [1, max_num_objects] 的张量，包含了 image_token_mask 中所有文本结束位置的索引，以及零填充以达到 max_num_objects 长度。
        image_token_idx_mask: 一个形状为 [1, max_num_objects] 的布尔张量，表示 image_token_idx 中哪些位置是有效的标记。
    """
    
    # 使用 torch.nonzero(image_token_mask, as_tuple=True) 来获取 image_token_mask 中所有 True 值的位置（索引）。
    # as_tuple=True 表示返回的结果为一个元组，第二个元素 [1] 即为这些位置的索引。
    # image_token_idx 因此包含了所有标记在文本中的结束位置的索引    
    image_token_idx = torch.nonzero(image_token_mask, as_tuple=True)[1]
    
    # image_token_idx_mask 是一个与 image_token_idx 相同长度的布尔张量，初始化为全 True。
    # 该掩码用于表示哪些索引在 image_token_idx 中是有效的
    image_token_idx_mask = torch.ones_like(image_token_idx, dtype=torch.bool)
    # 对索引和掩码进行填充
    if len(image_token_idx) < max_num_objects:        
        # 如果 image_token_idx 的长度小于 max_num_objects，则将 image_token_idx 用零进行填充，直到其长度达到 max_num_objects。
        # 同时，将 image_token_idx_mask 用 False 填充到 max_num_objects，以便掩码的长度与 image_token_idx 一致
        image_token_idx = torch.cat(
            [
                image_token_idx,
                torch.zeros(max_num_objects - len(image_token_idx), dtype=torch.long),
            ]
        )
        image_token_idx_mask = torch.cat(
            [
                image_token_idx_mask,
                torch.zeros(
                    max_num_objects - len(image_token_idx_mask),
                    dtype=torch.bool,
                ),
            ]
        )
    # 这确保了无论 image_token_idx 的原始长度是多少，最终结果的长度都为 max_num_objects，方便后续计算。
    image_token_idx = image_token_idx.unsqueeze(0)
    image_token_idx_mask = image_token_idx_mask.unsqueeze(0)
    return image_token_idx, image_token_idx_mask


class DemoDataset(object):
    def __init__(
        self,
        test_caption,
        test_reference_folder,
        tokenizer,
        object_transforms,
        image_token="<|image|>",
        max_num_objects=4,
        device=None,
    ) -> None:
        self.test_caption = test_caption
        self.test_reference_folder = test_reference_folder
        self.tokenizer = tokenizer
        self.image_token = image_token
        self.object_transforms = object_transforms

        tokenizer.add_tokens([image_token], special_tokens=True)
        self.image_token_id = tokenizer.convert_tokens_to_ids(image_token)
        self.max_num_objects = max_num_objects
        self.device = device
        self.image_ids = None

    def set_caption(self, caption):
        self.test_caption = caption

    def set_reference_folder(self, reference_folder):
        self.test_reference_folder = reference_folder

    def set_image_ids(self, image_ids=None):
        self.image_ids = image_ids

    def get_data(self):
        return self.prepare_data()

    def _tokenize_and_mask_noun_phrases_ends(self, caption):
        input_ids = self.tokenizer.encode(caption)

        noun_phrase_end_mask = [False for _ in input_ids]
        clean_input_ids = []
        clean_index = 0

        for i, id in enumerate(input_ids):
            if id == self.image_token_id:
                noun_phrase_end_mask[clean_index - 1] = True
            else:
                clean_input_ids.append(id)
                clean_index += 1

        max_len = self.tokenizer.model_max_length

        if len(clean_input_ids) > max_len:
            clean_input_ids = clean_input_ids[:max_len]
        else:
            clean_input_ids = clean_input_ids + [self.tokenizer.pad_token_id] * (
                max_len - len(clean_input_ids)
            )

        if len(noun_phrase_end_mask) > max_len:
            noun_phrase_end_mask = noun_phrase_end_mask[:max_len]
        else:
            noun_phrase_end_mask = noun_phrase_end_mask + [False] * (
                max_len - len(noun_phrase_end_mask)
            )

        clean_input_ids = torch.tensor(clean_input_ids, dtype=torch.long)
        noun_phrase_end_mask = torch.tensor(noun_phrase_end_mask, dtype=torch.bool)
        return clean_input_ids.unsqueeze(0), noun_phrase_end_mask.unsqueeze(0)

    def prepare_data(self):
        object_pixel_values = []
        image_ids = []

        for image_id in self.image_ids:
            reference_image_path = sorted(
                glob.glob(os.path.join(self.test_reference_folder, image_id, "*.jpg"))
                + glob.glob(os.path.join(self.test_reference_folder, image_id, "*.png"))
                + glob.glob(
                    os.path.join(self.test_reference_folder, image_id, "*.jpeg")
                )
            )[0]

            reference_image = self.object_transforms(
                read_image(reference_image_path, mode=ImageReadMode.RGB)
            ).to(self.device)
            object_pixel_values.append(reference_image)
            image_ids.append(image_id)

        input_ids, image_token_mask = self._tokenize_and_mask_noun_phrases_ends(
            self.test_caption
        )

        image_token_idx, image_token_idx_mask = prepare_image_token_idx(
            image_token_mask, self.max_num_objects
        )

        num_objects = image_token_idx_mask.sum().item()

        object_pixel_values = torch.stack(
            object_pixel_values
        )  # [max_num_objects, 3, 256, 256]
        object_pixel_values = object_pixel_values.to(
            memory_format=torch.contiguous_format
        ).float()

        return {
            "input_ids": input_ids,
            "image_token_mask": image_token_mask,
            "image_token_idx": image_token_idx,
            "image_token_idx_mask": image_token_idx_mask,
            "object_pixel_values": object_pixel_values,
            "num_objects": torch.tensor(num_objects),
            "filenames": image_ids,
        }


class IpComposerDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        tokenizer,
        train_transforms,
        object_transforms,
        object_processor,
        device=None,
        max_num_objects=4,
        num_image_tokens=1,
        image_token="<|image|>",
        object_appear_prob=1,
        uncondition_prob=0,
        text_only_prob=0,
        object_types=None,
        split="all",
        min_num_objects=None,
        balance_num_objects=False
    ):
        self.root = root
        self.tokenizer = tokenizer
        self.train_transforms = train_transforms
        self.object_transforms = object_transforms
        self.object_processor = object_processor
        self.max_num_objects = max_num_objects
        self.image_token = image_token
        self.num_image_tokens = num_image_tokens
        self.object_appear_prob = object_appear_prob
        self.device = device
        self.uncondition_prob = uncondition_prob
        self.text_only_prob = text_only_prob
        self.object_types = object_types
        # ip_adapter
        self.clip_image_processor = CLIPImageProcessor()

        if split == "all":
            image_ids_path = os.path.join(root, "image_ids.txt")
        elif split == "train":
            image_ids_path = os.path.join(root, "image_ids_train.txt")
        elif split == "test":
            image_ids_path = os.path.join(root, "image_ids_test.txt")
        else:
            raise ValueError(f"Unknown split {split}")

        with open(image_ids_path, "r") as f:
            self.image_ids = f.read().splitlines()

        tokenizer.add_tokens([image_token], special_tokens=True)
        self.image_token_id = tokenizer.convert_tokens_to_ids(image_token)

        if min_num_objects is not None:
            print(f"Filtering images with less than {min_num_objects} objects")
            filtered_image_ids = []
            for image_id in tqdm(self.image_ids):
                chunk = image_id[:5]
                info_path = os.path.join(self.root, chunk, image_id + ".json")
                with open(info_path, "r") as f:
                    info_dict = json.load(f)
                segments = info_dict["segments"]
                # TODO: object_types需要处理, 目前object只处理带有person的label
                if self.object_types is not None:
                    segments = [
                        segment
                        for segment in segments
                        if segment["coco_label"] in self.object_types
                    ]

                if len(segments) >= min_num_objects:
                    filtered_image_ids.append(image_id)
            self.image_ids = filtered_image_ids

        if balance_num_objects:
            _balance_num_objects(self)

    def __len__(self):
        return len(self.image_ids)

    def _tokenize_and_mask_noun_phrases_ends(self, caption, segments):
        for segment in reversed(segments):
            end = segment["end"]
            # 将名词短语的结尾处插入一个图像标记（image_token="<|image|>"）
            # 并生成一个掩码 noun_phrase_end_mask, 用于指示每个名词短语的结束位置
            caption = caption[:end] + self.image_token + caption[end:]

        input_ids = self.tokenizer.encode(caption)          # 对 caption 进行分词编码

        noun_phrase_end_mask = [False for _ in input_ids]   # 一个布尔掩码, 用于标记每个名词短语的结束位置。初始化时, 所有值都为 False。
        clean_input_ids = []
        clean_index = 0
        
        for i, id in enumerate(input_ids):
            if id == self.image_token_id:
                # 当前tokenid是image_token,则前一个token是名词结尾, 设为true
                noun_phrase_end_mask[clean_index - 1] = True
            else:
                # clean_input_ids 用于保存删除 image_token_id 后的编码, 确保仅保留真正的文本内容。
                clean_input_ids.append(id)
                clean_index += 1

        max_len = self.tokenizer.model_max_length
        
        # 如果 clean_input_ids 长度超过了 max_len, 则截断到最大长度, 否则用填充标记 pad_token_id 填充至 max_len。
        # noun_phrase_end_mask 也同样处理为最大长度, 使用 False 填充不足的部分。
        
        if len(clean_input_ids) > max_len:
            clean_input_ids = clean_input_ids[:max_len]
        else:
            clean_input_ids = clean_input_ids + [self.tokenizer.pad_token_id] * (
                max_len - len(clean_input_ids)
            )

        if len(noun_phrase_end_mask) > max_len:
            noun_phrase_end_mask = noun_phrase_end_mask[:max_len]
        else:
            noun_phrase_end_mask = noun_phrase_end_mask + [False] * (
                max_len - len(noun_phrase_end_mask)
            )

        clean_input_ids = torch.tensor(clean_input_ids, dtype=torch.long)
        noun_phrase_end_mask = torch.tensor(noun_phrase_end_mask, dtype=torch.bool)
        return clean_input_ids.unsqueeze(0), noun_phrase_end_mask.unsqueeze(0)

    @torch.no_grad()
    def preprocess(self, image, info_dict, segmap, image_id):
        """
            image: 输入的图像。
            info_dict: 包含图像元数据的字典, 包含 "caption" 和 "segments" 信息。
            caption: 图像的描述文本。
            segments: 包含对象分割信息的列表, 每个对象分割信息包含对象的 id、coco_label 和 bbox。
            segmap: 图像对应的分割图, 用于区分图像中的不同对象。
            image_id: 图像的唯一标识符。
        """
        #ip-adapter
        clip_image = self.clip_image_processor(images=image, return_tensors="pt").pixel_values
        
        caption = info_dict["caption"]
        segments = info_dict["segments"]
        
        # 从 info_dict 中提取 caption 和 segments。如果指定了 object_types, 则过滤 segments 以只保留特定类型的对象。
        if self.object_types is not None:
            segments = [
                segment
                for segment in segments
                if segment["coco_label"] in self.object_types
            ]
        # 双线性插值调整图像分辨率；最近邻插值调整segmap分辨率到args.train_resolution
        # 0.5概率水平翻转图像和segmap, 中心裁剪
        pixel_values, transformed_segmap = self.train_transforms(image, segmap)

        object_pixel_values = []
        object_segmaps = []

        # ip-adapter drop
        drop_image_embed=0
        
        prob = random.random()
        # 当随机值小于 uncondition_prob 时, 清空 caption 和 segments, 表示无条件输入。
        if prob < self.uncondition_prob:
            caption = ""
            segments = []
            drop_image_embed = 1
        # TODO: 是否有 bug
        # 如果介于 uncondition_prob 和 uncondition_prob + text_only_prob 之间, 只保留 caption, 表示仅文本输入
        elif prob < self.uncondition_prob + self.text_only_prob:
            segments = []
            drop_image_embed = 1
        else:
        # 根据 object_appear_prob 的概率来保留部分 segments, 用于增加数据的多样性
            segments = [
                segment
                for segment in segments
                if random.random() < self.object_appear_prob # object_appear_prob=0.9
            ]
        # 如果 segments 中的对象数量超过 max_num_objects, 
        # 则随机采样最多 max_num_objects 个对象, 并根据对象的 end 位置进行排序
        if len(segments) > self.max_num_objects: # max_num_objects=4
            # random sample objects
            segments = random.sample(segments, self.max_num_objects)

        segments = sorted(segments, key=lambda x: x["end"])
        # 生成和image相同尺寸的随机背景图 background （0-255）randinit
        background = self.object_processor.get_background(image)

        for segment in segments:
            id = segment["id"]
            bbox = segment["bbox"]  # [h1, w1, h2, w2]
            # 检查并将 bbox 的元素转换为整数
            bbox = [int(coord) if not isinstance(coord, int) else coord for coord in bbox]
            
            # 基于bbox提取图像 image指定 ID 的对象区域, 并用背景 background 替换掉其他部分
            object_image = self.object_processor(
                deepcopy(image), background, segmap, id, bbox
            )
            object_pixel_values.append(self.object_transforms(object_image))
            # 将 transformed_segmap 中对应 id 的区域存储在 object_segmaps 中
            object_segmaps.append(transformed_segmap == id)

        # 将image_token插入caption, tokenizer对caption分词, 对分词结果产生mask的布尔值列表表示名词结束
        input_ids, image_token_mask = self._tokenize_and_mask_noun_phrases_ends(
            caption, segments
        )
        # 调用 prepare_image_token_idx 函数，将 image_token_mask 中的 True 位置转换为对象标记索引 image_token_idx，
        # 并生成一个掩码 image_token_idx_mask，表示每个索引是否有效。
        # self.max_num_objects 指定了最大对象数量，确保 image_token_idx 和 image_token_idx_mask 具有一致的长度。
        image_token_idx, image_token_idx_mask = prepare_image_token_idx(
            image_token_mask, self.max_num_objects
        )
        # num_objects 记录了有效对象的数量，即 image_token_idx_mask 中 True 的数量
        num_objects = image_token_idx_mask.sum().item()
        # 根据有效对象数量截取 object_pixel_values 和 object_segmaps，只保留前 num_objects 个对象的像素值和分割图。
        object_pixel_values = object_pixel_values[:num_objects]
        object_segmaps = object_segmaps[:num_objects]

        # 如果 num_objects 大于 0，则 padding_object_pixel_values 以第一个对象的像素张量形状为参考创建全零填充。
        # 如果没有有效对象（即 num_objects 为 0），padding_object_pixel_values 则使用背景图像的变换结果填充。
        if num_objects > 0:
            padding_object_pixel_values = torch.zeros_like(object_pixel_values[0])
        else:
            padding_object_pixel_values = self.object_transforms(background)
            padding_object_pixel_values[:] = 0

        # 填充对象像素和分割图到最大数量
        if num_objects < self.max_num_objects:
            object_pixel_values += [
                torch.zeros_like(padding_object_pixel_values)
                for _ in range(self.max_num_objects - num_objects)
            ]
            object_segmaps += [
                torch.zeros_like(transformed_segmap)
                for _ in range(self.max_num_objects - num_objects)
            ]
        # 将对象像素值和分割图堆叠成张量
        object_pixel_values = torch.stack(
            object_pixel_values
        )  # [max_num_objects, 3, 256, 256]
        object_pixel_values = object_pixel_values.to(
            memory_format=torch.contiguous_format
        ).float()

        object_segmaps = torch.stack(
            object_segmaps
        ).float()  # [max_num_objects, 256, 256]

        return {
            "clip_image": clip_image,
            "drop_image_embed": drop_image_embed,
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "image_token_mask": image_token_mask,
            "image_token_idx": image_token_idx,
            "image_token_idx_mask": image_token_idx_mask,
            "object_pixel_values": object_pixel_values,
            "object_segmaps": object_segmaps,
            "num_objects": torch.tensor(num_objects),
            "image_ids": torch.tensor(image_id),
        }

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        chunk = image_id[:5]
        image_path = os.path.join(self.root, chunk, image_id + ".jpg")
        info_path = os.path.join(self.root, chunk, image_id + ".json")
        segmap_path = os.path.join(self.root, chunk, image_id + ".npy")

        image = read_image(image_path, mode=ImageReadMode.RGB)

        with open(info_path, "r") as f:
            info_dict = json.load(f)
        segmap = torch.from_numpy(np.load(segmap_path))

        if self.device is not None:
            image = image.to(self.device)
            segmap = segmap.to(self.device)

        return self.preprocess(image, info_dict, segmap, int(image_id))


def collate_fn(examples):
    # ip-adapter
    clip_images = torch.cat([example["clip_image"] for example in examples], dim=0)
    drop_image_embeds = [example["drop_image_embed"] for example in examples]
    
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    input_ids = torch.cat([example["input_ids"] for example in examples])
    image_ids = torch.stack([example["image_ids"] for example in examples])

    image_token_mask = torch.cat([example["image_token_mask"] for example in examples])
    image_token_idx = torch.cat([example["image_token_idx"] for example in examples])
    image_token_idx_mask = torch.cat(
        [example["image_token_idx_mask"] for example in examples]
    )

    object_pixel_values = torch.stack(
        [example["object_pixel_values"] for example in examples]
    )
    object_segmaps = torch.stack([example["object_segmaps"] for example in examples])

    num_objects = torch.stack([example["num_objects"] for example in examples])
    return {
        "clip_images": clip_images,
        "drop_image_embeds": drop_image_embeds,
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "image_token_mask": image_token_mask,
        "image_token_idx": image_token_idx,
        "image_token_idx_mask": image_token_idx_mask,
        "object_pixel_values": object_pixel_values,
        "object_segmaps": object_segmaps,
        "num_objects": num_objects,
        "image_ids": image_ids,
    }


def get_data_loader(dataset, batch_size, num_workers, shuffle=True):
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=shuffle,
        collate_fn=collate_fn,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    return dataloader
