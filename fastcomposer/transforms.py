from PIL import Image
import numpy as np
from torchvision import transforms as T
import torch
from collections import OrderedDict


class SegmentProcessor(torch.nn.Module):
    def forward(self, image, background, segmap, id, bbox):
        """
        image: 输入图像的张量,通常是形状为 (C, H, W) 的三维张量,其中 C 是通道数（例如 RGB 图像为 3）,H 和 W 分别是图像的高度和宽度。
        background: 与 image 形状相同的背景图像张量,用于替换指定对象以外的部分。
        segmap: 分割图像（或分割掩码）,其中每个像素的值表示不同对象的 ID。
        id: 目标对象的 ID,表示从 segmap 中提取的特定对象。
        bbox: 边界框,用于裁剪出指定对象的区域。格式为 [h1, w1, h2, w2],其中 (h1, w1) 是左上角坐标,(h2, w2) 是右下角坐标。
        """
        
        # 1. 创建一个布尔掩码 mask，其中 mask 为 True 的位置表示 segmap 中不属于目标对象 id 的像素位置。
        # 1.1 mask为True表示需要替换为背景区域的位置
        # 2. 使用布尔掩码将 image 中不属于目标对象的区域替换为 background 中对应的像素值。
        # 3. 根据传入的 bbox（边界框）对 image 进行裁剪，提取目标对象的区域
        mask = segmap != id
        image[:, mask] = background[:, mask]
        h1, w1, h2, w2 = bbox
        return image[:, w1:w2, h1:h2]

    def get_background(self, image):
        raise NotImplementedError


class RandomSegmentProcessor(SegmentProcessor):
    def get_background(self, image):
        background = torch.randint(
            0, 255, image.shape, dtype=image.dtype, device=image.device
        )
        return background


class PadToSquare(torch.nn.Module):
    def __init__(self, fill=0, padding_mode="constant"):
        super().__init__()
        self.fill = fill
        self.padding_mode = padding_mode

    def forward(self, image: torch.Tensor):
        _, h, w = image.shape
        if h == w:
            return image
        elif h > w:
            padding = (h - w) // 2
            image = torch.nn.functional.pad(
                image,
                (padding, padding, 0, 0),
                self.padding_mode,
                self.fill,
            )
        else:
            padding = (w - h) // 2
            image = torch.nn.functional.pad(
                image,
                (0, 0, padding, padding),
                self.padding_mode,
                self.fill,
            )
        return image


class CropTopSquare(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, image: torch.Tensor):
        _, h, w = image.shape
        if h <= w:
            return image
        return image[:, :w, :]


class AlwaysCropTopSquare(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, image: torch.Tensor):
        _, h, w = image.shape
        if h > w:
            return image[:, :w, :]
        else:  # h <= w
            return image[:, :, w // 2 - h // 2 : w // 2 + h // 2]


class RandomZoomIn(torch.nn.Module):
    def __init__(self, min_zoom=1.0, max_zoom=1.5):
        super().__init__()
        self.min_zoom = min_zoom
        self.max_zoom = max_zoom

    def forward(self, image: torch.Tensor):
        zoom = torch.rand(1) * (self.max_zoom - self.min_zoom) + self.min_zoom
        original_shape = image.shape
        new_height = max(32, int(zoom * image.shape[1]))
        new_width = max(32, int(zoom * image.shape[2]))
        image = T.functional.resize(
            image,
            # (int(zoom * image.shape[1]), int(zoom * image.shape[2])),
            (new_height, new_width),
            interpolation=T.InterpolationMode.BILINEAR,
            antialias=True,
        )
        # crop top square
        image = CropTopSquare()(image)
        return image


class CenterCropOrPadSides(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, image: torch.Tensor):
        _, h, w = image.shape
        if h > w:
            # pad sides with black
            padding = (h - w) // 2
            image = torch.nn.functional.pad(
                image,
                (padding, padding, 0, 0),
                "constant",
                0,
            )
            # resize to square
            image = T.functional.resize(
                image,
                (w, w),
                interpolation=T.InterpolationMode.BILINEAR,
                antialias=True,
            )
        else:
            # center crop to square
            padding = (w - h) // 2
            image = image[:, :, padding : padding + h]
        return image


class TrainTransformWithSegmap(torch.nn.Module):
    # TrainTransformWithSegmap 类是一个数据增强和变换模块,用于对图像和分割图（segmentation map）同时进行处理。
    def __init__(self, args):
        super().__init__()
        # 使用双线性插值（BILINEAR）将输入图像调整到指定的训练分辨率
        self.image_resize = T.Resize(
            args.train_resolution,
            interpolation=T.InterpolationMode.BILINEAR,
            antialias=True,
        )
        # 使用最近邻插值（NEAREST）将输入的分割图调整到训练分辨率。
        # 由于分割图通常包含类别索引（而非连续像素值）,最近邻插值可以避免插值过程产生的混合像素。
        self.segmap_resize = T.Resize(
            args.train_resolution,
            interpolation=T.InterpolationMode.NEAREST,
        )
        # 随机水平翻转图像和分割图,增加数据的多样性。
        self.flip = T.RandomHorizontalFlip()
        # 中心裁剪
        self.crop = CenterCropOrPadSides()

    def forward(self, image, segmap):
        # 调整图像大小;为分割图增加一个维度,以匹配 self.segmap_resize 的输入格式;分割图通过 self.segmap_resize 调整大小。
        image = self.image_resize(image)
        segmap = segmap.unsqueeze(0)
        segmap = self.segmap_resize(segmap)
        # 拼接图像和分割图: 
        image_and_segmap = torch.cat([image, segmap], dim=0)
        # 水平翻转和裁剪: 
        image_and_segmap = self.flip(image_and_segmap)
        image_and_segmap = self.crop(image_and_segmap)
        # 将翻转和裁剪后的拼接张量按通道分为图像和分割图部分
        image = image_and_segmap[:3]
        segmap = image_and_segmap[3:]
        image = (image.float() / 127.5) - 1
        segmap = segmap.squeeze(0)
        return image, segmap


class TestTransformWithSegmap(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.image_resize = T.Resize(
            args.test_resolution,
            interpolation=T.InterpolationMode.BILINEAR,
            antialias=True,
        )
        self.segmap_resize = T.Resize(
            args.test_resolution,
            interpolation=T.InterpolationMode.NEAREST,
        )
        self.crop = CenterCropOrPadSides()

    def forward(self, image, segmap):
        image = self.image_resize(image)
        segmap = segmap.unsqueeze(0)
        segmap = self.segmap_resize(segmap)
        image_and_segmap = torch.cat([image, segmap], dim=0)
        image_and_segmap = self.crop(image_and_segmap)
        image = image_and_segmap[:3]
        segmap = image_and_segmap[3:]
        image = (image.float() / 127.5) - 1
        segmap = segmap.squeeze(0)
        return image, segmap


def get_train_transforms(args):
    train_transforms = torch.nn.Sequential(
        T.Resize(
            args.train_resolution,
            interpolation=T.InterpolationMode.BILINEAR,
            antialias=True,
        ),
        T.RandomHorizontalFlip(),
        CenterCropOrPadSides(),
        T.ConvertImageDtype(torch.float32),
        T.Normalize([0.5], [0.5]),
    )
    return train_transforms


def get_train_transforms_with_segmap(args):
    train_transforms = TrainTransformWithSegmap(args)
    return train_transforms


def get_test_transforms(args):
    test_transforms = torch.nn.Sequential(
        T.Resize(
            args.test_resolution,
            interpolation=T.InterpolationMode.BILINEAR,
            antialias=True,
        ),
        CenterCropOrPadSides(),
        T.ConvertImageDtype(torch.float32),
        T.Normalize([0.5], [0.5]),
    )
    return test_transforms


def get_test_transforms_with_segmap(args):
    test_transforms = TestTransformWithSegmap(args)
    return test_transforms


def get_object_transforms(args):
    if args.no_object_augmentation:
        pre_augmentations = []
        augmentations = []
    else:
        # 1.预增强步骤: 随机缩放图像,缩放比例在 min_zoom=1.0 到 max_zoom=2.0 之间,概率为 p=0.5
        pre_augmentations = [
            (
                "zoomin",
                T.RandomApply([RandomZoomIn(min_zoom=1.0, max_zoom=2.0)], p=0.5),
            ),
        ]
        # 2.增强步骤: 
        #   rotate: 随机旋转图像,角度在 ±30° 范围内,概率为 p=0.75。
        #   jitter: 颜色抖动（亮度、对比度、饱和度、色调变化）,范围为 0.5,概率为 p=0.5。
        #   blur: 高斯模糊,内核大小为 5,标准差范围为 (0.1, 2.0),概率为 p=0.5。
        #   gray: 随机将图像转换为灰度图,概率为 p=0.1。
        #   flip: 随机水平翻转图像。
        #   elastic: 随机弹性变形,概率为 p=0.5。
        augmentations = [
            (
                "rotate",
                T.RandomApply(
                    [
                        T.RandomAffine(
                            # 默认使用常量填充（填充零值）
                            degrees=30, interpolation=T.InterpolationMode.BILINEAR
                        )
                    ],
                    p=0.75,
                ),
            ),
            ("jitter", T.RandomApply([T.ColorJitter(0.5, 0.5, 0.5, 0.5)], p=0.5)),
            ("blur", T.RandomApply([T.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.5)),
            ("gray", T.RandomGrayscale(p=0.1)),
            ("flip", T.RandomHorizontalFlip()),
            ("elastic", T.RandomApply([T.ElasticTransform()], p=0.5)),
        ]
    # 其他变换: 
    # pad_to_square : 将图像填充为正方形。
    # resize : 将图像调整为指定的分辨率（args.object_resolution）。
    # convert_to_float : 将图像数据类型转换为 torch.float32
    object_transforms = torch.nn.Sequential(
        OrderedDict(
            [
                *pre_augmentations,
                ("pad_to_square", PadToSquare(fill=0, padding_mode="constant")),
                (
                    "resize",
                    T.Resize(
                        (args.object_resolution, args.object_resolution),
                        interpolation=T.InterpolationMode.BILINEAR,
                        antialias=True,
                    ),
                ),
                *augmentations,
                ("convert_to_float", T.ConvertImageDtype(torch.float32)),
            ]
        )
    )
    return object_transforms


def get_test_object_transforms(args):
    object_transforms = torch.nn.Sequential(
        OrderedDict(
            [
                ("pad_to_square", PadToSquare(fill=0, padding_mode="constant")),
                (
                    "resize",
                    T.Resize(
                        (args.object_resolution, args.object_resolution),
                        interpolation=T.InterpolationMode.BILINEAR,
                        antialias=True,
                    ),
                ),
                ("convert_to_float", T.ConvertImageDtype(torch.float32)),
            ]
        )
    )
    return object_transforms


def get_object_processor(args):
    if args.object_background_processor == "random":
        # 生成随机背景
        object_processor = RandomSegmentProcessor()
    else:
        raise ValueError(f"Unknown object processor: {args.object_processor}")
    return object_processor
