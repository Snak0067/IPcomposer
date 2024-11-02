from PIL import Image
import numpy as np
from torchvision import transforms as T
import torch
from collections import OrderedDict

def get_ip_transform_processor(args):
    transform = T.Compose([
        T.Resize(args.train_resolution, interpolation=T.InterpolationMode.BILINEAR),
        T.CenterCrop(args.train_resolution),
        T.ToTensor(),
        T.Normalize([0.5], [0.5]),
    ])
    return transform