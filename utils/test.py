import argparse
import random

from model.model import *
from utils import config
import os.path
import pathlib
import shutil
from typing import Iterable

import torch
import torch.utils.data
from torch.nn import functional as F
import torchvision.transforms.functional as TF
import torchvision.utils
import pandas as pd
from PIL.Image import Image
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model.model import *
from utils import config
from utils.dataset import VIFImageDataset, VIFTestImageDataset
from utils.image import YCbCr2RGB
from utils.logs import PlainLogger

from utils.evaluate.image import evaluate as evaluate_image

def ensure_created(*paths):
    path = os.path.join(*paths)
    os.makedirs(path, exist_ok=True)
    return path


if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='evaluate')

    parser.add_argument('--model', type=str, required=True, help="Model checkpoint path to evaluate")
    parser.add_argument('--dataset-path', type=str, required=False, help='Dataset path (visible under vi/ and infrared under ir/)')
    parser.add_argument('--gray', action='store_true', help='Output images in grayscale')
    parser.add_argument('--no-image', action='store_true', help='No output images')
    parser.add_argument('--output', type=str, required=False, default='./data/results', help='Dir to store fused results')

    args = parser.parse_args()

    dataset_path = args.dataset_path

    # Set model
    dataset = VIFTestImageDataset(dataset_path)
    model = NewModelBase()
    model.load_state_dict(torch.load(args.model))
    model = model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    result_images_dir = args.output
    ensure_created(result_images_dir)

    # Do evaluation
    dataset = tqdm(dataset)
    
    def preprocess_image(visible_image, infrared_image, frac=16):
        from torchvision.transforms import functional as F
        import math
        orig_h, orig_w = visible_image.shape[-2:]
        h = math.ceil((orig_h + 64) / frac) * frac
        w = math.ceil((orig_w + 64) / frac) * frac
        pad_top = (h - orig_h) // 2
        pad_bottom = h - orig_h - pad_top
        pad_left = (w - orig_w) // 2
        pad_right = w - orig_w - pad_left
        visible_image = F.pad(visible_image, [pad_left, pad_top, pad_right, pad_bottom], padding_mode='reflect')
        infrared_image = F.pad(infrared_image, [pad_left, pad_top, pad_right, pad_bottom], padding_mode='reflect')
        return visible_image, infrared_image, (pad_left, pad_top, orig_h, orig_w)

    def postprocess_image(output, size):
        from torchvision.transforms import functional as TVF
        output = TVF.crop(output, size[1], size[0], size[2], size[3])
        return output               

    metrics = dict()
    for image_vi, ir, filename in dataset:
        image_vi, ir, orig_size = preprocess_image(image_vi, ir)
        image_vi = model.convert(image_vi.unsqueeze(0))
        vi, image_vi_Cb, image_vi_Cr = RGB2YCbCr(image_vi)
        ir = model.convert(ir.unsqueeze(0))

        with torch.no_grad():
            fused = model(vi, ir)
            if not args.gray:
                image = YCbCr2RGB(fused, image_vi_Cb, image_vi_Cr)
            else:
                image = fused
            image = postprocess_image(image, orig_size)
            image = TF.to_pil_image(image.squeeze(0))
            image.save(os.path.join(result_images_dir, filename))

    print(f'Evaluated fusion on model "{args.model}" with dataset "{dataset_path}"')
