import argparse

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import Any

from utils.metrics import Evaluator
import pandas
import os

from PIL import Image
from kornia.color import rgb_to_ycbcr, rgb_to_grayscale
from torchvision.transforms import functional as TVF


def imread(path, mode='RGB'):
    img = TVF.to_tensor(Image.open(path))

    c = img.shape[-3]

    if mode == 'RGB':
        if c == 1:
            img = img.expand(3, -1, -1)
    elif mode == 'GRAY':
        if c == 3:
            img = rgb_to_grayscale(img)
    elif mode == 'YCbCr':
        img = rgb_to_ycbcr(img)

    return img * 255


def evaluate(vi, ir, fused) -> dict[str, Any]:
    metrics = dict()
    metrics['EN'] = Evaluator.EN(fused).cpu().item()
    metrics['SD'] = Evaluator.SD(fused).cpu().item()
    metrics['SF'] = Evaluator.SF(fused).cpu().item()
    metrics['MI'] = Evaluator.MI(fused, ir, vi).cpu().item()
    metrics['SCD'] = Evaluator.SCD(fused, ir, vi).cpu().item()
    metrics['VIFF'] = Evaluator.VIFF(fused, ir, vi).cpu().item()
    metrics['Qabf'] = Evaluator.Qabf(fused, ir, vi).cpu().item()
    metrics['SSIM'] = Evaluator.SSIM(fused, ir, vi).mean().cpu().item()

    return metrics


def evaluate_image_files(vi_dir, ir_dir, fused_dir, progress=False, gpu=True) -> pandas.DataFrame:
    files = os.listdir(fused_dir)

    if progress:
        files = tqdm(files)

    metrics = dict()

    for file in files:
        img_vi = imread(os.path.join(vi_dir, file), mode='GRAY').squeeze(0).cuda()
        img_ir = imread(os.path.join(ir_dir, file), mode='GRAY').squeeze(0).cuda()
        img_fused = imread(os.path.join(fused_dir, file), mode='GRAY').squeeze(0).cuda()

        metrics[file] = evaluate(img_vi, img_ir, img_fused)

    df = pandas.DataFrame(data=metrics).T

    return df


def evaluate_images(imgs_vi, imgs_ir, imgs_fused):
    metrics = list()

    for i in range(len(imgs_fused)):
        img_vi = imgs_vi[i].squeeze(0).cuda()
        img_ir = imgs_ir[i].squeeze(0).cuda()
        img_fused = imgs_fused[i].squeeze(0).cuda()

        metrics.append(evaluate(img_vi, img_ir, img_fused))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vi', type=str, required=True)
    parser.add_argument('--ir', type=str, required=True)
    parser.add_argument('--fused', type=str, required=True)
    parser.add_argument('--output', type=str, required=False, default='print-mean')
    parser.add_argument('--output-dir', type=str, required=False)

    opts = parser.parse_args()

    df = evaluate_image_files(opts.vi, opts.ir, opts.fused, progress=True)

    if opts.output == 'print-mean':

        mean = df.mean(axis='rows').to_frame()

        print(mean.T)

    elif opts.output == 'tensorboard':
        output_dir = opts.output_dir

        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        writer = SummaryWriter(output_dir)

        for colname in df.columns:
            col = df[colname].sort_values()
            for idx, value in enumerate(col):
                writer.add_scalar(f'metrics/{colname}', value, idx)
