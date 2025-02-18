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

from .image import evaluate as evaluate_image


def evaluate_fusion(args, config: Config, model, model_name, dataset: Iterable):
    result_dir = model.session.get_work_dir_created('eval')
    result_metrics_dir = os.path.join(result_dir, 'metrics')
    result_tensorboard_dir = os.path.join(result_metrics_dir, model_name)
    result_images_dir = os.path.join(result_dir, 'images', config.common.name, model_name) if args.output is None else args.output

    if os.path.exists(result_tensorboard_dir):
        shutil.rmtree(result_tensorboard_dir)

    os.makedirs(result_metrics_dir, exist_ok=True)
    os.makedirs(result_images_dir, exist_ok=True)
    dataset = tqdm(dataset)
    if args.iii:
        model = model.model
    model.eval()
    metrics = dict()
    for image_vi, ir, filename in dataset:
        image_vi = model.convert(image_vi.unsqueeze(0))
        vi, image_vi_Cb, image_vi_Cr = RGB2YCbCr(image_vi)
        ir = model.convert(ir.unsqueeze(0))
        fused = model(vi, ir)
        if not args.no_image:
            if args.rgb:
                image = YCbCr2RGB(fused, image_vi_Cb, image_vi_Cr)
            else:
                image = fused
            image = TF.to_pil_image(image.squeeze(0))
            image.save(os.path.join(result_images_dir, filename))
        if not args.no_metric:
            metrics[filename] = evaluate_image(vi.squeeze(0).squeeze(0) * 255,
                                               ir.squeeze(0).squeeze(0) * 255,
                                               fused.squeeze(0).squeeze(0) * 255)

    if not args.no_metric:
        df = pd.DataFrame(data=metrics).T

        df.to_csv(f'{result_metrics_dir}/{model_name}.csv', )

        print(df.mean(axis='rows').to_frame().T)

        logger = PlainLogger('log', result_dir, avoid_conflict=False)
        logger.info(f'metric-{model_name}', df.mean(axis='rows').to_frame().T)

        writer = SummaryWriter(result_tensorboard_dir)

        for colname in df.columns:
            col = df[colname].sort_values()
            for idx, value in enumerate(col):
                writer.add_scalar(f'metrics/{colname}', value, idx)



def ensure_created(*paths):
    path = os.path.join(*paths)
    os.makedirs(path, exist_ok=True)
    return path


if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog='evaluate')

    parser.add_argument('--config', type=str, required=True, help="Path to config used in training")

    parser.add_argument('--model', type=str, required=True, help="Model checkpoint path to evaluate")
    parser.add_argument('--session-name', type=str, required=True,
                        help="Session name for this evaluation")
    parser.add_argument('--dataset', type=str, required=False, default='test',
                        help="Dataset type to evaluate, can be train or test, default: test")
    parser.add_argument('--torch-model', required=False, action='store_true',
                        help="Whether this checkpoint is from torch.save")
    parser.add_argument('--dataset-path', type=str, required=False, help='Dataset path (visible under vi/ and infrared under ir/)')

    subparsers = parser.add_subparsers(title='actions', required=True, dest='type')

    parser_fusion = subparsers.add_parser('fusion', description='evaluate fusion')
    parser_fusion.add_argument('--rgb', action='store_true', help='Output images in RGB')
    parser_fusion.add_argument('--no-image', action='store_true', help='No output images')
    parser_fusion.add_argument('--no-metric', action='store_true', help='No metric evaluation')
    parser_fusion.add_argument('--iii', action='store_true', help='Evaluate model from third stage')
    parser_fusion.add_argument('--output', type=str, required=False, help='Image output')

    parser_features = subparsers.add_parser('features', description='evaluate features')

    args = parser.parse_args()

    cfg = config.ConfigLoader()
    cfg.load_from_path(args.config)
    cfg = cfg.freeze()

    if args.dataset_path is not None:
        dataset_path = args.dataset_path
    else:
        dataset_path = cfg.dataset[args.dataset]

    dataset = None
    model_class = None

    # Set model
    if args.type == 'fusion':
        dataset = VIFTestImageDataset(dataset_path)
        if args.iii:
            model_class = ModelBaseStageIII
        else:
            model_class = ModelBaseStageII

    session = TrainingSession(cfg, args.session_name)

    if args.torch_model:
        state_dict = torch.load(args.model)
        model = model_class()
        model.load_state_dict(state_dict, strict=False)
    else:
        model = model_class.load_from_checkpoint(args.model)
    model.set_session(session)
    model.init_with_config(cfg)
    model = model.eval()
    if torch.cuda.is_available():
        model.cuda()

    # Do evaluation
    with torch.no_grad():
        if args.type == 'fusion':
            evaluate_fusion(args, cfg, model, os.path.splitext(os.path.basename(args.model))[0], dataset)

        elif args.type == 'semseg':
            pass

    print(f'Evaluated "{args.type}" on model "{args.model}" with dataset "{dataset_path}"')
