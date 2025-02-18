import argparse
import os.path

import torch.utils.data
from lightning.pytorch.loggers import TensorBoardLogger

from model.model import *
from utils import config
from utils.callbacks import SaveCheckpointCallback, GradientNormLog, ProfilingLogger
from utils.train.session import TrainingSession
import pathlib

import torch
import torch.utils.data
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.tensorboard import SummaryWriter

import utils.data
from model.model import *
from utils import config
from utils.callbacks import SaveCheckpointCallback, GradientNormLog, ClipGradientNorm, ProfilingLogger
from utils.dataset import VIFH5Dataset, VIFTestImageDataset, VIFValImageDataset, VIFH5SegDataset, VIFValSegImageDataset, \
    MetaDataModule, VIFSegImageDataset
from utils.train.session import TrainingSession

if __name__ == "__main__":
    L.seed_everything(43, workers=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help="Path to config used in training")
    parser.add_argument('--trainee', required=False, action='store_true', help="Trainee mode")
    parser.add_argument('--half', required=False, action='store_true', help="Half precision mode")
    parser.add_argument('--resume', type=str, required=False, help="Resume from checkpoint")
    parser.add_argument('--resume-torch-model', required=False, action='store_true',
                        help="Whether this checkpoint is from torch.save")
    parser.add_argument('--resume-params-only', required=False, action='store_true',
                        help="Resume only model parameters from checkpoint")
    parser.add_argument('--session-name', type=str, required=True, help="Training session for current training")
    parser.add_argument('--save-checkpoint-path', type=str, required=False, default=None,
                        help="Save checkpoint at given path")

    opts = parser.parse_args()

    cfg = config.ConfigLoader()
    cfg.load_from_path(opts.config)
    cfg = cfg.freeze()

    print(cfg)

    session = TrainingSession(cfg, opts.session_name)

    model_class = None


    # Load model by config
    if cfg.train.stage == 'I':
        model_class = ModelBaseStageI

        train_dataset = VIFH5SegDataset(cfg.dataset.train)
        val_dataset = VIFValSegImageDataset(cfg.dataset.test)

    elif cfg.train.stage == 'II':
        model_class = ModelBaseStageII

        train_dataset = VIFH5SegDataset(cfg.dataset.train)
        val_dataset = VIFValSegImageDataset(cfg.dataset.test)

    elif cfg.train.stage == 'III':
        model_class = ModelBaseStageIII
        train_dataset = VIFSegImageDataset(cfg.dataset.train)
        val_dataset = VIFValSegImageDataset(cfg.dataset.test)

    device = torch.device('cpu')
    if torch.cuda.is_available() and torch.cuda.device_count() > cfg.common.gpu:
        device = torch.device(f'cuda:{cfg.common.gpu}')

    precision = 16 if opts.half else 32

    logger = TensorBoardLogger(session.get_work_dir_created('tensorboard', cfg.common.name),
                               cfg.logging.tensorboard['version'])
    trainer = L.Trainer(
        detect_anomaly=cfg.train['detect_anomaly', False],
        logger=logger,
        max_epochs=cfg.train.max_epochs + 1,
        gradient_clip_val=0.01 if not cfg.train.stage == 'III' else None,
        gradient_clip_algorithm='norm',
        precision=precision,
        # strategy=TraineeStrategy(cfg, device),
        callbacks=[SaveCheckpointCallback(cfg.common.name, session.get_work_dir_created('checkpoint', cfg.common.name),
                                          cfg.checkpoint.save_model_freq),
                   # ClipGradientNorm(),
                   ProfilingLogger(),
                   GradientNormLog(),
                   ],
        accelerator='auto',
        # gpus=[cfg.common.gpu]
    )

    model = None
    if opts.resume_torch_model:
        model = torch.load(opts.resume)
        model.reset_train_state()
        checkpoint = None

    else:
        checkpoint = opts.resume
        if checkpoint is None or not os.path.isfile(checkpoint):
            checkpoint = None

        if checkpoint is not None and opts.resume_params_only:
            model = model_class.load_from_checkpoint(checkpoint)
            model.reset_train_state()
            checkpoint = None

        if model is None:
            model = model_class()

    model.set_session(session)
    model.init_with_config(cfg)

    if opts.half:
        model.half()

    print(f'Starting training {cfg.common.name} with session {session.session_name}')

    if cfg.train.stage == 'III':
        trainer.fit(model,
                    datamodule=MetaDataModule(cfg,
                                              train_dataset,
                                              val_dataset),
                    ckpt_path=checkpoint)
    else:
        trainer.fit(model,
                    torch.utils.data.DataLoader(train_dataset,
                                                batch_size=cfg.train.batch_size,
                                                num_workers=cfg.dataset.loader_threads,
                                                persistent_workers=True,
                                                shuffle=True),
                    torch.utils.data.DataLoader(val_dataset,
                                                batch_size=1,
                                                num_workers=cfg.dataset.loader_threads,
                                                persistent_workers=True,
                                                shuffle=False),
                    ckpt_path=checkpoint)

    trainer.save_checkpoint(session.get_work_dir_path('checkpoint', f'{cfg.common.name}-latest.pth'))

    if opts.save_checkpoint_path is not None:
        trainer.save_checkpoint(opts.save_checkpoint_path)
