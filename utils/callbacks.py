from typing import Any

import torch
from lightning.pytorch.loggers import TensorBoardLogger
from torchmetrics import MeanMetric

import utils.data as data
import lightning as L
import lightning.pytorch as pl
import torch.nn as nn
from model.base import ModelBase
import os

from lightning.pytorch.utilities import grad_norm

import utils.data as data
import lightning.pytorch as L
import torch.nn as nn


class SaveCheckpointCallback(L.Callback):
    def __init__(self, name: str, save_path: str, save_freq: int):
        super().__init__()
        self.name = name
        self.save_freq = save_freq
        self.save_path = save_path

    def on_train_epoch_end(self, trainer, module):
        if trainer.current_epoch % self.save_freq == 0:
            path = self.get_checkpoint_path(self.save_path, self.name, trainer.current_epoch)
            print(f'Saving the current model at epoch {trainer.current_epoch} to {path}')
            trainer.save_checkpoint(path)

    @staticmethod
    def get_checkpoint_path(path, name, epoch):
        return os.path.join(path, f'{name}-{epoch}.pth')



class GradientNormLog(L.Callback):
    def __init__(self):
        super().__init__()

    def on_before_optimizer_step(self, trainer, module, optimizer):
        iterations = module.iterations

        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        norms = grad_norm(module, norm_type=2)

        module.log_dict(norms)

        # if iterations % 50 == 0:
        #     for (k, v) in norms.items():
        #         print(f'{k}: {v}')


class ClipGradientNorm(L.Callback):
    def __init__(self):
        super().__init__()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        nn.utils.clip_grad_norm_(pl_module.parameters(), 0.01, norm_type=2)


class ProfilingLogger(L.callbacks.Callback):
    def __init__(self):
        super().__init__()

    def on_train_epoch_end(self, trainer, pl_module):
        print()
        print(f"Epoch {trainer.current_epoch} end, profiler output:")
        for k in pl_module.profiling.records():
            print(pl_module.profiling.records()[k])


class LossMetricsLogging(L.Callback):
    def __init__(self, logger: TensorBoardLogger):
        super().__init__()

        self.logger = logger

    def setup(self, trainer: "pl.Trainer", pl_module: ModelBase, stage: str) -> None:
        pl_module.metrics = object()

        pl_module.metrics.train_loss = MeanMetric()
        pl_module.metrics.val_loss = MeanMetric()

    def on_train_batch_end(
        self, trainer: "pl.Trainer", pl_module: ModelBase, outputs, batch: Any, batch_idx: int) -> None:

        if not isinstance(outputs, torch.Tensor):
            return

        loss = outputs.detach()
        pl_module.metrics.train_loss.update(loss)

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: ModelBase) -> None:

        loss = pl_module.metrics.train_loss.compute()
        pl_module.metrics.train_loss.reset()

    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: ModelBase,
        outputs,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:

        if not isinstance(outputs, torch.Tensor):
            return

        loss = outputs.detach()
        pl_module.metrics.val_loss.update(loss)

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: ModelBase) -> None:
        loss = pl_module.metrics.val_loss.compute()
        pl_module.metrics.val_loss.reset()
