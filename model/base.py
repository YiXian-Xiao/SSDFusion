import contextlib
from collections import namedtuple
from typing import Any

import torch
import lightning as L
from torchmetrics import MeanMetric, Metric

from utils import logs
from utils.config import Config
from utils.logs import ModuleLoggers
from utils.profiler import Profiler
from utils.train.session import TrainingSession

from .common import *


class ModelBase(Module):
    def __init__(self):
        super(ModelBase, self).__init__()

    def init_with_config(self, config: Config):
        self.config = config
        self.hparam = config.hparam
        self.loss_weight = self.hparam.loss_weight

        self._init_with_config(config)

    def _init_with_config(self, config: Config):
        pass

    def forward(self, *args, **kwargs):
        pass


class ModelBaseTraining(ModelBase, L.LightningModule):
    def __init__(self):
        super().__init__()

        self.session: TrainingSession = None

        self.register_buffer('_iterations', torch.zeros(1, dtype=torch.int32))
        self.register_buffer('_epochs', torch.zeros(1, dtype=torch.int32))

        self.profiling = Profiler()

        self.train_metrics = dict()
        self.val_metrics = dict()

    def set_session(self, session: TrainingSession):
        self.session = session

    def _init_with_config(self, config: Config):
        self.logging = ModuleLoggers(self.session.get_work_dir_path('logs'))

        self.logging.create_logger('logger', logs.Plain, 'logs')
        self.logging.create_logger('metrics_train', logs.TensorBoard, 'metrics/by_epoch/train')
        self.logging.create_logger('metrics_val', logs.TensorBoard, 'metrics/by_epoch/val')

    def on_validation_epoch_end(self):
        logger = self.logging.metrics_val.logger

        print(f'Metric for validation at {self.epochs} epochs')
        for k, v in self.val_metrics.items():
            value = v
            if isinstance(v, Metric):
                value = v.compute()

            logger.add_scalar(k, value, self.epochs)
            print(k, value, sep=': ')

    def train(self, mode: bool = True):

        if not mode:
            self.profiling.disable()
        else:
            self.profiling.enable()

        return super().train(mode)

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    @property
    def iterations(self):
        return int(self._iterations.item())

    @property
    def epochs(self):
        return int(self._epochs.item())

    def on_train_batch_end(self, outputs, batch: Any, batch_idx: int):
        self._iterations += 1

    def on_train_epoch_end(self):
        logger = self.logging.metrics_train.logger

        print(f'Metric for training at {self.epochs} epochs')
        for k, v in self.train_metrics.items():
            value = v
            if isinstance(v, Metric):
                value = v.compute()

            logger.add_scalar(k, value, self.epochs)
            print(k, value, sep=': ')

        self._epochs += 1

    def reset_train_state(self):
        self._iterations = torch.zeros(1, dtype=torch.int32).to(self._iterations)
        self._epochs = torch.zeros(1, dtype=torch.int32).to(self._epochs)

    def update_metrics(self, metrics: dict, storage, creator):
        for k, v in metrics.items():
            if k not in storage:
                storage[k] = creator()

            if torch.is_tensor(v):
                v = v.cpu()
            storage[k].update(v)


ModelVariant = namedtuple('ModelVariant', ['id', 'model_class'])
