import datetime
import os
from dataclasses import dataclass

import logging

from torch.utils.tensorboard import SummaryWriter

from utils.file import find_available_increasing_name


@dataclass
class _LoggerType:
    name: str


Plain = _LoggerType('plain')
TensorBoard = _LoggerType('tensorboard')


class Logger:
    def __init__(self, name, path):
        self.name = name
        self.path = path


class PlainLogger(Logger):
    def __init__(self, name, path, avoid_conflict=True):
        super().__init__(name, path)

        self.logger = logging.getLogger(name)
        self.logger.setLevel(level=logging.INFO)

        date = datetime.date.today()

        if avoid_conflict:
            filename = f'{path}/{find_available_increasing_name(path, f"{date}-")}.log'
        else:
            filename = os.path.join(path, f'{name}.log')

        handler = logging.FileHandler(filename, encoding='UTF-8')
        handler.setLevel(logging.INFO)

        formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', "%m-%d %H:%M")
        handler.setFormatter(formatter)

        console = logging.StreamHandler()
        console.setFormatter(formatter)
        console.setLevel(logging.INFO)

        self.logger.addHandler(handler)
        self.logger.addHandler(console)

    def log(self, tag, level, message):
        self.logger.log(level, msg=f'[{tag}]: {message}', stacklevel=2)

    def info(self, tag, message):
        self.logger.info(msg=f'[{tag}]: {message}', stacklevel=2)

    def error(self, tag, message):
        self.logger.error(msg=f'[{tag}]: {message}', stacklevel=2)

    def warning(self, tag, message):
        self.logger.warning(msg=f'[{tag}]: {message}', stacklevel=2)


class TensorBoardLogger(Logger):
    def __init__(self, name, path):
        super().__init__(name, path)
        self.path = path
        self._logger = SummaryWriter(path)

    @property
    def logger(self) -> SummaryWriter:
        return self._logger


class ModuleLoggers:
    def __init__(self, base_dir):
        super().__init__()
        self.base_dir = base_dir

    def create_logger(self, name, type: _LoggerType, subpath, *args, **kwargs):
        logger = None

        path = os.path.join(self.base_dir, subpath)

        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        if type == Plain:
            logger = PlainLogger(name, path, *args, **kwargs)
        elif type == TensorBoard:
            logger = TensorBoardLogger(name, path, *args, **kwargs)

        setattr(self, name, logger)

    def get_logger(self, name):
        logger = getattr(self, name)

        return logger
