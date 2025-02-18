from utils.config import Config

import os


class TrainingSession:
    def __init__(self, config: Config, session_name: str):
        self._config = config
        self._session_name = session_name
        self._work_dir = os.path.join('work', self.session_name)

    @property
    def config(self):
        return self._config

    @property
    def session_name(self):
        return self._session_name

    @property
    def work_dir(self):
        return self._work_dir

    def get_work_dir_path(self, *paths):
        return os.path.join(self.work_dir, *paths)

    def get_work_dir_created(self, *paths):
        path = self.get_work_dir_path(*paths)
        os.makedirs(path, exist_ok=True)
        return path
