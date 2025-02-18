import time
from collections import namedtuple, OrderedDict
from contextlib import contextmanager


class Profiler:

    _TimeUnits = namedtuple('TimeUnits', ['millisecond', 'second', 'minute'])
    Units = _TimeUnits(('ms/it', 1000), ('s/it', 1), ('m/it', 1 / 60))

    def __init__(self, unit=Units.millisecond):
        super().__init__()
        self._records = OrderedDict()
        self.enabled = True
        self.unit = unit

    def disable(self):
        self.enabled = False

    def enable(self):
        self.enabled = True

    def records(self):
        return self._records

    def reset(self):
        self._records = dict()

    def reset_scope(self, name):
        if name in self._records:
            self._records[name].reset()

    @contextmanager
    def scope(self, name, reset=False):
        """
        Records execution time inside the scope

        :param reset: Whether to reset the scope's metrics
        :param name: A hierarchical name is recommended, e.g. example/test
        """

        if name not in self._records:
            self._records[name] = ProfilerRecord(name, self.unit[0])

        if reset:
            self.reset_scope(name)

        start_time = time.time()
        try:
            yield None
        finally:
            if self.enabled:
                elapsed_time = time.time() - start_time

                self._records[name] += elapsed_time * self.unit[1]


class ProfilerRecord:
    def __init__(self, name, unit=''):
        self.name = name
        self.unit = unit
        self.value = 0.0
        self.max = float('-inf')
        self.min = float('inf')
        self.count = 0
        self.precision = 3

    def __repr__(self):
        return f'{self.name}: {self.min:.{self.precision}f}/{self.average():.{self.precision}f}/{self.max:.{self.precision}f}({self.unit})'

    def increase(self, value):
        self.count += 1
        self.value += value
        self.max = max(value, self.max)
        self.min = min(value, self.min)

    def average(self):
        return (self.value / self.count) if self.count != 0 else 0.0

    def reset(self):
        self.value = 0
        self.count = 0

    def __add__(self, other):
        self.increase(other)
        return self
