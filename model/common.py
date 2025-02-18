import torch

from torch import nn, Tensor


class _DeviceAware(nn.Module):
    def __init__(self, device=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer('_dummy', torch.tensor([], device=device), persistent=False)

    def device(self):
        return self._dummy.device

    def convert(self, convertee: Tensor, dtype=True):
        return convertee.to(self._dummy.device, dtype=self._dummy.dtype if dtype else convertee.dtype)


class Module(_DeviceAware):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
