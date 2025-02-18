import timeit

import torch

from model.model import NewModelBase


def setup():
    model = NewModelBase()
    return model.eval().cuda().requires_grad_(False)

def random_input():
    return (torch.randn(1, 1, 256, 256, device=torch.device('cuda')),
            torch.randn(1, 1, 256, 256, device=torch.device('cuda')))

if __name__ == '__main__':
    print(timeit.timeit('m(*random_input())', setup='from __main__ import setup, random_input; m = setup()', number=1000))
