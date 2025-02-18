from mmseg import apis as mmseg
import torch
import torchinfo
from torch.utils.tensorboard import SummaryWriter

from model.model import NewModelBase, ModelBaseStageI

import lightning as L
import os

from copy import deepcopy

def test_model(model, total=1000):
    model = model.cpu()
    from timerit import Timerit
    from tqdm import tqdm
    t1 = Timerit(num=total)
    
    with torch.no_grad():
        for timer in tqdm(t1, total=total):
            input_data = (torch.randn((1, 1, 256, 256)), torch.randn((1, 1, 256, 256)))
            with timer:
                model(*input_data)

    t1.print(3)
    
if __name__ == '__main__':
    # torchinfo.summary(NewModelBase(), input_data=(torch.randn((1, 1, 256, 256)), torch.randn((1, 1, 256, 256))))
    
    from ptoolbox.utils.benchmark import profile, test_sample_model
    
    # profile(NewModelBase().cuda(), (torch.randn((1, 1, 256, 256)).cuda(), torch.randn((1, 1, 256, 256)).cuda()))

    test_sample_model(NewModelBase().cuda(), lambda: (torch.randn((1, 1, 256, 256)).cuda(), torch.randn((1, 1, 256, 256)).cuda()))
    # torchinfo.summary(mmseg.init_model('config/msrs/segmodel-test.py'), device='cuda:0', input_data=(torch.randn((1, 3, 512, 512))), depth=2)

    