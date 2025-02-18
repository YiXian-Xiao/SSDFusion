from math import exp

import kornia
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.constants import eps


# Implementation from PSFusion
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)  # sigma = 1.5    shape: [11, 1]
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(
        0)  # unsqueeze()函数,增加维度  .t() 进行了转置 shape: [1, 1, 11, 11]
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()  # window shape: [1,1, 11, 11]
    return window


def mse(img1, img2, window_size=9):
    max_val = 255
    min_val = 0
    L = max_val - min_val
    padd = window_size // 2

    (_, channel, height, width) = img1.size()

    img1_f = F.unfold(img1, (window_size, window_size), padding=padd)
    img2_f = F.unfold(img2, (window_size, window_size), padding=padd)

    res = (img1_f - img2_f + eps) ** 2

    res = torch.sum(res, dim=1, keepdim=True) / (window_size ** 2)

    res = F.fold(res, output_size=(height, width), kernel_size=(1, 1))
    return res


# 方差计算
def std(img, window_size=9):
    padd = window_size // 2
    (_, channel, height, width) = img.size()
    window = create_window(window_size, channel=channel).to(img.device)
    mu = F.conv2d(img, window, padding=padd, groups=channel)
    mu_sq = mu.pow(2)
    sigma1 = F.conv2d(img * img, window, padding=padd, groups=channel) - mu_sq

    return sigma1


def final_mse1(img_ir, img_vis, img_fuse, mask=None):
    mse_ir = mse(img_ir, img_fuse)
    mse_vi = mse(img_vis, img_fuse)

    std_ir = std(img_ir)
    std_vi = std(img_vis)
    # std_ir = sum(img_ir)
    # std_vi = sum(img_vis)

    zero = torch.zeros_like(std_ir)
    one = torch.ones_like(std_vi)

    m = torch.mean(img_ir)
    map1 = torch.where((std_ir - std_vi) > 0, one, zero)
    # map2 = torch.where((std_ir - std_vi) >= 0, zero, one)
    map_ir = torch.where(map1 + mask > 0, one, zero)
    map_vi = 1 - map_ir

    res = map_ir * mse_ir + map_vi * mse_vi
    # res = res * w_vi
    return res.mean()


class FusionLoss(nn.Module):
    def __init__(self):
        super(FusionLoss, self).__init__()
        self.sobel = SobelXY()
        self.ssim = kornia.losses.SSIMLoss(11, reduction='mean')
        self.weight_intensity = 1
        self.weight_grad = 10

    def forward(self, image_vis, image_ir, generate_img, mask):
        image_y = image_vis[:, :1, :, :]

        loss_in = self.intensity_loss(generate_img, image_vis, image_ir, mask)
        loss_grad = self.gradient_loss(generate_img, image_y, image_ir)
        loss_total = self.weight_intensity * loss_in + self.weight_grad * loss_grad

        return loss_total, loss_in, loss_grad

    def l2_loss(self, input, target, *args, **kwargs):
        return F.mse_loss(input, target, *args, **kwargs)

    def intensity_loss(self, img_fused, img_vi, img_ir, mask):
        return final_mse1(img_ir, img_vi, img_fused, mask)  # + 0 * F.l1_loss(fu, torch.max(ir, vi))

    def gradient_loss(self, img_hat, img_x, img_y=None):
        img_grad = self.sobel(img_x)
        hat_grad = self.sobel(img_hat)

        if img_y is not None:
            y_grad = self.sobel(img_y)
            img_grad = torch.max(img_grad, y_grad)

        return F.l1_loss(hat_grad, img_grad)

    def img_loss(self, src, tgt, weights=(0.9, 0.1)):
        return weights[0] * (F.l1_loss(src, tgt) + F.mse_loss(src, tgt)) + weights[1] * self.gradient_loss(src, tgt)


class SobelXY(nn.Module):
    def __init__(self):
        super(SobelXY, self).__init__()
        kernelx = [[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]]
        kernely = [[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False)
        self.weighty = nn.Parameter(data=kernely, requires_grad=False)

    def forward(self, x):
        sobelx = F.conv2d(x, self.weightx, padding=1)
        sobely = F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx) + torch.abs(sobely)


def cc(src, hat):
    eps = torch.finfo(torch.float32).eps
    """Correlation coefficient for (N, C, H, W) image; torch.float32 [0.,1.]."""
    N, C, _, _ = src.shape
    src = src.reshape(N, C, -1)
    hat = hat.reshape(N, C, -1)
    src = src - src.mean(dim=-1, keepdim=True)
    hat = hat - hat.mean(dim=-1, keepdim=True)

    b = torch.sqrt(torch.sum(src ** 2, dim=-1)) * torch.sqrt(torch.sum(hat ** 2, dim=-1))
    cc = (torch.sum(src * hat, dim=-1) /
          (eps + b))
    cc = torch.clamp(cc, -1., 1.)
    return cc.mean()
