from collections import namedtuple

import numpy as np
import cv2
import torch
from kornia.metrics import ssim
from torch.nn import functional as F
import math
import torchmetrics
from torchmetrics.functional import clustering as TMFC


Metric = namedtuple('Metric', ['name', 'metric_type', 'metric'])


def image_read_cv2(path, mode='RGB'):
    img_BGR = cv2.imread(path).astype('float32')
    assert mode == 'RGB' or mode == 'GRAY' or mode == 'YCrCb', 'mode error'
    if mode == 'RGB':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
    elif mode == 'GRAY':
        img = np.round(cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY))
    elif mode == 'YCrCb':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb)
    return img


def convolve2d(x, kernel, padding=0):
    if len(x.shape) == 2:
        x = x.unsqueeze(0).unsqueeze(0)

    if len(kernel.shape) == 2:
        kernel = kernel.unsqueeze(0).unsqueeze(0)

    x = F.conv2d(x, kernel, padding=padding, stride=1)

    return x.squeeze(0).squeeze(0)


class Evaluator():
    @classmethod
    def input_check(cls, imgF, imgA=None, imgB=None):
        if imgA is None:
            assert type(imgF) == torch.Tensor, 'type error'
            assert len(imgF.shape) == 2, 'dimension error'
        else:
            assert type(imgF) == type(imgA) == type(imgB) == torch.Tensor, 'type error'
            assert imgF.shape == imgA.shape == imgB.shape, 'shape error'
            assert len(imgF.shape) == 2, 'dimension error'

    @classmethod
    def EN(cls, img):  # entropy
        cls.input_check(img)
        a = torch.round(img).to(dtype=torch.uint8).flatten()
        h = torch.bincount(a) / a.shape[0]
        return -torch.sum(h * torch.log2(h + (h == 0)))

    @classmethod
    def SD(cls, img):
        cls.input_check(img)
        return torch.std(img)

    @classmethod
    def SF(cls, img):
        cls.input_check(img)
        return torch.sqrt(torch.mean((img[:, 1:] - img[:, :-1]) ** 2) + torch.mean((img[1:, :] - img[:-1, :]) ** 2))

    @classmethod
    def AG(cls, img):  # Average gradient
        cls.input_check(img)
        Gx, Gy = torch.zeros_like(img), torch.zeros_like(img)

        Gx[:, 0] = img[:, 1] - img[:, 0]
        Gx[:, -1] = img[:, -1] - img[:, -2]
        Gx[:, 1:-1] = (img[:, 2:] - img[:, :-2]) / 2

        Gy[0, :] = img[1, :] - img[0, :]
        Gy[-1, :] = img[-1, :] - img[-2, :]
        Gy[1:-1, :] = (img[2:, :] - img[:-2, :]) / 2
        return torch.mean(torch.sqrt((Gx ** 2 + Gy ** 2) / 2))

    @classmethod
    def MI(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        image_F = torch.round(image_F).to(dtype=torch.uint8).flatten()
        image_A = torch.round(image_A).to(dtype=torch.uint8).flatten()
        image_B = torch.round(image_B).to(dtype=torch.uint8).flatten()
        return TMFC.mutual_info_score(image_F, image_A) + TMFC.mutual_info_score(image_F, image_B)

    @classmethod
    def MSE(cls, image_F, image_A, image_B):  # MSE
        cls.input_check(image_F, image_A, image_B)
        return (torch.mean((image_A - image_F) ** 2) + torch.mean((image_B - image_F) ** 2)) / 2

    @classmethod
    def CC(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)

        image_A = image_A - torch.mean(image_A)
        image_B = image_B - torch.mean(image_B)
        image_F = image_F - torch.mean(image_F)

        rAF = torch.sum(image_A * image_F) / torch.sqrt(
            (torch.sum(image_A ** 2)) * (torch.sum(image_F ** 2)))
        rBF = torch.sum(image_B * image_F) / torch.sqrt(
            (torch.sum(image_B ** 2)) * (torch.sum(image_F ** 2)))
        return (rAF + rBF) / 2

    @classmethod
    def PSNR(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        return 10 * torch.log10(torch.max(image_F) ** 2 / cls.MSE(image_F, image_A, image_B))

    @classmethod
    def SCD(cls, image_F, image_A, image_B):  # The sum of the correlations of differences
        cls.input_check(image_F, image_A, image_B)
        imgF_A = image_F - image_A
        imgF_B = image_F - image_B
        corr1 = torch.sum((image_A - torch.mean(image_A)) * (imgF_B - torch.mean(imgF_B))) / torch.sqrt(
            (torch.sum((image_A - torch.mean(image_A)) ** 2)) * (torch.sum((imgF_B - torch.mean(imgF_B)) ** 2)))
        corr2 = torch.sum((image_B - torch.mean(image_B)) * (imgF_A - torch.mean(imgF_A))) / torch.sqrt(
            (torch.sum((image_B - torch.mean(image_B)) ** 2)) * (torch.sum((imgF_A - torch.mean(imgF_A)) ** 2)))
        return corr1 + corr2

    @classmethod
    def VIFF(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        return cls.compare_viff(image_A, image_F) + cls.compare_viff(image_B, image_F)

    @classmethod
    def compare_viff(cls, ref, dist):  # viff of a pair of pictures
        sigma_nsq = 2
        eps = 1e-10

        num = 0.0
        den = 0.0
        for scale in range(1, 5):

            N = 2 ** (4 - scale + 1) + 1
            sd = N / 5.0

            # Create a Gaussian kernel as MATLAB's
            m, n = [(ss - 1.) / 2. for ss in (N, N)]
            y, x = np.ogrid[-m:m + 1, -n:n + 1]
            y = torch.Tensor(y).to(ref)
            x = torch.Tensor(x).to(ref)
            h = torch.exp(-(x * x + y * y) / (2. * sd * sd))
            h[h < torch.finfo(h.dtype).eps * h.max()] = 0
            sumh = h.sum()
            if sumh != 0:
                win = h / sumh

            if scale > 1:
                ref = convolve2d(ref, torch.rot90(win, 2)) # convolve2d()
                dist = convolve2d(dist, torch.rot90(win, 2)) # mode='valid')
                ref = ref[::2, ::2]
                dist = dist[::2, ::2]

            mu1 = convolve2d(ref, torch.rot90(win, 2))
            mu2 = convolve2d(dist, torch.rot90(win, 2))
            mu1_sq = mu1 * mu1
            mu2_sq = mu2 * mu2
            mu1_mu2 = mu1 * mu2
            sigma1_sq = convolve2d(ref * ref, torch.rot90(win, 2)) - mu1_sq
            sigma2_sq = convolve2d(dist * dist, torch.rot90(win, 2)) - mu2_sq
            sigma12 = convolve2d(ref * dist, torch.rot90(win, 2)) - mu1_mu2

            sigma1_sq[sigma1_sq < 0] = 0
            sigma2_sq[sigma2_sq < 0] = 0

            g = sigma12 / (sigma1_sq + eps)
            sv_sq = sigma2_sq - g * sigma12

            g[sigma1_sq < eps] = 0
            sv_sq[sigma1_sq < eps] = sigma2_sq[sigma1_sq < eps]
            sigma1_sq[sigma1_sq < eps] = 0

            g[sigma2_sq < eps] = 0
            sv_sq[sigma2_sq < eps] = 0

            sv_sq[g < 0] = sigma2_sq[g < 0]
            g[g < 0] = 0
            sv_sq[sv_sq <= eps] = eps

            num += torch.sum(torch.log10(1 + g * g * sigma1_sq / (sv_sq + sigma_nsq)))
            den += torch.sum(torch.log10(1 + sigma1_sq / sigma_nsq))

        vifp = num / den

        if torch.isnan(vifp):
            return 1.0
        else:
            return vifp

    @classmethod
    def Qabf(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        gA, aA = cls.Qabf_getArray(image_A)
        gB, aB = cls.Qabf_getArray(image_B)
        gF, aF = cls.Qabf_getArray(image_F)
        QAF = cls.Qabf_getQabf(aA, gA, aF, gF)
        QBF = cls.Qabf_getQabf(aB, gB, aF, gF)

        # 计算QABF
        deno = torch.sum(gA + gB)
        nume = torch.sum(torch.multiply(QAF, gA) + torch.multiply(QBF, gB))
        return nume / deno

    @classmethod
    def Qabf_getArray(cls, img):
        # Sobel Operator Sobel
        h1 = torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).to(img)
        h2 = torch.Tensor([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]).to(img)
        h3 = torch.Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).to(img)

        SAx = convolve2d(img, h3, padding='same')
        SAy = convolve2d(img, h1, padding='same')
        gA = torch.sqrt(torch.multiply(SAx, SAx) + torch.multiply(SAy, SAy))
        aA = torch.zeros_like(img)
        aA[SAx == 0] = math.pi / 2
        aA[SAx != 0] = torch.arctan(SAy[SAx != 0] / SAx[SAx != 0])
        return gA, aA

    @classmethod
    def Qabf_getQabf(cls, aA, gA, aF, gF):
        L = 1
        Tg = 0.9994
        kg = -15
        Dg = 0.5
        Ta = 0.9879
        ka = -22
        Da = 0.8
        GAF, AAF, QgAF, QaAF, QAF = torch.zeros_like(aA), torch.zeros_like(aA), torch.zeros_like(aA), torch.zeros_like(
            aA), torch.zeros_like(aA)
        GAF[gA > gF] = gF[gA > gF] / gA[gA > gF]
        GAF[gA == gF] = gF[gA == gF]
        GAF[gA < gF] = gA[gA < gF] / gF[gA < gF]
        AAF = 1 - torch.abs(aA - aF) / (math.pi / 2)
        QgAF = Tg / (1 + torch.exp(kg * (GAF - Dg)))
        QaAF = Ta / (1 + torch.exp(ka * (AAF - Da)))
        QAF = QgAF * QaAF
        return QAF

    @classmethod
    def SSIM(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        image_F = image_F.unsqueeze(0).unsqueeze(0)
        image_A = image_A.unsqueeze(0).unsqueeze(0)
        image_B = image_B.unsqueeze(0).unsqueeze(0)
        return ssim(image_F, image_A, window_size=7) + ssim(image_F, image_B, window_size=7)
        # consistent with skimage's ssim window size
