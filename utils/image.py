import cv2
import torch
import random
import kornia
import kornia.geometry.transform as KGT
import kornia.utils as KU
import kornia.filters as KF

from PIL import Image


def RGB2YCbCr(rgb_image):
    """
    将RGB格式转换为YCrCb格式
    用于中间结果的色彩空间转换中,因为此时rgb_image默认size是[B, C, H, W]
    :param rgb_image: RGB格式的图像数据
    :return: Y, Cr, Cb
    """

    R = rgb_image[:, 0:1]
    G = rgb_image[:, 1:2]
    B = rgb_image[:, 2:3]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    with torch.no_grad():
        Cr = (R - Y) * 0.713 + 0.5
        Cb = (B - Y) * 0.564 + 0.5

    Y = Y.clamp(0.0, 1.0)
    Cr = Cr.clamp(0.0, 1.0).detach()
    Cb = Cb.clamp(0.0, 1.0).detach()

    return Y, Cb, Cr


def YCbCr2RGB(Y, Cb, Cr):
    """
    将YcrCb格式转换为RGB格式
    :param Y:
    :param Cb:
    :param Cr:
    :return:
    """
    ycrcb = torch.cat([Y, Cr, Cb], dim=1)
    B, C, W, H = ycrcb.shape
    im_flat = ycrcb.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor([[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
                       ).to(Y)
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).to(Y)
    temp = (im_flat + bias).mm(mat)
    out = temp.reshape(B, W, H, C).transpose(1, 3).transpose(2, 3)
    out = out.clamp(0, 1.0)
    return out


def randrot(img):
    mode = random.randint(0, 4)
    return rot(img, mode)


def randfilp(img):
    mode = random.randint(0, 3)
    return flip(img, mode)


def rot(img, rot_mode):
    if rot_mode == 0:
        img = img.transpose(-2, -1)
        img = img.flip(-2)
    elif rot_mode == 1:
        img = img.flip(-2)
        img = img.flip(-1)
    elif rot_mode == 2:
        img = img.flip(-2)
        img = img.transpose(-2, -1)
    return img


def flip(img, flip_mode):
    if flip_mode == 0:
        img = img.flip(-2)
    elif flip_mode == 1:
        img = img.flip(-1)
    return img


def randflow(img, angle=7, trans=0.07, ratio=1, sigma=15, base=500):
    h, w = img.shape[-2], img.shape[-1]
    # affine
    if base is not None:
        base_scale = base / torch.tensor([w, h]).to(img).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        angle = max(w, h) / base * angle
        # print('hello')
    else:
        base_scale = 1
    rand_angle = (torch.rand(1) * 2 - 1) * angle
    rand_trans = (torch.rand(1, 2) * 2 - 1) * trans
    M = KGT.get_affine_matrix2d(translations=rand_trans, center=torch.zeros(1, 2), scale=torch.ones(1, 2),
                                angle=rand_angle)
    M = M.inverse()
    grid = KU.create_meshgrid(h, w).to(img)
    warp_grid = kornia.geometry.linalg.transform_points(M, grid)
    # warp_grid = grid
    # elastic
    disp = torch.rand([1, 2, h, w]) * 2 - 1
    # disp = KF.gaussian_blur2d(disp,kernel_size=[(3*sigma)//2*2+1,(3*sigma)//2*2+1],sigma=[sigma,sigma])
    for i in range(5):
        disp = KF.gaussian_blur2d(disp, kernel_size=((3 * sigma) // 2 * 2 + 1, (3 * sigma) // 2 * 2 + 1),
                                  sigma=(sigma, sigma))
    disp = KF.gaussian_blur2d(disp, kernel_size=((3 * sigma) // 2 * 2 + 1, (3 * sigma) // 2 * 2 + 1),
                              sigma=(sigma, sigma)).permute(0, 2, 3, 1) * ratio

    disp = (disp + warp_grid - grid) * base_scale
    trans_grid = grid + disp
    mask = trans_grid < -1
    mask = torch.logical_or(trans_grid > 1, mask)

    return trans_grid, trans_grid - grid, mask


clahe_obj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def clahe(cvimg):
    return clahe_obj.apply(cvimg)


def cv_img_to_pil(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
