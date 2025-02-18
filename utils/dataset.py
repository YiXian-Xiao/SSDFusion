import cv2
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import numpy as np
from PIL import Image
from kornia.color import rgb_to_ycbcr, rgb_to_grayscale
from torch.utils import data

from utils.image import randflow, randrot, randfilp, clahe, cv_img_to_pil
from torch.utils.data import Dataset
import os
import h5py


class VIFImageDataset(torch.utils.data.Dataset):
    """
    Load dataset with infrared folder path and visible folder path
    """

    # TODO: remove ground truth reference
    def __init__(self, path, crop_size=256):
        super(VIFImageDataset, self).__init__()
        self.vis_folder = os.path.join(path, 'vi')
        self.ir_folder = os.path.join(path, 'ir')
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(crop_size)
        ])
        # gain infrared and visible images list
        self.vis_list = sorted(os.listdir(self.vis_folder))
        self.ir_list = sorted(os.listdir(self.ir_folder))

    def __getitem__(self, index):
        # gain image path
        image_name = self.ir_list[index]
        vis_path = os.path.join(self.vis_folder, image_name)
        ir_path = os.path.join(self.ir_folder, image_name)
        # read image as type Tensor
        vis = self.imread(path=vis_path)
        ir = self.imread(path=ir_path)
        ir = rgb_to_grayscale(ir)

        vis_ir = torch.cat([vis, ir], dim=1)
        vis_ir = randfilp(vis_ir)
        vis_ir = randrot(vis_ir)
        patch = self.transform(vis_ir)

        vis, ir = torch.split(patch, [3, 1], dim=1)
        h, w = vis_ir.shape[2], vis_ir.shape[3]
        return ir.squeeze(0), vis.squeeze(0)

    def __len__(self):
        return len(self.vis_list)

    @staticmethod
    def imread(path, label=False):
        img = Image.open(path).convert('RGB')
        im_ts = TF.to_tensor(img).unsqueeze(0)
        return im_ts


class VIFSegImageDataset(torch.utils.data.Dataset):
    """
    Load dataset with infrared folder path and visible folder path
    """

    def __init__(self, path, crop_size=256):
        super(VIFSegImageDataset, self).__init__()
        self.vis_folder = os.path.join(path, 'vi')
        self.ir_folder = os.path.join(path, 'ir')
        self.label_folder = os.path.join(path, 'Segmentation_labels')
        self.crop = torchvision.transforms.RandomCrop(crop_size)
        # gain infrared and visible images list
        self.vis_list = sorted(os.listdir(self.vis_folder))
        self.ir_list = sorted(os.listdir(self.ir_folder))
        self.label_list = sorted(os.listdir(self.label_folder))
        print(len(self.vis_list), len(self.ir_list), len(self.label_list))

    def __getitem__(self, index):
        # gain image path
        image_name = self.ir_list[index]
        vis_path = os.path.join(self.vis_folder, image_name)
        ir_path = os.path.join(self.ir_folder, image_name)
        label_path = os.path.join(self.label_folder, image_name)
        # read image as type Tensor
        vis = self.imread(path=vis_path)
        ir = self.imread(path=ir_path)
        label = self.imread(path=label_path, label=True)

        vis_ir = torch.cat([vis, ir, label], dim=1)
        if vis_ir.shape[-1] <= 256 or vis_ir.shape[-2] <= 256:
            vis_ir = TF.resize(vis_ir, 256)
        vis_ir = randfilp(vis_ir)
        vis_ir = randrot(vis_ir)
        patch = self.crop(vis_ir)

        vis, ir, label = torch.split(patch, [3, 3, 1], dim=1)
        h, w = vis_ir.shape[2], vis_ir.shape[3]
        label = label.type(torch.LongTensor)
        return ir.squeeze(0), vis.squeeze(0), label.squeeze(0)

    def __len__(self):
        return len(self.vis_list)

    @staticmethod
    def imread(path, label=False):
        if label:
            img = Image.open(path)
            im_ts = TF.to_tensor(img).unsqueeze(0) * 255
        else:
            img = Image.open(path).convert('RGB')
            im_ts = TF.to_tensor(img).unsqueeze(0)
        return im_ts


class RegDataset(torch.utils.data.Dataset):
    """
    Load dataset with infrared folder path and visible folder path
    """

    # TODO: remove ground truth reference
    def __init__(self, path, crop=lambda x: x):
        super(RegDataset, self).__init__()
        self.vis_folder = os.path.join(path, 'vi')
        self.ir_folder = os.path.join(path, 'ir')
        self.crop = torchvision.transforms.RandomCrop(256)
        # gain infrared and visible images list
        self.vis_list = sorted(os.listdir(self.vis_folder))
        self.ir_list = sorted(os.listdir(self.ir_folder))
        print(len(self.vis_list), len(self.ir_list))

    def __getitem__(self, index):
        # gain image path
        vis_path = os.path.join(self.vis_folder, self.vis_list[index])
        ir_path = os.path.join(self.ir_folder, self.ir_list[index])

        assert os.path.basename(vis_path) == os.path.basename(
            ir_path), f"Mismatch ir:{os.path.basename(ir_path)} vi:{os.path.basename(vis_path)}."

        # read image as type Tensor
        vis = self.imread(path=vis_path, flags=cv2.IMREAD_GRAYSCALE)
        ir = self.imread(path=ir_path, flags=cv2.IMREAD_GRAYSCALE)

        vis_ir = torch.cat([vis, ir], dim=1)
        if vis_ir.shape[-1] <= 256 or vis_ir.shape[-2] <= 256:
            vis_ir = TF.resize(vis_ir, 256)
        vis_ir = randfilp(vis_ir)
        vis_ir = randrot(vis_ir)

        flow, disp, _ = randflow(vis_ir, 10, 0.1, 1)
        vis_ir_warped = F.grid_sample(vis_ir, flow, align_corners=False, mode='bilinear')
        patch = torch.cat([vis_ir, vis_ir_warped, disp.permute(0, 3, 1, 2)], dim=1)
        patch = self.crop(patch)

        vis, ir, vis_warped, ir_warped, disp = torch.split(patch, [3, 3, 3, 3, 2], dim=1)
        h, w = vis_ir.shape[2], vis_ir.shape[3]
        scale = (torch.FloatTensor([w, h]).unsqueeze(0).unsqueeze(0) - 1) / (self.crop.size[0] * 1.0 - 1)
        # print(self.crop.size[0])
        disp = disp.permute(0, 2, 3, 1) * scale
        # vis_warped_ = self.ST(vis.unsqueeze(0),disp.unsqueeze(0))
        # TF.to_pil_image(((vis_warped-vis_warped_).abs()).squeeze(0)).save('error.png')
        # disp_crop = disp[:,h//2-150:h//2+150,w//2-150:w//2+150,:]*scale
        return ir.squeeze(0), vis.squeeze(0), ir_warped.squeeze(0), vis_warped.squeeze(0), disp.squeeze(0)

    def __len__(self):
        return len(self.vis_list)

    @staticmethod
    def imread(path, flags=cv2.IMREAD_GRAYSCALE):
        # im_cv = cv2.imread(str(path), flags)
        # assert im_cv is not None, f"Image {str(path)} is invalid."
        # im_ts = kornia.utils.image_to_tensor(im_cv / 255.,keepdim=False).type(torch.FloatTensor)
        img = Image.open(path).convert('RGB')
        im_ts = TF.to_tensor(img).unsqueeze(0)
        return im_ts


class VIFSegTestImageDataset(Dataset):
    def __init__(
            self,
            rootpth,
            mode='train',
            *args,
            **kwargs
    ):
        super(VIFSegTestImageDataset, self).__init__(*args, **kwargs)
        assert mode in ('train', 'val', 'test')
        self.mode = mode
        self.ignore_lb = 255
        # impth = osp.join(rootpth, Method, mode)
        self.vi_dir = os.path.join(rootpth, 'vi')
        self.ir_dir = os.path.join(rootpth, 'ir')
        self.label_dir = os.path.join(rootpth, 'Segmentation_labels')
        self.file_list = sorted(os.listdir(self.vi_dir))

        ## pre-processing
        self.to_tensor = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

    def __getitem__(self, idx):
        fn = self.file_list[idx]
        vi_path = os.path.join(self.vi_dir, fn)
        ir_path = os.path.join(self.ir_dir, fn)
        lbpth = os.path.join(self.label_dir, fn)
        vi = self.to_tensor(Image.open(vi_path).convert('RGB'))
        ir = self.to_tensor(Image.open(ir_path).convert('RGB'))
        label = Image.open(lbpth)
        label = np.array(label).astype(np.int64)[np.newaxis, :]
        return vi, ir, label, fn

    def __len__(self):
        return len(self.file_list)


class VIFH5Dataset(data.Dataset):

    def __init__(self, h5file_path):
        self.h5file_path = h5file_path
        h5f = h5py.File(h5file_path, 'r')
        self.keys = list(h5f['ir_patchs'].keys())
        h5f.close()
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip(),
        ])

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        h5f = h5py.File(self.h5file_path, 'r')
        key = self.keys[index]
        IR = np.array(h5f['ir_patchs'][key])
        VIS = np.array(h5f['vis_patchs'][key])
        h5f.close()
        img_vi, img_ir = torch.Tensor(VIS), torch.Tensor(IR)

        imgs = torch.cat([img_vi, img_ir])
        imgs = self.transform(imgs)
        img_vi, img_ir = imgs[0:3], imgs[3:]

        return img_vi, img_ir


class VIFTestImageDataset(torch.utils.data.Dataset):

    def __init__(self, path):
        super(VIFTestImageDataset, self).__init__()
        self.vis_folder = os.path.join(path, 'vi')
        self.ir_folder = os.path.join(path, 'ir')
        # gain infrared and visible images list
        self.filelist = sorted(os.listdir(self.vis_folder))

    def __getitem__(self, index):
        image_name = self.filelist[index]
        vis_path = os.path.join(self.vis_folder, image_name)
        ir_path = os.path.join(self.ir_folder, image_name)
        vis = self.imread(path=vis_path, equalize=True)
        ir = self.imread(path=ir_path)

        return ir.squeeze(0), vis.squeeze(0), image_name

    def __len__(self):
        return len(self.filelist)

    @staticmethod
    def imread(path, equalize=False):
        img = Image.open(path).convert('RGB')
        im_ts = rgb_to_grayscale(TF.to_tensor(img).unsqueeze(0))
        return im_ts
