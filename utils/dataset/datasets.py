import cv2
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import numpy as np
from PIL import Image
from kornia.color import rgb_to_ycbcr, rgb_to_grayscale
from lightning import LightningDataModule
from torch.utils import data

from utils.config import Config
from utils.image import randflow, randrot, randfilp, clahe, cv_img_to_pil
from torch.utils.data import Dataset, DataLoader
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
        return vis.squeeze(0), ir.squeeze(0)

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
        self.crop_size = crop_size
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
        ir = self.imread(path=ir_path, gray=True)
        label = self.imread(path=label_path, label=True)

        vis_ir = torch.cat([vis, ir, label], dim=1)
        if vis_ir.shape[-1] <= self.crop_size or vis_ir.shape[-2] <= self.crop_size:
            vis_ir = TF.resize(vis_ir, self.crop_size)
        vis_ir = randfilp(vis_ir)
        vis_ir = randrot(vis_ir)
        patch = self.crop(vis_ir)

        vis, ir, label = torch.split(patch, [3, 1, 1], dim=1)
        h, w = vis_ir.shape[2], vis_ir.shape[3]
        label = label.type(torch.LongTensor)
        return vis.squeeze(0), ir.squeeze(0), label.squeeze(0)

    def __len__(self):
        return len(self.vis_list)

    @staticmethod
    def imread(path, label=False, gray=False):
        if label:
            img = Image.open(path)
            im_ts = TF.to_tensor(img).unsqueeze(0) * 255
        else:
            img = Image.open(path).convert('RGB')
            if gray:
                im_ts = rgb_to_grayscale(TF.to_tensor(img).unsqueeze(0))
            else:
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


class MetaDataModule(LightningDataModule):

    def __init__(self, cfg: Config, train_dataset, val_dataset):
        super().__init__()
        self.cfg = cfg
        self.dataset_train = train_dataset
        self.dataset_val = val_dataset

        generator = torch.Generator().manual_seed(42)
        self.dataset_meta_train, self.dataset_meta_test = data.random_split(self.dataset_train,
                                                                            (0.8, 0.2),
                                                                            generator=generator)
        self.dataset_meta_train = self.wrap_dataloader(self.dataset_meta_train, batch_size=1)
        self.dataset_meta_test = self.wrap_dataloader(self.dataset_meta_test, batch_size=1)

    def setup(self, stage: str) -> None:
        pass

    def wrap_dataloader(self, dataset, batch_size=None, shuffle=True):
        return torch.utils.data.DataLoader(dataset,
                                           batch_size=self.cfg.train.batch_size if batch_size is None else 1,
                                           num_workers=self.cfg.dataset.loader_threads,
                                           persistent_workers=True,
                                           shuffle=shuffle)

    def train_dataloader(self):
        return self.wrap_dataloader(self.dataset_train)

    def get_meta_train_datasets(self):
        return self.dataset_meta_train, self.dataset_meta_test


class VIFH5SegDataset(data.Dataset):

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
        LBL = np.array(h5f['label_patchs'][key])
        h5f.close()
        img_vi, img_ir, label = torch.tensor(VIS), torch.tensor(IR), torch.tensor(LBL)

        imgs = torch.cat([img_vi, img_ir, label])
        imgs = self.transform(imgs)
        img_vi, img_ir, label = imgs[0:3], imgs[3:4], imgs[4:5]

        return img_vi, img_ir, label


class VIFH5MetaDataset(data.Dataset):

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
        LBL = np.array(h5f['label_patchs'][key])
        h5f.close()
        img_vi, img_ir, label = torch.Tensor(VIS), torch.Tensor(IR), torch.Tensor(LBL)

        imgs = torch.cat([img_vi, img_ir, label])
        imgs = self.transform(imgs)
        img_vi, img_ir, label = imgs[0:3], imgs[3:4], imgs[4:5]

        return img_vi, img_ir, label


class VIFH5SegDataset(data.Dataset):

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
        LBL = np.array(h5f['label_patchs'][key])
        h5f.close()
        img_vi, img_ir, label = torch.Tensor(VIS), torch.Tensor(IR), torch.Tensor(LBL)

        imgs = torch.cat([img_vi, img_ir, label])
        imgs = self.transform(imgs)
        img_vi, img_ir, label = imgs[0:3], imgs[3:4], imgs[4:5]

        return img_vi, img_ir, label


class VIFValImageDataset(torch.utils.data.Dataset):

    def __init__(self, path, crop=128):
        super().__init__()
        self.vis_folder = os.path.join(path, 'vi')
        self.ir_folder = os.path.join(path, 'ir')
        # gain infrared and visible images list
        self.filelist = sorted(os.listdir(self.vis_folder))
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(crop, antialias=True),
            torchvision.transforms.CenterCrop(crop)
        ])

    def __getitem__(self, index):
        image_name = self.filelist[index]
        vis_path = os.path.join(self.vis_folder, image_name)
        ir_path = os.path.join(self.ir_folder, image_name)
        vis = self.imread(path=vis_path, equalize=True)
        ir = self.imread(path=ir_path)

        imgs = self.transform(torch.cat([vis, ir]))

        vis, ir = imgs.chunk(2)
        ir = rgb_to_grayscale(ir)

        return vis, ir

    def __len__(self):
        return len(self.filelist)

    @staticmethod
    def imread(path, equalize=False):
        img = Image.open(path).convert('RGB')
        im_ts = TF.to_tensor(img)
        return im_ts


class VIFValSegImageDataset(torch.utils.data.Dataset):

    def __init__(self, path, crop=128):
        super().__init__()
        self.vis_folder = os.path.join(path, 'vi')
        self.ir_folder = os.path.join(path, 'ir')
        self.label_folder = os.path.join(path, 'Segmentation_labels')
        # gain infrared and visible images list
        self.filelist = sorted(os.listdir(self.vis_folder))
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(crop, antialias=True),
            torchvision.transforms.CenterCrop(crop)
        ])

    def __getitem__(self, index):
        image_name = self.filelist[index]
        vis_path = os.path.join(self.vis_folder, image_name)
        ir_path = os.path.join(self.ir_folder, image_name)
        label_path = os.path.join(self.label_folder, image_name)
        vis = self.imread(path=vis_path, equalize=True)
        ir = self.imread(path=ir_path)
        label = self.imread_gray(path=label_path)

        imgs = self.transform(torch.cat([vis, ir, label]))

        vis, ir, label = imgs[0:3], imgs[3:6], imgs[6:7]

        ir = rgb_to_grayscale(ir)

        return vis, ir, label

    def __len__(self):
        return len(self.filelist)

    @staticmethod
    def imread(path, equalize=False):
        img = Image.open(path).convert('RGB')
        im_ts = TF.to_tensor(img)
        return im_ts

    @staticmethod
    def imread_gray(path, equalize=False):
        img = Image.open(path).convert('L')
        im_ts = TF.to_tensor(img)
        return im_ts * 255


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
        ir = self.imread(path=ir_path, gray=True)

        return vis.squeeze(0), ir.squeeze(0), image_name

    def __len__(self):
        return len(self.filelist)

    @staticmethod
    def imread(path, equalize=False, gray=False):
        img = Image.open(path).convert('RGB')
        if gray:
            im_ts = rgb_to_grayscale(TF.to_tensor(img)).unsqueeze(0)
        else:
            im_ts = TF.to_tensor(img).unsqueeze(0)
        return im_ts


class VIFTestSegImageDataset(torch.utils.data.Dataset):

    def __init__(self, path):
        super().__init__()
        self.vis_folder = os.path.join(path, 'vi')
        self.ir_folder = os.path.join(path, 'ir')
        self.label_folder = os.path.join(path, 'Segmentation_labels')
        # gain infrared and visible images list
        self.filelist = sorted(os.listdir(self.vis_folder))

    def __getitem__(self, index):
        image_name = self.filelist[index]
        vis_path = os.path.join(self.vis_folder, image_name)
        ir_path = os.path.join(self.ir_folder, image_name)
        label_path = os.path.join(self.label_folder, image_name)
        vis = self.imread(path=vis_path, equalize=True)
        ir = self.imread(path=ir_path)
        label = self.imread_gray(path=label_path)
        return vis.squeeze(0), ir.squeeze(0), label, image_name

    def __len__(self):
        return len(self.filelist)

    @staticmethod
    def imread(path, equalize=False):
        img = Image.open(path).convert('RGB')
        im_ts = rgb_to_grayscale(TF.to_tensor(img).unsqueeze(0))
        return im_ts

    @staticmethod
    def imread_gray(path, equalize=False):
        img = Image.open(path).convert('L')
        im_ts = TF.to_tensor(img)
        return im_ts * 255
