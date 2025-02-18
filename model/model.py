import itertools
import os
from copy import deepcopy
from enum import Enum

from torch.autograd import Variable
from tqdm import tqdm

from utils.dataset import MetaDataModule

from typing import Any

from mmseg import apis as mmseg
import torch
import lightning as L
from kornia.color import rgb_to_ycbcr, ycbcr_to_rgb, rgb_to_grayscale
from torch import nn, Tensor
from torch.nn import UpsamplingBilinear2d
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import MeanMetric, Metric
from torchvision.utils import make_grid

from model import losses
from model.base import ModelBase, ModelBaseTraining
from model.layers import Downsampler
from model.losses import FusionLoss
from model.modules import Restormer_Encoder, Restormer_Decoder, FusionModule, FeatureRefiner, WrappedSegmentor, \
    MetaFeatureExtractor, FeatureDecoder, FeatureTransform
from utils import logs
from utils.config import Config, adapters
from utils.image import RGB2YCbCr, YCbCr2RGB
from utils.profiler import Profiler
from utils.train import collect_params
from utils.train.session import TrainingSession


class NewModelBase(ModelBase):
    def __init__(self):
        super(NewModelBase, self).__init__()

        self.shared_encoder = Restormer_Encoder(dim=32)
        self.fusion_0 = FusionModule(32, 32, 32)

        self.refiner_1 = FeatureRefiner(32, 32, drop_path_rate=0.05)
        self.fusion_1 = FusionModule(32, 32, 32)
        self.down_1 = Downsampler(32)

        self.refiner_2 = FeatureRefiner(64, 32, drop_path_rate=0.1)
        self.fusion_2 = FusionModule(32, 32, 32)

        self.shared_decoder = Restormer_Decoder(dim=96)
        # Refactor drop path rate to config?

        # TODO: Find a way to guide hierarchical feature extraction

        self.up_2 = UpsamplingBilinear2d(scale_factor=2)
        self.up_4 = UpsamplingBilinear2d(scale_factor=4)

    def forward(self, image_vi, image_ir):
        feat_vi_0 = self.shared_encoder(image_vi)
        feat_ir_0 = self.shared_encoder(image_ir)

        fusion_output_0 = self.fusion_0(feat_ir_0, feat_vi_0)

        feat_ir_1, feat_vi_1 = self.refiner_1(feat_ir_0, feat_vi_0)
        fusion_output_1 = self.fusion_1(feat_ir_1, feat_vi_1)

        feat_ir_2, feat_vi_2 = self.refiner_2(torch.cat([self.down_1(feat_ir_0), feat_ir_1], dim=1),
                                              torch.cat([self.down_1(feat_vi_0), feat_vi_1], dim=1))
        fusion_output_2 = self.fusion_2(feat_ir_2, feat_vi_2)

        image_fused, _ = self.shared_decoder(image_ir, torch.cat([fusion_output_0.fused_feat,
                                                                  self.up_2(fusion_output_1.fused_feat),
                                                                  self.up_4(fusion_output_2.fused_feat)], dim=1))

        return image_fused


class NewModelBaseTraining(NewModelBase, ModelBaseTraining):
    def __init__(self):
        super().__init__()

        self.loss = FusionLoss()

class ModelBaseStageI(NewModelBaseTraining):
    def __init__(self):
        super().__init__()
        self.apply(self.no_fusion)

    @staticmethod
    def no_fusion(m):
        if isinstance(m, FusionModule):
            m.stage = 1

    def training_step(self, batch, batch_idx):
        with self.profiling.scope('train/forward'):
            image_vi, image_ir, _ = batch

            image_vi_Y, image_vi_Cb, image_vi_Cr = RGB2YCbCr(image_vi)

            feat_ir_0 = self.shared_encoder(image_ir)
            feat_vi_0 = self.shared_encoder(image_vi_Y)

            fusion_output_0 = self.fusion_0(feat_ir_0, feat_vi_0)

            feat_ir_1, feat_vi_1 = self.refiner_1(feat_ir_0, feat_vi_0)
            fusion_output_1 = self.fusion_1(feat_ir_1, feat_vi_1)

            feat_ir_2, feat_vi_2 = self.refiner_2(torch.cat([self.down_1(feat_ir_0), feat_ir_1], dim=1),
                                                  torch.cat([self.down_1(feat_vi_0), feat_vi_1], dim=1))
            fusion_output_2 = self.fusion_2(feat_ir_2, feat_vi_2)

            base_feats_ir = [fusion_output_0.base_feat_ir,
                             self.up_2(fusion_output_1.base_feat_ir),
                             self.up_4(fusion_output_2.base_feat_ir)]
            base_feat_ir = torch.cat(base_feats_ir, dim=1)
            detail_feats_ir = [fusion_output_0.detail_feat_ir,
                               self.up_2(fusion_output_1.detail_feat_ir),
                               self.up_4(fusion_output_2.detail_feat_ir)]
            detail_feat_ir = torch.cat(detail_feats_ir, dim=1)
            base_feats_vi = [fusion_output_0.base_feat_vi,
                             self.up_2(fusion_output_1.base_feat_vi),
                             self.up_4(fusion_output_2.base_feat_vi)]
            base_feat_vi = torch.cat(base_feats_vi, dim=1)
            detail_feats_vi = [fusion_output_0.detail_feat_vi,
                               self.up_2(fusion_output_1.detail_feat_vi),
                               self.up_4(fusion_output_2.detail_feat_vi)]
            detail_feat_vi = torch.cat(detail_feats_vi, dim=1)

            image_ir_hat, _ = self.shared_decoder(image_ir, torch.cat([fusion_output_0.fused_feat_ir,
                                                                      self.up_2(fusion_output_1.fused_feat_ir),
                                                                      self.up_4(fusion_output_2.fused_feat_ir)], dim=1))

            image_vi_Y_hat, _ = self.shared_decoder(image_ir, torch.cat([fusion_output_0.fused_feat_vi,
                                                                      self.up_2(fusion_output_1.fused_feat_vi),
                                                                      self.up_4(fusion_output_2.fused_feat_vi)], dim=1))

        with self.profiling.scope('train/calc_loss'):

            cc_loss_B = losses.cc(base_feats_vi[0], base_feats_ir[0])
            cc_loss_D = losses.cc(detail_feats_vi[0], detail_feats_ir[0])
            loss_decomp = (cc_loss_D ** 2 / (1.01 + cc_loss_B)) * self.loss_weight.weights_decomp[0]

            for layer in range(1, 3):
                cc_loss_B = losses.cc(base_feats_vi[layer], base_feats_ir[layer])
                cc_loss_D = losses.cc(detail_feats_vi[layer], detail_feats_ir[layer])

                loss_decomp += (cc_loss_D ** 2 / (1.01 + cc_loss_B)) * self.loss_weight.weights_decomp[layer]

            mse_loss_V = 5 * self.loss.ssim(image_vi_Y, image_vi_Y_hat) + self.loss.img_loss(image_vi_Y, image_vi_Y_hat)
            mse_loss_I = 5 * self.loss.ssim(image_ir, image_ir_hat) + self.loss.img_loss(image_ir, image_ir_hat)

            Gradient_loss = self.loss.gradient_loss(image_vi_Y_hat, image_vi_Y)

            loss = (self.loss_weight.coeff_mse_loss_VF * mse_loss_V +
                    self.loss_weight.coeff_mse_loss_IF * mse_loss_I +
                    self.loss_weight.coeff_decomp * loss_decomp +
                    self.loss_weight.coeff_tv * Gradient_loss)

        iterations = self.iterations + 1
        if self.config.logging.tensorboard.enabled:
            with self.profiling.scope('train/logging'):
                tensorboard: SummaryWriter = self.logger.experiment

                metrics = {
                    'loss/i/mse_vi': mse_loss_V,
                    'loss/i/mse_ir': mse_loss_I,
                    'loss/i/decomp': loss_decomp,
                    'loss/i/grad': Gradient_loss,
                    'loss/i/total': loss
                }

                for idx, lr in enumerate(self.lr_schedulers().get_last_lr()):
                    metrics[f'hparams/lr/{idx}'] = lr

                for k, v in metrics.items():
                    tensorboard.add_scalar(k, v, iterations)

                self.update_metrics(metrics, self.train_metrics, lambda: MeanMetric())

                if iterations % self.config.logging.tensorboard.log_image_freq == 0:
                    with self.profiling.scope('train/logging/image'):
                        with torch.no_grad():
                            tensorboard.add_image('display/i', make_grid([
                                image_vi_Y[0], image_ir[0],
                                image_vi_Y_hat[0], image_ir_hat[0]
                            ], nrow=2), iterations)

        return loss

    def validation_step(self, batch, batch_idx):
        with self.profiling.scope('val/forward'):
            image_vi, image_ir, _ = batch

            image_vi_Y, image_vi_Cb, image_vi_Cr = RGB2YCbCr(image_vi)

            feat_ir_0 = self.shared_encoder(image_ir)
            feat_vi_0 = self.shared_encoder(image_vi_Y)

            fusion_output_0 = self.fusion_0(feat_ir_0, feat_vi_0)

            feat_ir_1, feat_vi_1 = self.refiner_1(feat_ir_0, feat_vi_0)
            fusion_output_1 = self.fusion_1(feat_ir_1, feat_vi_1)

            feat_ir_2, feat_vi_2 = self.refiner_2(torch.cat([self.down_1(feat_ir_0), feat_ir_1], dim=1),
                                                  torch.cat([self.down_1(feat_vi_0), feat_vi_1], dim=1))
            fusion_output_2 = self.fusion_2(feat_ir_2, feat_vi_2)

            base_feats_ir = [fusion_output_0.base_feat_ir,
                             self.up_2(fusion_output_1.base_feat_ir),
                             self.up_4(fusion_output_2.base_feat_ir)]
            base_feat_ir = torch.cat(base_feats_ir, dim=1)
            detail_feats_ir = [fusion_output_0.detail_feat_ir,
                               self.up_2(fusion_output_1.detail_feat_ir),
                               self.up_4(fusion_output_2.detail_feat_ir)]
            detail_feat_ir = torch.cat(detail_feats_ir, dim=1)
            base_feats_vi = [fusion_output_0.base_feat_vi,
                             self.up_2(fusion_output_1.base_feat_vi),
                             self.up_4(fusion_output_2.base_feat_vi)]
            base_feat_vi = torch.cat(base_feats_vi, dim=1)
            detail_feats_vi = [fusion_output_0.detail_feat_vi,
                               self.up_2(fusion_output_1.detail_feat_vi),
                               self.up_4(fusion_output_2.detail_feat_vi)]
            detail_feat_vi = torch.cat(detail_feats_vi, dim=1)

            image_ir_hat, _ = self.shared_decoder(image_ir, torch.cat([fusion_output_0.fused_feat_ir,
                                                                       self.up_2(fusion_output_1.fused_feat_ir),
                                                                       self.up_4(fusion_output_2.fused_feat_ir)],
                                                                      dim=1))

            image_vi_Y_hat, _ = self.shared_decoder(image_ir, torch.cat([fusion_output_0.fused_feat_vi,
                                                                         self.up_2(fusion_output_1.fused_feat_vi),
                                                                         self.up_4(fusion_output_2.fused_feat_vi)],
                                                                        dim=1))

        with self.profiling.scope('val/calc_loss'):

            cc_loss_B = losses.cc(base_feats_vi[0], base_feats_ir[0])
            cc_loss_D = losses.cc(detail_feats_vi[0], detail_feats_ir[0])
            loss_decomp = (cc_loss_D ** 2 / (1.01 + cc_loss_B)) * self.loss_weight.weights_decomp[0]

            for layer in range(1, 3):
                cc_loss_B = losses.cc(base_feats_vi[layer], base_feats_ir[layer])
                cc_loss_D = losses.cc(detail_feats_vi[layer], detail_feats_ir[layer])

                loss_decomp += (cc_loss_D ** 2 / (1.01 + cc_loss_B)) * self.loss_weight.weights_decomp[layer]

            mse_loss_V = 5 * self.loss.ssim(image_vi_Y, image_vi_Y_hat) + self.loss.img_loss(image_vi_Y, image_vi_Y_hat)
            mse_loss_I = 5 * self.loss.ssim(image_ir, image_ir_hat) + self.loss.img_loss(image_ir, image_ir_hat)

            Gradient_loss = self.loss.gradient_loss(image_vi_Y_hat, image_vi_Y)

            loss = (self.loss_weight.coeff_mse_loss_VF * mse_loss_V +
                    self.loss_weight.coeff_mse_loss_IF * mse_loss_I +
                    self.loss_weight.coeff_decomp * loss_decomp +
                    self.loss_weight.coeff_tv * Gradient_loss)

        if self.config.logging.tensorboard.enabled:
            with self.profiling.scope('val/logging'):
                metrics = {
                    'loss/i/mse_vi': mse_loss_V,
                    'loss/i/mse_ir': mse_loss_I,
                    'loss/i/decomp': loss_decomp,
                    'loss/i/grad': Gradient_loss,
                    'loss/i/total': loss
                }

                self.update_metrics(metrics, self.val_metrics, lambda: MeanMetric())

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            collect_params(self.shared_encoder,
                           self.shared_decoder,
                           self.refiner_1,
                           self.fusion_1,
                           self.refiner_2,
                           self.fusion_2),
            lr=self.hparam.lr.initial_lr,
            # weight_decay=0.00001
        )

        scheduler = adapters.get_lr_scheduler(optimizer=optimizer,
                                              opts=self.hparam.lr.scheduler)

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler
        }


class ModelBaseStageII(NewModelBaseTraining):
    def __init__(self):
        super().__init__()

    def training_step(self, batch, batch_idx):
        with self.profiling.scope('train/forward'):
            image_vi, image_ir, label = batch

            image_vi_Y, image_vi_Cb, image_vi_Cr = RGB2YCbCr(image_vi)

            feat_ir_0 = self.shared_encoder(image_ir)
            feat_vi_0 = self.shared_encoder(image_vi_Y)

            fusion_output_0 = self.fusion_0(feat_ir_0, feat_vi_0)

            feat_ir_1, feat_vi_1 = self.refiner_1(feat_ir_0, feat_vi_0)
            fusion_output_1 = self.fusion_1(feat_ir_1, feat_vi_1)

            feat_ir_2, feat_vi_2 = self.refiner_2(torch.cat([self.down_1(feat_ir_0), feat_ir_1], dim=1),
                                                  torch.cat([self.down_1(feat_vi_0), feat_vi_1], dim=1))
            fusion_output_2 = self.fusion_2(feat_ir_2, feat_vi_2)

            image_fused, _ = self.shared_decoder(image_ir, torch.cat([fusion_output_0.fused_feat,
                                                                      self.up_2(fusion_output_1.fused_feat),
                                                                      self.up_4(fusion_output_2.fused_feat)], dim=1))
            base_feats_ir = [fusion_output_0.base_feat_ir,
                             self.up_2(fusion_output_1.base_feat_ir),
                             self.up_4(fusion_output_2.base_feat_ir)]
            base_feat_ir = torch.cat(base_feats_ir, dim=1)
            detail_feats_ir = [fusion_output_0.detail_feat_ir,
                               self.up_2(fusion_output_1.detail_feat_ir),
                               self.up_4(fusion_output_2.detail_feat_ir)]
            detail_feat_ir = torch.cat(detail_feats_ir, dim=1)
            base_feats_vi = [fusion_output_0.base_feat_vi,
                             self.up_2(fusion_output_1.base_feat_vi),
                             self.up_4(fusion_output_2.base_feat_vi)]
            base_feat_vi = torch.cat(base_feats_vi, dim=1)
            detail_feats_vi = [fusion_output_0.detail_feat_vi,
                               self.up_2(fusion_output_1.detail_feat_vi),
                               self.up_4(fusion_output_2.detail_feat_vi)]
            detail_feat_vi = torch.cat(detail_feats_vi, dim=1)

        with self.profiling.scope('train/calc_loss'):

            cc_loss_B = losses.cc(base_feats_vi[0], base_feats_ir[0])
            cc_loss_D = losses.cc(detail_feats_vi[0], detail_feats_ir[0])
            loss_decomp = (cc_loss_D ** 2 / (1.01 + cc_loss_B)) * self.loss_weight.weights_decomp[0]

            for layer in range(1, 3):
                cc_loss_B = losses.cc(base_feats_vi[layer], base_feats_ir[layer])
                cc_loss_D = losses.cc(detail_feats_vi[layer], detail_feats_ir[layer])

                loss_decomp += (cc_loss_D ** 2 / (1.01 + cc_loss_B)) * self.loss_weight.weights_decomp[layer]

            fusionloss, _, _ = self.loss(image_vi_Y, image_ir, image_fused, label)

            loss = fusionloss + self.loss_weight.coeff_decomp * loss_decomp

        iterations = self.iterations + 1
        if self.config.logging.tensorboard.enabled:
            with self.profiling.scope('train/logging'):
                tensorboard: SummaryWriter = self.logger.experiment

                metrics = {
                    'loss/ii/fusion': fusionloss,
                    'loss/ii/decomp': loss_decomp,
                    'loss/ii/total': loss
                }

                for idx, lr in enumerate(self.lr_schedulers().get_last_lr()):
                    metrics[f'hparams/lr/{idx}'] = lr

                for k, v in metrics.items():
                    tensorboard.add_scalar(k, v, iterations)

                self.update_metrics(metrics, self.train_metrics, lambda: MeanMetric())

                for idx, lr in enumerate(self.lr_schedulers().get_last_lr()):
                    tensorboard.add_scalar(f'hparams/lr/{idx}', lr, iterations)

                if iterations % self.config.logging.tensorboard.log_image_freq == 0:
                    with self.profiling.scope('train/logging/image'):
                        with torch.no_grad():
                            tensorboard.add_image('display/ii', make_grid([
                                image_vi_Y[0], image_ir[0],
                                image_fused[0]
                            ], nrow=3), iterations)

        return loss

    def validation_step(self, batch, batch_idx):
        with self.profiling.scope('val/forward'):
            image_vi, image_ir, label = batch

            image_vi_Y, image_vi_Cb, image_vi_Cr = RGB2YCbCr(image_vi)

            feat_ir_0 = self.shared_encoder(image_ir)
            feat_vi_0 = self.shared_encoder(image_vi_Y)

            fusion_output_0 = self.fusion_0(feat_ir_0, feat_vi_0)

            feat_ir_1, feat_vi_1 = self.refiner_1(feat_ir_0, feat_vi_0)
            fusion_output_1 = self.fusion_1(feat_ir_1, feat_vi_1)

            feat_ir_2, feat_vi_2 = self.refiner_2(torch.cat([self.down_1(feat_ir_0), feat_ir_1], dim=1),
                                                  torch.cat([self.down_1(feat_vi_0), feat_vi_1], dim=1))
            fusion_output_2 = self.fusion_2(feat_ir_2, feat_vi_2)

            image_fused, _ = self.shared_decoder(image_ir, torch.cat([fusion_output_0.fused_feat,
                                                                      self.up_2(fusion_output_1.fused_feat),
                                                                      self.up_4(fusion_output_2.fused_feat)], dim=1))

            base_feats_ir = [fusion_output_0.base_feat_ir,
                             self.up_2(fusion_output_1.base_feat_ir),
                             self.up_4(fusion_output_2.base_feat_ir)]
            base_feat_ir = torch.cat(base_feats_ir, dim=1)
            detail_feats_ir = [fusion_output_0.detail_feat_ir,
                               self.up_2(fusion_output_1.detail_feat_ir),
                               self.up_4(fusion_output_2.detail_feat_ir)]
            detail_feat_ir = torch.cat(detail_feats_ir, dim=1)
            base_feats_vi = [fusion_output_0.base_feat_vi,
                             self.up_2(fusion_output_1.base_feat_vi),
                             self.up_4(fusion_output_2.base_feat_vi)]
            base_feat_vi = torch.cat(base_feats_vi, dim=1)
            detail_feats_vi = [fusion_output_0.detail_feat_vi,
                               self.up_2(fusion_output_1.detail_feat_vi),
                               self.up_4(fusion_output_2.detail_feat_vi)]
            detail_feat_vi = torch.cat(detail_feats_vi, dim=1)

        with self.profiling.scope('val/calc_loss'):
            cc_loss_B = losses.cc(base_feats_vi[0], base_feats_ir[0])
            cc_loss_D = losses.cc(detail_feats_vi[0], detail_feats_ir[0])
            loss_decomp = (cc_loss_D ** 2 / (1.01 + cc_loss_B)) * self.loss_weight.weights_decomp[0]

            for layer in range(1, 3):
                cc_loss_B = losses.cc(base_feats_vi[layer], base_feats_ir[layer])
                cc_loss_D = losses.cc(detail_feats_vi[layer], detail_feats_ir[layer])

                loss_decomp += (cc_loss_D ** 2 / (1.01 + cc_loss_B)) * self.loss_weight.weights_decomp[layer]

            fusionloss, _, _ = self.loss(image_vi_Y, image_ir, image_fused, label)

            loss = fusionloss + self.loss_weight.coeff_decomp * loss_decomp

        if self.config.logging.tensorboard.enabled:
            with self.profiling.scope('val/logging'):
                metrics = {
                    'loss/ii/fusion': fusionloss,
                    'loss/ii/decomp': loss_decomp,
                    'loss/ii/total': loss
                }

                self.update_metrics(metrics, self.val_metrics, lambda: MeanMetric())

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            collect_params(self.shared_encoder,
                           self.shared_decoder,
                           self.refiner_1,
                           self.fusion_1,
                           self.refiner_2,
                           self.fusion_2),
            lr=self.hparam.lr.initial_lr,
            # weight_decay=0.00001
        )

        scheduler = adapters.get_lr_scheduler(optimizer=optimizer,
                                              opts=self.hparam.lr.scheduler)

        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler
        }


class ModuleHook(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, module, args, kwargs, output):
        pass


class ModelBaseStageIII(ModelBaseTraining):
    class Modes(Enum):
        FUSION = 1
        META = 2
        SEG = 3

    def __init__(self):
        super().__init__()

        self.train_meta = False
        self.automatic_optimization = False
        # self.fusion_0.register_forward_hook(ModuleHook(), with_kwargs=True)
        self.mfe_0 = MetaFeatureExtractor((32, 32), 32, 32)
        self.mfe_1 = MetaFeatureExtractor((32, 64), 32, 32)
        self.mfe_2 = MetaFeatureExtractor((32, 160), 32, 32)

        self.ft_0 = FeatureTransform(32, 32)
        self.ft_1 = FeatureTransform(32, 32)
        self.ft_2 = FeatureTransform(32, 32)

        self.model = NewModelBase()
        self.segmodel: WrappedSegmentor = None

        self.loss = FusionLoss()

    def forward(self, model, image_vi, image_ir, label, mode: Modes = Modes.FUSION):
        image_vi = model.convert(image_vi)
        image_ir = model.convert(image_ir)
        label = model.convert(label, dtype=False)
        image_vi_Y, image_vi_Cb, image_vi_Cr = RGB2YCbCr(image_vi)

        feat_vi_0 = model.shared_encoder(image_vi_Y)
        feat_ir_0 = model.shared_encoder(image_ir)

        fusion_output_0 = model.fusion_0(feat_ir_0, feat_vi_0)

        feat_ir_1, feat_vi_1 = model.refiner_1(feat_ir_0, feat_vi_0)
        fusion_output_1 = model.fusion_1(feat_ir_1, feat_vi_1)

        feat_ir_2, feat_vi_2 = model.refiner_2(torch.cat([model.down_1(feat_ir_0), feat_ir_1], dim=1),
                                               torch.cat([model.down_1(feat_vi_0), feat_vi_1], dim=1))
        fusion_output_2 = model.fusion_2(feat_ir_2, feat_vi_2)

        image_fused_Y, _ = model.shared_decoder(image_ir, torch.cat([fusion_output_0.fused_feat,
                                                                  model.up_2(fusion_output_1.fused_feat),
                                                                  model.up_4(fusion_output_2.fused_feat)], dim=1))

        base_feats_ir = [fusion_output_0.base_feat_ir,
                         model.up_2(fusion_output_1.base_feat_ir),
                         model.up_4(fusion_output_2.base_feat_ir)]
        base_feat_ir = torch.cat(base_feats_ir, dim=1)
        detail_feats_ir = [fusion_output_0.detail_feat_ir,
                           model.up_2(fusion_output_1.detail_feat_ir),
                           model.up_4(fusion_output_2.detail_feat_ir)]
        detail_feat_ir = torch.cat(detail_feats_ir, dim=1)
        base_feats_vi = [fusion_output_0.base_feat_vi,
                         model.up_2(fusion_output_1.base_feat_vi),
                         model.up_4(fusion_output_2.base_feat_vi)]
        base_feat_vi = torch.cat(base_feats_vi, dim=1)
        detail_feats_vi = [fusion_output_0.detail_feat_vi,
                           model.up_2(fusion_output_1.detail_feat_vi),
                           model.up_4(fusion_output_2.detail_feat_vi)]
        detail_feat_vi = torch.cat(detail_feats_vi, dim=1)

        cc_loss_B = losses.cc(base_feats_vi[0], base_feats_ir[0])
        cc_loss_D = losses.cc(detail_feats_vi[0], detail_feats_ir[0])
        loss_decomp = (cc_loss_D ** 2 / (1.01 + cc_loss_B)) * self.loss_weight.weights_decomp[0]

        for layer in range(1, 3):
            cc_loss_B = losses.cc(base_feats_vi[layer], base_feats_ir[layer])
            cc_loss_D = losses.cc(detail_feats_vi[layer], detail_feats_ir[layer])

            loss_decomp += (cc_loss_D ** 2 / (1.01 + cc_loss_B)) * self.loss_weight.weights_decomp[layer]

        fusionloss, _, _ = self.loss(image_vi_Y, image_ir, image_fused_Y, label)

        loss_fusion = fusionloss + self.loss_weight.coeff_decomp * loss_decomp

        if mode == ModelBaseStageIII.Modes.FUSION:
            return image_fused_Y, loss_fusion

        elif mode == ModelBaseStageIII.Modes.META:
            image_fused = YCbCr2RGB(image_fused_Y, image_vi_Cb, image_vi_Cr)

            with torch.no_grad():
                seg_feats: list = self.segmodel.segmentor.extract_feat(image_fused)

            feat_t_0 = self.ft_0(fusion_output_0.detail_feat)
            feat_t_1 = self.ft_1(fusion_output_1.detail_feat)
            feat_t_2 = self.ft_2(fusion_output_2.detail_feat)

            feat_meta_0 = self.mfe_0(fusion_output_0.detail_feat, seg_feats[0])
            feat_meta_1 = self.mfe_1(fusion_output_1.detail_feat, seg_feats[1])
            feat_meta_2 = self.mfe_2(fusion_output_2.detail_feat, seg_feats[2])

            return image_fused_Y, (feat_t_0, feat_t_1, feat_t_2), (feat_meta_0, feat_meta_1, feat_meta_2), loss_fusion

    def _init_with_config(self, config: Config):
        super()._init_with_config(config)

        self.segmodel = WrappedSegmentor(mmseg.init_model(config.semseg.config,
                                                          config.semseg['model'],
                                                          self.device()),  # should be fine with this device
                                         config.semseg.input.mean,
                                         config.semseg.input.std)

    def train_fusion(self, model, image_vi, image_ir, label):
        pass

    def training_step(self, batch, batch_idx):
        data: MetaDataModule = self.trainer.datamodule

        if (self.epochs) % self.hparam.meta.interval == 0:
            if not self.train_meta:
                self.logging.logger.info('train', f'conducting meta learning inner update at {self.epochs} epochs')
                self.train_meta = True

                model = deepcopy(self.model)
                name_to_param = dict(model.named_parameters())

                optim_f = torch.optim.SGD(model.parameters(), lr=self.hparam.lr.inner.initial_lr)
                optim_meta = torch.optim.SGD(collect_params(self.mfe_0,
                                                            self.mfe_1,
                                                            self.mfe_2,
                                                            self.ft_0,
                                                            self.ft_1,
                                                            self.ft_2), lr=self.hparam.lr.inner.initial_lr)

                m_train, m_test = data.get_meta_train_datasets()
                for train_batch, test_batch in tqdm(zip(m_train, m_test)):
                    optim_f.zero_grad()
                    optim_meta.zero_grad()

                    image_fused_Y, feats_t, feats_meta, loss_fusion = self.forward(self.model, *train_batch,
                                                                                   ModelBaseStageIII.Modes.META)
                    loss_meta = sum(
                        [self.loss_weight.meta[i] * F.mse_loss(feat_meta, feat_t)
                         for i, (feat_meta, feat_t) in enumerate(zip(feats_meta, feats_t))])

                    self.manual_backward(loss_meta, retain_graph=True)

                    for name, param in self.model.named_parameters():
                        cur_grad = param.grad
                        if cur_grad is not None:
                            if name_to_param[name].grad is None:
                                name_to_param[name].grad = Variable(torch.zeros(cur_grad.size()).to(param))
                            name_to_param[name].grad.data.add_(cur_grad)

                    self.clip_gradients(optim_f, 0.01, 'norm')
                    optim_f.step()

                    image_fused_Y, loss_fusion = self.forward(model, *train_batch, ModelBaseStageIII.Modes.FUSION)

                    self.manual_backward(loss_fusion)

                    self.clip_gradients(optim_meta, 0.01, 'norm')
                    optim_meta.step()
        else:
            self.train_meta = False

        image_vi, image_ir, label = batch

        opt = self.optimizers()
        opt.zero_grad()

        self.model = self.convert(self.model)
        self.model.train()

        image_fused_Y, feats_t, feats_meta, loss_fusion = self.forward(self.model, image_vi, image_ir, label,
                                                                       ModelBaseStageIII.Modes.META)

        loss_meta = sum(
            [self.loss_weight.meta[i] * F.mse_loss(feat_meta, feat_t)
             for i, (feat_meta, feat_t) in enumerate(zip(feats_meta, feats_t))])

        loss = loss_fusion + loss_meta

        self.manual_backward(loss)
        self.clip_gradients(opt, 0.01, 'norm')
        opt.step()


        iterations = self.iterations + 1
        if self.config.logging.tensorboard.enabled:
            with self.profiling.scope('train/logging'):
                tensorboard: SummaryWriter = self.logger.experiment

                metrics = {
                    'loss/iii/fusion': loss_fusion,
                    'loss/iii/meta': loss_meta,
                    'loss/iii/total': loss
                }

                for idx, lr in enumerate(self.lr_schedulers().get_last_lr()):
                    metrics[f'hparams/lr/{idx}'] = lr

                for k, v in metrics.items():
                    tensorboard.add_scalar(k, v, iterations)

                self.update_metrics(metrics, self.train_metrics, lambda: MeanMetric())

                for idx, lr in enumerate(self.lr_schedulers().get_last_lr()):
                    tensorboard.add_scalar(f'hparams/lr/{idx}', lr, iterations)

                if iterations % self.config.logging.tensorboard.log_image_freq == 0:
                    with self.profiling.scope('train/logging/image'):
                        with torch.no_grad():
                            tensorboard.add_image('display/iii', make_grid([
                                rgb_to_grayscale(image_vi[0:1])[0], image_ir[0],
                                image_fused_Y[0]
                            ], nrow=3), iterations)

    def load_state_dict(self, state_dict,
                        strict=False):
        super().load_state_dict(state_dict, False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            collect_params(self.model),
            lr=self.hparam.lr.outer.initial_lr,
            # weight_decay=0.00001
        )

        scheduler = adapters.get_lr_scheduler(optimizer=optimizer,
                                              opts=self.hparam.lr.outer.scheduler)

        optimizer_outer = {
            'optimizer': optimizer,
            'lr_scheduler': scheduler
        }

        return optimizer_outer
