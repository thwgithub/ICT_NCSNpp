# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 16:13:04 2024

@author: Lenovo
"""

'''
firstly install the following two packages
pip install -q lightning gdown torchmetrics einops torchinfo --no-cache --upgrade
pip install -q -e git+https://github.com/Kinyugo/consistency_models.git#egg=consistency_models
and then reboot the pycharm, and it works
'''
import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Callable, List, Optional, Tuple, Union

import torch
from torchvision import datasets as dsets
from einops import rearrange
from einops.layers.torch import Rearrange
from lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from matplotlib import pyplot as plt
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchinfo import summary
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid

from consistency_models import (
    ConsistencySamplingAndEditing,
    ImprovedConsistencyTraining,
    pseudo_huber_loss,
)
from consistency_models.utils import update_ema_model_

########################################loading dataset#################################################################
@dataclass
class ImageDataModuleConfig:
    data_dir: str = "/home/common/hwtan/data/data_cifar10"
    image_size: Tuple[int, int] = (32, 32)
    batch_size: int = 64
    num_workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = True


class ImageDataModule(LightningDataModule):
    def __init__(self, config: ImageDataModuleConfig) -> None:
        super().__init__()

        self.config = config

    def setup(self, stage: str = None) -> None:
        transform = T.Compose(
            [
                T.Resize(self.config.image_size),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Lambda(lambda x: (x * 2) - 1),
            ]
        )
        #self.dataset = ImageFolder(self.config.data_dir, transform=transform)#if the dataset is the non-pytorch dataset,and execute it
        self.dataset = dsets.CIFAR10(self.config.data_dir, transform=transform)
        '''
        dataloader = torch.utils.data.DataLoader(traindata,
                                             batch_size=32,
                                             shuffle=True, drop_last=True, num_workers=8, pin_memory=True,
                                             persistent_workers=True)
        '''
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers,
        )
########################################################################################################################

####################################################UNetModelpara_START#################################################
@dataclass
class UNetConfig:
    attention_type: str = "ddpm"
    attn_resolutions: Tuple[int, ...] = (16,)
    beta_max: float = 20.0
    beta_min: float = 0.1
    centered: bool = False
    ch_mult: Tuple[int, ...] = (2, 2, 2)
    conditional: bool = True
    continuous = True
    conv_size: int = 3
    dropout: float = 0.3
    ema_rate: float = 0.99993
    embedding_type: str = 'fourier'
    fir: bool = True
    fir_kernel = [1, 3, 3, 1]
    fourier_scale: int = 16
    image_size: int = 32
    init_scale: float = 0.0
    name: str = 'ncsnpp'
    nf: int = 128
    nonlinearity: str = 'swish'
    normalization: str = 'GroupNorm'
    num_channels: int = 3
    num_res_blocks: int = 4 #set 8 if deepen network
    num_scales: int = 18
    progressive: str = 'none'
    progressive_combine: str = 'sum'
    progressive_input: str = 'residual'
    resamp_with_conv: bool = True
    resblock_type: str = 'biggan'
    scale_by_sigma: bool = True
    sigma_max: float = 100.0
    sigma_min: float = 0.02
    skip_rescale: bool = True
################################################UNetModelpara_END#######################################################

###############################################UnetModel_START##########################################################
from sdemodelspp_diy import utils, layers, layerspp, normalization
import torch.nn as nn
import functools
import torch
import numpy as np

ResnetBlockDDPM = layerspp.ResnetBlockDDPMpp
ResnetBlockBigGAN = layerspp.ResnetBlockBigGANpp
Combine = layerspp.Combine
conv3x3 = layerspp.conv3x3
conv1x1 = layerspp.conv1x1
get_act = layers.get_act
get_normalization = normalization.get_normalization
default_initializer = layers.default_init

@utils.register_model(name='ncsnpp')
class UNet(nn.Module):
  """NCSN++ model"""

  def __init__(self, config):
    super().__init__()
    self.config = config
    self.act = act = nn.SiLU()
    self.register_buffer('sigmas', torch.tensor(utils.get_sigmas(config)))

    self.nf = nf = config.nf
    ch_mult = config.ch_mult
    self.num_res_blocks = num_res_blocks = config.num_res_blocks
    self.attn_resolutions = attn_resolutions = config.attn_resolutions
    dropout = config.dropout
    resamp_with_conv = config.resamp_with_conv
    self.num_resolutions = num_resolutions = len(ch_mult)
    self.all_resolutions = all_resolutions = [config.image_size // (2 ** i) for i in range(num_resolutions)]

    self.conditional = conditional = config.conditional  # noise-conditional
    fir = config.fir
    fir_kernel = config.fir_kernel
    self.skip_rescale = skip_rescale = config.skip_rescale
    self.resblock_type = resblock_type = config.resblock_type
    self.progressive = progressive = config.progressive
    self.progressive_input = progressive_input = config.progressive_input
    self.embedding_type = embedding_type = config.embedding_type
    init_scale = config.init_scale
    assert progressive in ['none', 'output_skip', 'residual']
    assert progressive_input in ['none', 'input_skip', 'residual']
    assert embedding_type in ['fourier', 'positional']
    combine_method = config.progressive_combine
    combiner = functools.partial(Combine, method=combine_method)

    modules = []
    # timestep/noise_level embedding; only for continuous training
    if embedding_type == 'fourier':
      # Gaussian Fourier features embeddings.
      assert config.continuous, "Fourier features are only used for continuous training."

      modules.append(layerspp.GaussianFourierProjection(
        embedding_size=nf, scale=config.fourier_scale
      ))
      embed_dim = 2 * nf

    elif embedding_type == 'positional':
      embed_dim = nf

    else:
      raise ValueError(f'embedding type {embedding_type} unknown.')

    if conditional:
      modules.append(nn.Linear(embed_dim, nf * 4))
      modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
      nn.init.zeros_(modules[-1].bias)
      modules.append(nn.Linear(nf * 4, nf * 4))
      modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
      nn.init.zeros_(modules[-1].bias)

    AttnBlock = functools.partial(layerspp.AttnBlockpp,
                                  init_scale=init_scale,
                                  skip_rescale=skip_rescale)

    Upsample = functools.partial(layerspp.Upsample,
                                 with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)

    if progressive == 'output_skip':
      self.pyramid_upsample = layerspp.Upsample(fir=fir, fir_kernel=fir_kernel, with_conv=False)
    elif progressive == 'residual':
      pyramid_upsample = functools.partial(layerspp.Upsample,
                                           fir=fir, fir_kernel=fir_kernel, with_conv=True)

    Downsample = functools.partial(layerspp.Downsample,
                                   with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)

    if progressive_input == 'input_skip':
      self.pyramid_downsample = layerspp.Downsample(fir=fir, fir_kernel=fir_kernel, with_conv=False)
    elif progressive_input == 'residual':
      pyramid_downsample = functools.partial(layerspp.Downsample,
                                             fir=fir, fir_kernel=fir_kernel, with_conv=True)

    if resblock_type == 'ddpm':
      ResnetBlock = functools.partial(ResnetBlockDDPM,
                                      act=act,
                                      dropout=dropout,
                                      init_scale=init_scale,
                                      skip_rescale=skip_rescale,
                                      temb_dim=nf * 4)

    elif resblock_type == 'biggan':
      ResnetBlock = functools.partial(ResnetBlockBigGAN,
                                      act=act,
                                      dropout=dropout,
                                      fir=fir,
                                      fir_kernel=fir_kernel,
                                      init_scale=init_scale,
                                      skip_rescale=skip_rescale,
                                      temb_dim=nf * 4)

    else:
      raise ValueError(f'resblock type {resblock_type} unrecognized.')

    # Downsampling block

    channels = config.num_channels
    if progressive_input != 'none':
      input_pyramid_ch = channels

    modules.append(conv3x3(channels, nf))
    hs_c = [nf]

    in_ch = nf
    for i_level in range(num_resolutions):
      # Residual blocks for this resolution
      for i_block in range(num_res_blocks):
        out_ch = nf * ch_mult[i_level]
        modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
        in_ch = out_ch

        if all_resolutions[i_level] in attn_resolutions:
          modules.append(AttnBlock(channels=in_ch))
        hs_c.append(in_ch)

      if i_level != num_resolutions - 1:
        if resblock_type == 'ddpm':
          modules.append(Downsample(in_ch=in_ch))
        else:
          modules.append(ResnetBlock(down=True, in_ch=in_ch))

        if progressive_input == 'input_skip':
          modules.append(combiner(dim1=input_pyramid_ch, dim2=in_ch))
          if combine_method == 'cat':
            in_ch *= 2

        elif progressive_input == 'residual':
          modules.append(pyramid_downsample(in_ch=input_pyramid_ch, out_ch=in_ch))
          input_pyramid_ch = in_ch

        hs_c.append(in_ch)

    in_ch = hs_c[-1]
    modules.append(ResnetBlock(in_ch=in_ch))
    modules.append(AttnBlock(channels=in_ch))
    modules.append(ResnetBlock(in_ch=in_ch))

    pyramid_ch = 0
    # Upsampling block
    for i_level in reversed(range(num_resolutions)):
      for i_block in range(num_res_blocks + 1):
        out_ch = nf * ch_mult[i_level]
        modules.append(ResnetBlock(in_ch=in_ch + hs_c.pop(),
                                   out_ch=out_ch))
        in_ch = out_ch

      if all_resolutions[i_level] in attn_resolutions:
        modules.append(AttnBlock(channels=in_ch))

      if progressive != 'none':
        if i_level == num_resolutions - 1:
          if progressive == 'output_skip':
            modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                        num_channels=in_ch, eps=1e-6))
            modules.append(conv3x3(in_ch, channels, init_scale=init_scale))
            pyramid_ch = channels
          elif progressive == 'residual':
            modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                        num_channels=in_ch, eps=1e-6))
            modules.append(conv3x3(in_ch, in_ch, bias=True))
            pyramid_ch = in_ch
          else:
            raise ValueError(f'{progressive} is not a valid name.')
        else:
          if progressive == 'output_skip':
            modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                        num_channels=in_ch, eps=1e-6))
            modules.append(conv3x3(in_ch, channels, bias=True, init_scale=init_scale))
            pyramid_ch = channels
          elif progressive == 'residual':
            modules.append(pyramid_upsample(in_ch=pyramid_ch, out_ch=in_ch))
            pyramid_ch = in_ch
          else:
            raise ValueError(f'{progressive} is not a valid name')

      if i_level != 0:
        if resblock_type == 'ddpm':
          modules.append(Upsample(in_ch=in_ch))
        else:
          modules.append(ResnetBlock(in_ch=in_ch, up=True))

    assert not hs_c

    if progressive != 'output_skip':
      modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32),
                                  num_channels=in_ch, eps=1e-6))
      modules.append(conv3x3(in_ch, channels, init_scale=init_scale))

    self.all_modules = nn.ModuleList(modules)

  def forward(self, x, time_cond):
    # timestep/noise_level embedding; only for continuous training
    modules = self.all_modules
    m_idx = 0
    if self.embedding_type == 'fourier':
      # Gaussian Fourier features embeddings.
      used_sigmas = time_cond
      temb = modules[m_idx](torch.log(used_sigmas))
      m_idx += 1

    elif self.embedding_type == 'positional':
      # Sinusoidal positional embeddings.
      timesteps = time_cond
      used_sigmas = self.sigmas[time_cond.long()]
      temb = layers.get_timestep_embedding(timesteps, self.nf)

    else:
      raise ValueError(f'embedding type {self.embedding_type} unknown.')

    if self.conditional:
      temb = modules[m_idx](temb)
      m_idx += 1
      temb = modules[m_idx](self.act(temb))
      m_idx += 1
    else:
      temb = None

    if self.config.centered:
      # If input data is in [0, 1]
      x = 2 * x - 1.

    # Downsampling block
    input_pyramid = None
    if self.progressive_input != 'none':
      input_pyramid = x

    hs = [modules[m_idx](x)]
    m_idx += 1
    for i_level in range(self.num_resolutions):
      # Residual blocks for this resolution
      for i_block in range(self.num_res_blocks):
        h = modules[m_idx](hs[-1], temb)
        m_idx += 1
        if h.shape[-1] in self.attn_resolutions:
          h = modules[m_idx](h)
          m_idx += 1

        hs.append(h)

      if i_level != self.num_resolutions - 1:
        if self.resblock_type == 'ddpm':
          h = modules[m_idx](hs[-1])
          m_idx += 1
        else:
          h = modules[m_idx](hs[-1], temb)
          m_idx += 1

        if self.progressive_input == 'input_skip':
          input_pyramid = self.pyramid_downsample(input_pyramid)
          h = modules[m_idx](input_pyramid, h)
          m_idx += 1

        elif self.progressive_input == 'residual':
          input_pyramid = modules[m_idx](input_pyramid)
          m_idx += 1
          if self.skip_rescale:
            input_pyramid = (input_pyramid + h) / np.sqrt(2.)
          else:
            input_pyramid = input_pyramid + h
          h = input_pyramid

        hs.append(h)

    h = hs[-1]
    h = modules[m_idx](h, temb)
    m_idx += 1
    h = modules[m_idx](h)
    m_idx += 1
    h = modules[m_idx](h, temb)
    m_idx += 1

    pyramid = None

    # Upsampling block
    for i_level in reversed(range(self.num_resolutions)):
      for i_block in range(self.num_res_blocks + 1):
        h = modules[m_idx](torch.cat([h, hs.pop()], dim=1), temb)
        m_idx += 1

      if h.shape[-1] in self.attn_resolutions:
        h = modules[m_idx](h)
        m_idx += 1

      if self.progressive != 'none':
        if i_level == self.num_resolutions - 1:
          if self.progressive == 'output_skip':
            pyramid = self.act(modules[m_idx](h))
            m_idx += 1
            pyramid = modules[m_idx](pyramid)
            m_idx += 1
          elif self.progressive == 'residual':
            pyramid = self.act(modules[m_idx](h))
            m_idx += 1
            pyramid = modules[m_idx](pyramid)
            m_idx += 1
          else:
            raise ValueError(f'{self.progressive} is not a valid name.')
        else:
          if self.progressive == 'output_skip':
            pyramid = self.pyramid_upsample(pyramid)
            pyramid_h = self.act(modules[m_idx](h))
            m_idx += 1
            pyramid_h = modules[m_idx](pyramid_h)
            m_idx += 1
            pyramid = pyramid + pyramid_h
          elif self.progressive == 'residual':
            pyramid = modules[m_idx](pyramid)
            m_idx += 1
            if self.skip_rescale:
              pyramid = (pyramid + h) / np.sqrt(2.)
            else:
              pyramid = pyramid + h
            h = pyramid
          else:
            raise ValueError(f'{self.progressive} is not a valid name')

      if i_level != 0:
        if self.resblock_type == 'ddpm':
          h = modules[m_idx](h)
          m_idx += 1
        else:
          h = modules[m_idx](h, temb)
          m_idx += 1

    assert not hs

    if self.progressive == 'output_skip':
      h = pyramid
    else:
      h = self.act(modules[m_idx](h))
      m_idx += 1
      h = modules[m_idx](h)
      m_idx += 1

    assert m_idx == len(modules)
    if self.config.scale_by_sigma:
      used_sigmas = used_sigmas.reshape((x.shape[0], *([1] * len(x.shape[1:]))))
      h = h / used_sigmas

    return h

  def save_pretrained(self, pretrained_path: str) -> None:
      os.makedirs(pretrained_path, exist_ok=True)

      with open(os.path.join(pretrained_path, "config.json"), mode="w") as f:
          json.dump(asdict(self.config), f)

      torch.save(self.state_dict(), os.path.join(pretrained_path, "model.pt"))

  @classmethod
  def from_pretrained(cls, pretrained_path: str) -> "UNet":
      with open(os.path.join(pretrained_path, "config.json"), mode="r") as f:
          config_dict = json.load(f)
      config = UNetConfig(**config_dict)

      model = cls(config)

      state_dict = torch.load(
          os.path.join(pretrained_path, "model.pt"), map_location=torch.device("cpu")
      )
      model.load_state_dict(state_dict)

      return model
###############################################UnetModel_END############################################################
#summary(UNet(UNetConfig()), input_size=((1, 3, 32, 32), (1,)))


##LitUnet

@dataclass
class LitImprovedConsistencyModelConfig:
    ema_decay_rate: float = 0.9999
    lr: float = 1e-4
    betas: Tuple[float, float] = (0.9, 0.999)
    lr_scheduler_start_factor: float = 1e-5
    lr_scheduler_iters: int = 10_000
    sample_every_n_steps: int = 10_000
    num_samples: int = 8
    sampling_sigmas: Tuple[Tuple[int, ...], ...] = (
        (80,),
        (80.0, 0.661),
        (80.0, 24.4, 5.84, 0.9, 0.661),
    )

class LitImprovedConsistencyModel(LightningModule):
    def __init__(
        self,
        consistency_training: ImprovedConsistencyTraining,
        consistency_sampling: ConsistencySamplingAndEditing,
        model: UNet,
        ema_model: UNet,
        config: LitImprovedConsistencyModelConfig,
    ) -> None:
        super().__init__()

        self.consistency_training = consistency_training
        self.consistency_sampling = consistency_sampling
        self.model = model
        self.ema_model = ema_model
        self.config = config

        # Freeze the EMA model and set it to eval mode
        for param in self.ema_model.parameters():
            param.requires_grad = False
        self.ema_model = self.ema_model.eval()

    def training_step(self, batch: Union[Tensor, List[Tensor]], batch_idx: int) -> None:
        if isinstance(batch, list):
            batch = batch[0]

        output = self.consistency_training(
            self.model, batch, self.global_step, self.trainer.max_steps
        )

        loss = (
            pseudo_huber_loss(output.predicted, output.target) * output.loss_weights
        ).mean()

        self.log_dict({"train_loss": loss, "num_timesteps": output.num_timesteps})

        return loss

    def on_train_batch_end(
        self, outputs: Any, batch: Union[Tensor, List[Tensor]], batch_idx: int
    ) -> None:
        update_ema_model_(self.model, self.ema_model, self.config.ema_decay_rate)

        if (
            (self.global_step + 1) % self.config.sample_every_n_steps == 0
        ) or self.global_step == 0:
            self.__sample_and_log_samples(batch)

    def configure_optimizers(self):
        opt = torch.optim.RAdam(
            self.model.parameters(), lr=self.config.lr, betas=self.config.betas
        )
        sched = torch.optim.lr_scheduler.LinearLR(
            opt,
            start_factor=self.config.lr_scheduler_start_factor,
            total_iters=self.config.lr_scheduler_iters,
        )
        sched = {"scheduler": sched, "interval": "step", "frequency": 1}

        return [opt], [sched]

    @torch.no_grad()
    def __sample_and_log_samples(self, batch: Union[Tensor, List[Tensor]]) -> None:
        if isinstance(batch, list):
            batch = batch[0]

        # Ensure the number of samples does not exceed the batch size
        num_samples = min(self.config.num_samples, batch.shape[0])
        noise = torch.randn_like(batch[:num_samples])

        # Log ground truth samples
        self.__log_images(
            batch[:num_samples].detach().clone(), f"ground_truth", self.global_step
        )

        for sigmas in self.config.sampling_sigmas:
            samples = self.consistency_sampling(
                self.ema_model, noise, sigmas, clip_denoised=True, verbose=True
            )
            samples = samples.clamp(min=-1.0, max=1.0)

            # Generated samples
            self.__log_images(
                samples,
                f"generated_samples-sigmas={sigmas}",
                self.global_step,
            )

    @torch.no_grad()
    def __log_images(self, images: Tensor, title: str, global_step: int) -> None:
        images = images.detach().float()

        grid = make_grid(
            images.clamp(-1.0, 1.0), value_range=(-1.0, 1.0), normalize=True
        )
        self.logger.experiment.add_image(title, grid, global_step)

###training loop
@dataclass
class TrainingConfig:
    image_dm_config: ImageDataModuleConfig
    unet_config: UNetConfig
    consistency_training: ImprovedConsistencyTraining
    consistency_sampling: ConsistencySamplingAndEditing
    lit_icm_config: LitImprovedConsistencyModelConfig
    trainer: Trainer
    seed: int = 42
    model_ckpt_path: str = "/home/hwtan/projects/ICT/ICTcheckpoints/icm_ncsn++_cifar10_64_100ktest"
    resume_ckpt_path: Optional[str] = None


def run_training(config: TrainingConfig) -> None:
    # Set seed
    seed_everything(config.seed)

    # Create data module
    dm = ImageDataModule(config.image_dm_config)

    # Create model and its EMA
    model = UNet(config.unet_config)
    ema_model = UNet(config.unet_config)
    ema_model.load_state_dict(model.state_dict())

    # Create lightning module
    lit_icm = LitImprovedConsistencyModel(
        config.consistency_training,
        config.consistency_sampling,
        model,
        ema_model,
        config.lit_icm_config,
    )

    # Run training
    config.trainer.fit(lit_icm, dm, ckpt_path=config.resume_ckpt_path)
    # Save model
    lit_icm.model.save_pretrained(config.model_ckpt_path)

# run training
training_config = TrainingConfig(
    image_dm_config=ImageDataModuleConfig(),
    unet_config=UNetConfig(),
    consistency_training=ImprovedConsistencyTraining(final_timesteps=11),
    consistency_sampling=ConsistencySamplingAndEditing(),
    lit_icm_config=LitImprovedConsistencyModelConfig(
        sample_every_n_steps=1000, lr_scheduler_iters=1000
    ),
    trainer=Trainer(
        devices=[1],# use one GPU
        max_steps=100000,
        precision="32-true",
        log_every_n_steps=10,
        logger=TensorBoardLogger(".", name="/home/hwtan/projects/ICT/ICTlogs/logs_ncsn++_cifar10_64_100ktest", version="icm"),
        callbacks=[LearningRateMonitor(logging_interval="step")],
    ),
)
import time
start_time = time.time()
run_training(training_config)
print("Generator Trains %d times with total time: %.2fh" % (100000, (time.time() - start_time)/3600))

# sampling and zero-shot editing
seed_everything(42)

'''
# utils
def plot_images(images: Tensor, cols: int = 4) -> None:
    rows = max(images.shape[0] // cols, 1)
    fig, axs = plt.subplots(rows, cols)
    axs = axs.flatten()
    for i, image in enumerate(images):
        axs[i].imshow(image.permute(1, 2, 0).numpy() / 2 + 0.5)
        axs[i].set_axis_off()


# checkpoint loading
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float32 #torch.bfloat16 if torch.cuda.is_available() else torch.float32
unet = UNet.from_pretrained("/home/hwtan/projects/ICT/ICTcheckpoints/icm_ncsn++_cifar10_64_100ktest").eval().to(device=device, dtype=dtype)

# Load Sample Batch
dm = ImageDataModule(ImageDataModuleConfig(batch_size=4))
dm.setup()
batch, _ = next(iter(dm.train_dataloader()))
batch = batch.to(device=device, dtype=dtype)

plot_images(batch.float().cpu())

# Experiments
# sampling
consistency_sampling_and_editing = ConsistencySamplingAndEditing()

with torch.no_grad():
    samples = consistency_sampling_and_editing(
        unet,
        torch.randn((4, 3, 32, 32), device=device, dtype=dtype),
        sigmas=[80.0, 24.4, 5.84, 0.9, 0.661],  # Use more steps for better samples e.g 2-5
        clip_denoised=True,
        verbose=True,
    )
plot_images(samples.float().cpu())

# inpainting
random_erasing = T.RandomErasing(p=1.0, scale=(0.2, 0.5), ratio=(0.5, 0.5))
masked_batch = random_erasing(batch)
mask = torch.logical_not(batch == masked_batch)

plot_images(masked_batch.float().cpu())

with torch.no_grad():
    inpainted_batch = consistency_sampling_and_editing(
        unet,
        masked_batch,
        sigmas=[80.0, 24.4, 5.84, 0.9, 0.661],
        mask=mask.to(dtype=dtype),
        clip_denoised=True,
        verbose=True,
    )

plot_images(torch.cat((masked_batch, inpainted_batch), dim=0).float().cpu())

# Interpolation
batch_a = batch.clone()
batch_b = torch.flip(batch, dims=(0,))

plot_images(torch.cat((batch_a, batch_b), dim=0).float().cpu())

with torch.no_grad():
    interpolated_batch = consistency_sampling_and_editing.interpolate(
        unet,
        batch_a,
        batch_b,
        ab_ratio=0.5,
        sigmas=[80.0, 24.4, 5.84, 0.9, 0.661],
        clip_denoised=True,
        verbose=True,
    )

plot_images(torch.cat((batch_a, batch_b, interpolated_batch), dim=0).float().cpu())
'''