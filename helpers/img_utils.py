#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/02/02 

import sys
from PIL import Image, ImageFilter
from PIL.Image import Image as PILImage
from typing import Tuple, List, Union

from torch import Tensor
import torch
import torch.nn as nn
from torchvision.utils import make_grid
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt

dtype = np.float32
npimg = NDArray[dtype]
eps   = np.finfo(dtype).eps


''' decrators '''
def valid_img(fn):
  def wrapper(img:PILImage, *args, **kwargs):
    assert isinstance(img, PILImage)
    assert img.mode in ['RGB', 'L']

    return fn(img, *args, **kwargs)
  return wrapper

def valid_im(fn):
  def wrapper(im:npimg, *args, **kwargs):
    assert isinstance(im, np.ndarray)
    assert len(im.shape) == 3     # [H, W, C=1/3]
    assert 0.0 <= im.min() and im.max() <= 1.0
    assert im.dtype == dtype

    return fn(im, *args, **kwargs)
  return wrapper

def valid_delta(fn):
  def wrapper(im:npimg, *args, **kwargs):
    assert isinstance(im, np.ndarray)
    assert -1.0 <= im.min() and im.max() <= 1.0
    assert im.dtype == dtype

    return fn(im, *args, **kwargs)
  return wrapper

def valid_X(fn):
  def wrapper(X:Tensor, *args, **kwargs):
    assert isinstance(X, Tensor)
    assert len(X.shape) == 4
    assert X.shape[0] == 1
    assert X.shape[1] in [1, 3]

    return fn(X, *args, **kwargs)
  return wrapper

''' PIL Image '''

def get_img(fp:str, mode:str=None) -> PILImage:
  return Image.open(fp).convert(mode)

def img_stat(img:PILImage, name='img'):
  print(f'[{name}]')
  print(f'  size: {img.size}')
  print(f'  mode: {img.mode}')


def img_resize(img:PILImage, size:Tuple[int, int]) -> PILImage:
  return img.resize(size, resample=Image.Resampling.LANCZOS)


''' Numpy array '''

def get_im(fp:str, mode:str=None) -> npimg:
  return img_to_im(get_img(fp, mode))

def im_stat(im:npimg, name='im', px=False):
  if px: im = (im * 255).astype(np.int32)
  print(f'[{name}]')
  print(f'  shape: {im.shape}')
  print(f'  max:   {im.max()}')
  print(f'  min:   {im.min()}')
  print(f'  avg:   {im.mean()}')
  print(f'  std:   {im.std()}')


def im_Linf(im1:npimg, im2:npimg) -> float:
  return np.abs(im1 - im2).max()

def im_L1(im1:npimg, im2:npimg) -> float:
  return np.abs(im1 - im2).mean()

def im_dist(im1:npimg, im2:npimg):
  print('Linf:', im_Linf(im1, im2))
  print('L1:',   im_L1  (im1, im2))

def im_eq(im1:npimg, im2:npimg, eps:float=1/256) -> bool:
  return im_Linf(im1, im2) < eps


def im_clip(im:npimg) -> npimg:
  return im.clip(0.0, 1.0)

def im_minmax_norm(im:npimg, chan_wise=True) -> npimg:
  if chan_wise:
    vmin = im.min(axis=0, keepdims=True).min(axis=1, keepdims=True)
    vmax = im.max(axis=0, keepdims=True).max(axis=1, keepdims=True)
    if (vmax - vmin).mean() < eps: return im
  else:
    vmin, vmax = im.min(), im.max()
    if vmax - vmin < eps: return im
  return (im - vmin) / (vmax - vmin)

def im_norm(im:npimg, chan_wise=True, ret_stats=False) -> Union[npimg, Tuple[npimg, Tuple[float, float]]]:
  if chan_wise:
    try:
      std = im.std (axis=(0, 1), keepdims=True)
      if std.mean() < eps: return im
      avg = im.mean(axis=(0, 1), keepdims=True)
    except:
      im_t = torch.from_numpy(im)
      std = im_t.std (axis=[0, 1], keepdims=True).numpy()
      if std.mean() < eps: return im
      avg = im_t.mean(axis=[0, 1], keepdims=True).numpy()
  else:
    std = im.std()
    if std < eps: return im
    avg = im.mean()
  
  if ret_stats:
    return (im - avg) / std, (avg, std)
  else:
    return (im - avg) / std


@valid_delta
def im_shift_01(im:npimg) -> npimg:
  return (im + 1) / 2

@valid_im
def im_shift_n1p1(im:npimg) -> npimg:
  return im * 2 - 1


@valid_im
def im_mask_lowcut(im:npimg, thresh:float=0.0) -> npimg:
  assert type(thresh) == float
  # map pixel under thresh to 0, renorm above thresh to [0.0, 1.0]
  mask = im >= thresh
  im = im * mask
  return im_minmax_norm(im)

@valid_im
def im_mask_highcut(im:npimg, thresh:float=0.0) -> npimg:
  assert type(thresh) == float
  # clip pixel above thresh to 1.0, renorm below thresh to [0.0, 1.0]
  mask = im <= thresh
  im_low = im * mask
  im_low = im_minmax_norm(im_low)
  im_high = np.ones_like(im) * ~mask
  return im_low + im_high


def im_highext_torch(im:npimg, k:int=7) -> npimg:
  X = im_to_X(im)
  X = nn.MaxPool2d(kernel_size=k, stride=1, padding=k//2)(X)
  im = X_to_im(X)
  return im

def im_highext_pil(im:npimg, size:int=7) -> npimg:
  img = im_to_img(im)
  img = img.filter(ImageFilter.MaxFilter(size=size))
  im = img_to_im(img)
  return im

@valid_delta
def im_delta_to_motion(im:npimg, thresh:float=0.0, expand:int=7, px:int=4) -> npimg:
  assert type(thresh) == float

  if 'to grey' and im.shape[-1] == 3:
    im = im_shift_01(im)
    im = img_to_im(im_to_img(im).convert('L'))
    im = im_shift_n1p1(im)
  
  im = np.abs(im)                     # [0.0, 1.0], get the magnitude (maxval may < 1.0)
  im = im_minmax_norm(im)             # strech to fill [0.0, 1.0]
  im = im_mask_highcut(im, thresh)    # high-cut, [0.0, 1.0]
  F = -np.log2(1 - px/255) ** -1      # high-boost, 使得像素差值 px 的 mask 强度为 0.5
  alpha = (1 - thresh) * F            # 颠倒一下数值映射顺序
  im = 1 - (1 - im) ** (1 + alpha)    # some concave function on [0.0, 1.0], boost up values
  im = im_highext_torch(im, expand)   # high-ext
  return im


def show_img(im:npimg, title=None):
  plt.clf()
  plt.imshow(im)
  plt.axis('off')
  plt.suptitle(title)
  plt.tight_layout()
  plt.show()

def show_grid(ims:List[npimg], title=None):
  X = torch.from_numpy(ims)
  if len(X.shape) == 3: X.unsqueeze_(dim=-1) # [B, H, W, C=1|3]
  X = X.permute(0, 3, 1, 2)                  # [B, C=1|3, H, W]
  print('X.shape:', X.shape)

  grid = make_grid(X, nrow=int(len(ims)**0.5)).numpy().transpose((1, 2, 0))
  show_img(grid, title)


''' Convert '''

@valid_img
def img_to_im(img:PILImage) -> npimg:
  im = np.asarray(img, dtype=dtype) / 255.0   # [0.0, 1.0]
  if len(im.shape) == 2:
    im = np.expand_dims(im, axis=-1)          # [H, W, C=1/3]
  return im

@valid_im
def im_to_img(im:npimg) -> PILImage:
  im = (im * np.iinfo(np.uint8).max).astype(np.uint8)   # [0, 255]
  if im.shape[-1] == 1: im = im.squeeze(axis=-1)        # [H, W, C=3] or # [H, W]
  return Image.fromarray(im)                            # 'RGB' or 'L'

@valid_im
def im_to_X(im:npimg) -> Tensor:
  im = im.transpose([2, 0, 1])  # [H, W, C=1] => [C=1, H, W]
  X = torch.from_numpy(im)      # [C=1, H, W]
  X.unsqueeze_(dim=0)           # [B=1, C=1, H, W]
  return X

@valid_X
def X_to_im(X:Tensor) -> npimg:
  X = X.squeeze(dim=0)          # [B=1, C=1, H, W]
  im = X.numpy()                # [C=1, H, W]
  im = im.transpose([1, 2, 0])  # [H, W, C=1]
  return im


if __name__ == '__main__':
  fp = sys.argv[1]

  img = get_img(fp)
  img_stat(img)

  im = img_to_im(img)
  im_stat(im, px=True)
