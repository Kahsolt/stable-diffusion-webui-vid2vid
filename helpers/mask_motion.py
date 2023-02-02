#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/02/03

# a script to quick understand motion mask low-cut

from PIL import Image, ImageFilter
from PIL.Image import Image as PILImage
from argparse import ArgumentParser

import torch
from torchvision.utils import make_grid
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt

MASK_DELTA_HB_BINS = 16
MASK_DELTA_EXPAND_K = 7


def img_to_im(img: PILImage) -> NDArray[np.float32]:
  assert isinstance(img, PILImage)

  return np.asarray(img, dtype=np.float32) / 255.0    # [0.0, 1.0]

def im_to_img(im: NDArray[np.float32]) -> PILImage:
  assert 0 <= im.min() and im.max() <= 1.0
  assert isinstance(im, np.ndarray)
  assert im.dtype in [np.float16, np.float32, np.float64, np.half, np.single, np.double]

  return Image.fromarray((im * np.iinfo(np.uint8).max).astype(np.uint8))


def get_delta_mask(fp:str, thresh:int=0) -> NDArray[np.float32]:
  img = Image.open(fp).convert('L')
  im = np.asarray(img, dtype=np.float32) / 255.0  # [0.0, 1.0], non-normalized
  im = im * 2 - 1                                 # [-1.0, 1.0]
  im = np.abs(im)                                 # [0.0, 1.0], get the magnitude (maxval may < 1.0)
  im /= im.max()                                  # re-norm to [0.0, 1.0]
  alpha = -np.log(2) / np.log(1 - thresh / 255)   # 使得像素差值 thresh 的 mask 强度为 0.5
  im = 1 - (1 - im) ** (1 + alpha)                # concave function on [0.0, 1.0]
  img = im_to_img(im)
  img = img.filter(ImageFilter.MaxFilter(size=MASK_DELTA_EXPAND_K))   # 扩散选区
  im = img_to_im(img)
  return im


def show_lowcut(args):
  ims = np.stack([get_delta_mask(args.img, thr) for thr in np.linspace(1, MASK_DELTA_HB_BINS, 15)], axis=0)  # [B, H, W]
  X = torch.from_numpy(ims).unsqueeze(dim=1)    # [B, C=3, H, W]
  print(X.shape)

  X_grid = make_grid(X, nrow=int(len(ims)**0.5))
  plt.imshow(X_grid.numpy().transpose((1, 2, 0)))
  plt.suptitle(f'low-cut from 1 to {MASK_DELTA_HB_BINS}, step_size={(MASK_DELTA_HB_BINS-1)/15}')
  plt.axis('off')
  plt.tight_layout()
  plt.show()


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('img', help='path to mask image file')
  args = parser.parse_args()

  show_lowcut(args)
