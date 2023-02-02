#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/01/24 

# a script to quick understand depth mask low-cut

from PIL import Image
from argparse import ArgumentParser

import torch
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt

STEP_SIZE = 4


def renorm_mask(im:np.ndarray, thresh:int) -> np.ndarray:
  # map under thresh to 0, renorm above thresh to [0.0, 1.0]
  mask_v = im >= thresh
  im_v = im * mask_v
  vmin, vmax = im_v.min(), im_v.max()
  im = (im.astype(np.float16) - vmin) / (vmax - vmin)
  im *= mask_v
  im = (im * np.iinfo(np.uint8).max).astype(np.uint8)
  return im


def show_lowcut(args):
  img = Image.open(args.img)
  print('img.size:', img.size)
  
  im = np.asarray(img, dtype=np.uint8)

  ims = np.stack([renorm_mask(im, thr) for thr in range(0, 255, STEP_SIZE)], axis=0)  # [B, H, W]
  X = torch.from_numpy(ims).unsqueeze(dim=1)    # [B, C=1, H, W]
  print(X.shape)

  X_grid = make_grid(X, nrow=int(len(ims)**0.5))
  plt.imshow(X_grid.numpy().transpose((1, 2, 0)))
  plt.suptitle(f'low-cut from 0 to 255, step_size={STEP_SIZE}')
  plt.axis('off')
  plt.tight_layout()
  plt.show()


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('img', help='path to mask image file')
  args = parser.parse_args()

  show_lowcut(args)
