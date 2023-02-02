#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/02/01 

# a script to experiment on frame delta as mask

import sys
from pathlib import Path
from PIL import Image

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt


def get_img(fp: Path) -> NDArray[np.float16]:
  img = Image.open(fp).convert('RGB')
  im = np.asarray(img, dtype=np.float16) / 255.0  # [H, W, C]
  im = im * 2 - 1   # [-1, 1]
  return im


def make_frame_mask(fp:Path):
  assert fp.is_file()

  delta = get_img(fp)

  mask = np.abs(delta)        # [0, 1]
  print('mask.max:', mask.max())
  print('mask.min:', mask.min())
  
  mask_n = mask / mask.max()  # rescaled
  print('mask_n.max:', mask_n.max())
  print('mask_n.min:', mask_n.min())

  plt.subplot(121) ; plt.imshow(mask  .astype(np.float32)) ; plt.axis('off')
  plt.subplot(122) ; plt.imshow(mask_n.astype(np.float32)) ; plt.axis('off')
  plt.tight_layout()
  plt.show()


if __name__ == '__main__':
  make_frame_mask(Path(sys.argv[1]))
