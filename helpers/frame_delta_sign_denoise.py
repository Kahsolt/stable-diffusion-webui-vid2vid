#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/02/02

# a script to experiment on frame delta denoise

import sys
from pathlib import Path
from PIL import Image, ImageFilter

import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt


def get_img(fp: Path) -> NDArray[np.float16]:
  img = Image.open(fp).convert('RGB')
  im = np.asarray(img, dtype=np.int32)  # [H, W, C]
  im = im - 128     # [-128, 127]
  return im


def frame_delta_denoise(fp:Path):
  assert fp.is_file()

  im = get_img(fp)

  im_s = np.sign(im)    # [-1, 0, 1]
  im_s = im_s + 1       # [0, 1, 2]

  img = Image.fromarray(im_s.astype(np.uint8))
  img_dn = img.filter(ImageFilter.ModeFilter(size=3))   # just vote on three numbers
  im_dn = np.asarray(img_dn)

  plt.subplot(121) ; plt.imshow((im_s  * 127.5).astype(np.float32)) ; plt.axis('off')
  plt.subplot(122) ; plt.imshow((im_dn * 127.5).astype(np.float32)) ; plt.axis('off')
  plt.tight_layout()
  plt.show()


if __name__ == '__main__':
  frame_delta_denoise(Path(sys.argv[1]))
