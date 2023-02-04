#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/02/01 

# a script to experiment on frame delta

import sys
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

from img_utils import *


def make_frame_delta(in_dp:Path):
  assert in_dp.is_dir()
  out_dp = in_dp / '..' / f'framedelta.{in_dp.name}'
  out_dp.mkdir(exist_ok=True)

  fps = list(in_dp.iterdir())
  im0, im1 = None, get_im(fps[0])
  for fp in tqdm(fps[1:]):
    im0, im1 = im1, get_im(fp)
    d = im1 - im0           # [-1, 1]
    d_n = im_shift_01(d)    # [0, 1]
    img = im_to_img(d_n)
    img.save(out_dp / f'{fp.stem}.png')


def compare_frame_delta(dp1:Path, dp2:Path):
  assert dp1.is_dir() and dp2.is_dir()

  fps1 = list(dp1.iterdir())
  fps2 = list(dp2.iterdir())
  if len(fps1) != len(fps2):
    print('Warn: file count mismatch!')
    minlen = min(len(fps1), len(fps2))
    fps1 = fps1[:minlen]
    fps2 = fps2[:minlen]

  for fp1, fp2 in zip(fps1, fps2):
    im1, im2 = get_im(fp1).astype(dtype), get_im(fp2).astype(dtype)   # [0, 1]
    s1,  s2  = (im1 > 0.5).astype(dtype), (im2 > 0.5).astype(dtype)

    plt.subplot(231) ; plt.imshow(im1)       ; plt.axis('off')
    plt.subplot(232) ; plt.imshow(s1)        ; plt.axis('off')
    plt.subplot(233) ; plt.hist  (im1.flatten(), bins=100)
    plt.subplot(234) ; plt.imshow(im2)       ; plt.axis('off')
    plt.subplot(235) ; plt.imshow(s2)        ; plt.axis('off')
    plt.subplot(236) ; plt.hist  (im2.flatten(), bins=100)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
  if len(sys.argv) == 2:
    print('>> make...')
    make_frame_delta(Path(sys.argv[1]))
  elif len(sys.argv) == 3:
    print('>> compare...')
    compare_frame_delta(Path(sys.argv[1]), Path(sys.argv[2]))
  else: raise ValueError
