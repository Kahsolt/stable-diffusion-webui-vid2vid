#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/02/10 

# a script to benchmark MaxPool of PIL and torch
# error is cause by uint8-quant in PIL but not in torch, it can be ignored when subtle

from time import time
from typing import Tuple
import numpy as np

from img_utils import *

N  = 100
HW = 512
C  = 3

def benchmark(k:int) -> Tuple[float, float]:
  # prepare data
  ims = [np.random.uniform(size=[HW, HW, C]).astype(dtype=dtype) for _ in range(N)]
  
  # torch
  t = time()
  ims_tc = [im_highext_torch(im, k) for im in ims]
  tm_tc = time() - t
  
  # PIL
  t = time()
  ims_pil = [im_highext_pil(im, k) for im in ims]
  tm_pil = time() - t

  # check correctness
  err = 0.0
  for imA, imB in zip(ims_pil, ims_tc):
    err += np.abs(imA - imB).mean()

  return tm_tc, tm_pil, err / N


if __name__ == '__main__':
  for k in range(3, 15+1, 2):
    tm_tc, tm_pil, err = benchmark(k)
    print(f'k = {k}')
    print(f'  torch: {tm_tc}')
    print(f'  PIL: {tm_pil}')
    print(f'  err: {err}')
