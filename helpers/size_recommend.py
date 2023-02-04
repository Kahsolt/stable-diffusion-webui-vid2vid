#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/02/02 

# a script to quick recommend suitable canvas sizes for img2img
# NOTE: results are sorted by aspect-ratio similarity

import os
import sys
from functools import cmp_to_key

HW_MIN  = 360
HW_MAX  = 720
HW_STEP = 32
TOP_K   = 15 

def cmp(x, y):
  if x[0] != y[0]:
    return x[0] - y[0]            # ar diff
  else:
    if y[2][0] != x[2][0]:
      return y[2][0] - x[2][0]    # width
    else:
      return y[2][1] - x[2][1]    # height

try:
  sz = [int(x.strip()) for x in sys.argv[1:]]
  if   len(sz) == 1: w, h = sz[0], sz[0]
  elif len(sz) == 2: w, h = sz
  else: raise ValueError

  r = w / h
  print(f'>> origianl aspect ratio: {r}')

  dr_wh = []
  for W in range(HW_MIN, HW_MAX+1, HW_STEP):
    for H in range(HW_MIN, HW_MAX+1, HW_STEP):
      dr_wh.append((abs(W / H - r), W / H, (W, H)))
  dr_wh = sorted(dr_wh, key=cmp_to_key(cmp))

  for _, r, (w, h) in dr_wh[:TOP_K]:
    print(f'   w: {w}, h: {h}; AR: {r}')

except:
  _ = os.path.basename(__file__)
  print(f'Usage: {_} <width> [height]')
  print(f'  {_} 512')
  print(f'  {_} 1024 768')
