#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/02/02 

# a script to quick recommend suitable canvas sizes for img2img
# NOTE: results are sorted by aspect-ratio similarity

from traceback import print_exc

MODEL_HW  = 512
HW_SETP = 32
HW_MIN = 384
HW_MAX = 1024

print('Input video resolution width and height, e.g.:')
print('   1024 768 ')
print('   1024x768 ')
print('   1024 x 768 ')
print()

try:
  while True:
    s = input('>> Video resolution (q to quit): ')
    if s in ['q', 'quit']: break

    try:
      if 'x' in s:
        w, h = [int(x.strip()) for x in s.split('x')]
      else:
        w, h = [int(x.strip()) for x in s.split(' ')]
      r = w / h

      dr_wh = []
      for W in range(HW_MIN, HW_MAX+1, HW_SETP):
        for H in range(HW_MIN, HW_MAX+1, HW_SETP):
          dr_wh.append((abs(W / H - r), (W, H)))
      dr_wh.sort()

      for _, (w, h) in dr_wh[:5]:
        print(f'w: {w}, h: {h}')

    except:
      print('<< bad input format')
      print_exc()

except KeyboardInterrupt:
  pass
