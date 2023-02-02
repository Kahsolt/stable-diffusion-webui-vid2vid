#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/02/02 

# a script to quick recommend suitable canvas sizes for img2img
# NOTE: results are sorted by aspect-ratio similarity

from traceback import print_exc

HW_MIN  = 360
HW_MAX  = 720
HW_STEP = 8
TOP_K   = 15 

print('Input video resolution width and height, e.g.:')
print('   1024 768 ')
print('   1024x768 ')
print('   1024 x 768 ')
print()

try:
  while True:
    s = input('>> query (q to quit): ')
    if s in ['q', 'quit']: break

    try:
      if 'x' in s:
        w, h = [int(x.strip()) for x in s.split('x')]
      else:
        w, h = [int(x.strip()) for x in s.split(' ')]
      r = w / h
      
      print(f'<< origianl aspect ratio: {r}')

      dr_wh = []
      for W in range(HW_MIN, HW_MAX+1, HW_STEP):
        for H in range(HW_MIN, HW_MAX+1, HW_STEP):
          dr_wh.append((abs(W / H - r), W / H, (W, H)))
      dr_wh.sort()

      for _, r, (w, h) in dr_wh[:TOP_K]:
        print(f'   w: {w}, h: {h}; AR: {r}')

    except:
      print('<< bad input format')
      print_exc()

except KeyboardInterrupt:
  pass
