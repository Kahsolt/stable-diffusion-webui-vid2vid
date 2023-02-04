#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/02/01

# a script to quick inspect basic image info

import sys
from img_utils import *

fp = sys.argv[1]

img = get_img(fp)
img_stat(img)

im = img_to_im(img)
im_stat(im, px=True)
