#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/01/24 

# a script to quick understand depth mask low-cut

import sys
import numpy as np

from img_utils import *

fp = sys.argv[1]

mask = get_im(fp)
im_stat(mask)

ims = np.stack([im_mask_lowcut(mask, px / 255) for px in range(0, 256, 4)], axis=0)
show_grid(ims, title=f'low-cut from 0 to 255, step_size=4')
