#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/02/03

# a script to quick understand motion mask low-cut

import sys
import numpy as np

from img_utils import *

fp = sys.argv[1]

delta = im_shift_n1p1(get_im(fp))
im_stat(delta)

ims = np.stack([im_delta_to_motion(delta, px / 255, expand=7) for px in range(0, 256, 4)], axis=0)
show_grid(ims, title=f'low-cut from 0 to 255, step_size=4')
