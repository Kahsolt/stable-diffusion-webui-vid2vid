#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/02/04

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from img_utils import *

# set the frame numder to debug :)
iframe = 24
lowcut = 32
highext = 9

ws_dp = Path(r'D:\Desktop\Workspace\@Githubs\stable-diffusion-webui\outputs\sd-webui-vid2vid\demo')
imgA  = get_img(ws_dp / 'img2img'    / f'{iframe:05d}.png')
imgB  = get_img(ws_dp / 'img2img'    / f'{iframe+1:05d}.png')
ref   = get_img(ws_dp / 'frames'     / f'{iframe+1:05d}.png')
fd    = get_img(ws_dp / 'framedelta' / f'{iframe+1:05d}.png')
fd    = img_resize(fd, imgB.size)

imA   = img_to_im(imgA)
imB   = img_to_im(imgB)
ref   = img_to_im(ref)

tgt_d = im_shift_n1p1(img_to_im(fd))  # [-1, 1]
cur_d = imB - imA                     # [-1, 1]
dd    = cur_d - tgt_d                 # [-2, 2]

mask = im_delta_to_motion(tgt_d, thresh=lowcut/255, expand=highext)  # [0, 1]
im_stat(mask, 'mask')

new_d = cur_d * mask
im_stat(cur_d, 'cur_d')
im_stat(tgt_d, 'tgt_d')
im_stat(new_d, 'new_d')

imZ = im_clip(imA + new_d)

plt.clf()
plt.subplot(331) ; plt.imshow(np.abs(cur_d)) ; plt.title('cur_d') ; plt.axis('off')
plt.subplot(332) ; plt.imshow(np.abs(tgt_d)) ; plt.title('tgt_d') ; plt.axis('off')
plt.subplot(333) ; plt.imshow(mask)          ; plt.title('mask')  ; plt.axis('off')
plt.subplot(334) ; plt.imshow(np.abs(dd))    ; plt.title('dd')    ; plt.axis('off')
plt.subplot(335) ; plt.imshow(np.abs(new_d)) ; plt.title('new_d') ; plt.axis('off')
plt.subplot(336) ; plt.imshow(ref)           ; plt.title('tgt')   ; plt.axis('off')
plt.subplot(337) ; plt.imshow(imA)           ; plt.title('i2i_A') ; plt.axis('off')
plt.subplot(338) ; plt.imshow(imB)           ; plt.title('i2i_B') ; plt.axis('off')
plt.subplot(339) ; plt.imshow(imZ)           ; plt.title('i2i_Z') ; plt.axis('off')
plt.tight_layout()
plt.show()
