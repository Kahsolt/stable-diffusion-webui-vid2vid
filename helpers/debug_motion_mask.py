#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/02/04

import os
import tkinter as tk
import tkinter.ttk as ttk
import tkinter.filedialog as tkfdlg
from traceback import print_exc

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from img_utils import *

FIG_SIZE = (16, 8)

class App:

  def __init__(self):
    self.setup_gui()

    try:
      self.wnd.mainloop()
    except KeyboardInterrupt:
      self.wnd.quit()
    except: print_exc()

  def setup_gui(self):
    # window
    wnd = tk.Tk()
    wnd.title('Motion Mask Debugger')
    wnd.protocol('WM_DELETE_WINDOW', wnd.quit)
    self.wnd = wnd

    # top: query
    frm1 = ttk.Label(wnd)
    frm1.pack(side=tk.TOP, expand=tk.YES, fill=tk.X)
    if True:
      self.var_ws      = tk.StringVar(frm1, value='')
      self.var_idx     = tk.IntVar   (frm1, value=1)
      self.var_highext = tk.IntVar   (frm1, value=7)
      self.var_lowcut  = tk.IntVar   (frm1, value=0)
      
      frm11 = ttk.Label(frm1)
      frm11.pack(expand=tk.YES, fill=tk.X)
      if True:
        ttk.Label(frm11, text='Workspace Folder: ').pack(side=tk.LEFT, expand=tk.NO)
        ttk.Label(frm11, textvariable=self.var_ws).pack(side=tk.LEFT, expand=tk.YES, fill=tk.X)
        ttk.Button(frm11, text='Open...', command=self.open_).pack()

      frm12 = ttk.Label(frm1)
      frm12.pack(expand=tk.YES, fill=tk.X)
      if True:
        frm121 = ttk.Label(frm12)
        frm121.pack(side=tk.LEFT, expand=tk.YES, fill=tk.X)
        if True:
          tk.Label(frm121, text='High-expand').pack(side=tk.LEFT, expand=tk.NO)
          tk.Scale(frm121, command=lambda _: self.redraw(), variable=self.var_highext, orient=tk.HORIZONTAL, from_=1, to=15, resolution=2).pack(side=tk.RIGHT, expand=tk.YES, fill=tk.X)

        frm122 = ttk.Label(frm12)
        frm122.pack(side=tk.RIGHT, expand=tk.YES, fill=tk.X)
        if True:
          tk.Label(frm122, text='Low-cut').pack(side=tk.LEFT, expand=tk.NO)
          tk.Scale(frm122, command=lambda _: self.redraw(), variable=self.var_lowcut, orient=tk.HORIZONTAL, from_=0, to=255, resolution=1).pack(side=tk.RIGHT, expand=tk.YES, fill=tk.X)

      frm13 = ttk.Label(frm1)
      frm13.pack(expand=tk.YES, fill=tk.X)
      if True:
        ttk.Label(frm13, text='Frame Number').pack(side=tk.LEFT, expand=tk.NO)
        sc = tk.Scale(frm13, command=lambda _: self.redraw(new=True), variable=self.var_idx, orient=tk.HORIZONTAL, from_=1, to=1, resolution=1, tickinterval=10)
        sc.pack(expand=tk.YES, fill=tk.X)
        self.sc_idx = sc

    # bottom: plot
    frm2 = ttk.Frame(wnd)
    frm2.pack(expand=tk.YES, fill=tk.BOTH)
    if True:
      fig, axs = plt.subplots(3, 3)
      fig.tight_layout()
      fig.set_size_inches(FIG_SIZE)
      cvs = FigureCanvasTkAgg(fig, frm2)
      toolbar = NavigationToolbar2Tk(cvs)
      toolbar.update()
      cvs.get_tk_widget().pack(expand=tk.YES, fill=tk.BOTH)
      self.fig, self.axs, self.cvs = fig, axs, cvs

  def open_(self):
    dp = tkfdlg.askdirectory(mustexist=True)
    if not dp: return

    n_frames = len(os.listdir(os.path.join(dp, 'img2img')))
    print(f'>> found n_frames: {n_frames}')
    
    self.var_ws.set(dp)
    self.var_idx.set(1)
    self.sc_idx.configure(to=n_frames)
    self.sc_idx.pack(expand=tk.YES, fill=tk.X)
    self.redraw(new=True)

  def redraw(self, new=False):
    ws_dp = self.var_ws.get()
    if not ws_dp: return
    ws_dp = Path(ws_dp)
    if not ws_dp.exists(): return
    
    iframe  = self.var_idx    .get()
    lowcut  = self.var_lowcut .get()
    highext = self.var_highext.get()

    if new: # new frame
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
      #im_stat(mask, 'mask')

      new_d = cur_d * mask
      #im_stat(cur_d, 'cur_d')
      #im_stat(tgt_d, 'tgt_d')
      #im_stat(new_d, 'new_d')

      imZ = im_clip(imA + new_d)

      ax = self.axs[0][0] ; ax.cla() ; ax.imshow(np.abs(cur_d)) ; ax.set_title('cur_d') ; ax.axis('off')
      ax = self.axs[0][1] ; ax.cla() ; ax.imshow(np.abs(tgt_d)) ; ax.set_title('tgt_d') ; ax.axis('off')
      ax = self.axs[0][2] ; ax.cla() ; ax.imshow(mask)          ; ax.set_title('mask')  ; ax.axis('off')
      ax = self.axs[1][0] ; ax.cla() ; ax.imshow(np.abs(dd))    ; ax.set_title('dd')    ; ax.axis('off')
      ax = self.axs[1][1] ; ax.cla() ; ax.imshow(np.abs(new_d)) ; ax.set_title('new_d') ; ax.axis('off')
      ax = self.axs[1][2] ; ax.cla() ; ax.imshow(ref)           ; ax.set_title('tgt')   ; ax.axis('off')
      ax = self.axs[2][0] ; ax.cla() ; ax.imshow(imA)           ; ax.set_title('i2i_A') ; ax.axis('off')
      ax = self.axs[2][1] ; ax.cla() ; ax.imshow(imB)           ; ax.set_title('i2i_B') ; ax.axis('off')
      ax = self.axs[2][2] ; ax.cla() ; ax.imshow(imZ)           ; ax.set_title('i2i_Z') ; ax.axis('off')
      self.fig.tight_layout()
      self.cvs.draw()

      self.imA   = imA
      self.tgt_d = tgt_d
      self.cur_d = cur_d
    
    else:  # just differential update
      mask = im_delta_to_motion(self.tgt_d, thresh=lowcut/255, expand=highext)  # [0, 1]
      #im_stat(mask, 'mask')
      new_d = self.cur_d * mask
      imZ = im_clip(self.imA + new_d)

      ax = self.axs[0][2] ; ax.cla() ; ax.imshow(mask)          ; ax.set_title('mask')  ; ax.axis('off')
      ax = self.axs[1][1] ; ax.cla() ; ax.imshow(np.abs(new_d)) ; ax.set_title('new_d') ; ax.axis('off')
      ax = self.axs[2][2] ; ax.cla() ; ax.imshow(imZ)           ; ax.set_title('i2i_Z') ; ax.axis('off')
      self.fig.tight_layout()
      self.cvs.draw()


if __name__ == '__main__':
  App()
