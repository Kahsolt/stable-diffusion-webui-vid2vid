#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/01/24 

# a script to quick understand sigma schedule

import tkinter as tk
import tkinter.ttk as ttk
import tkinter.messagebox as tkmsg
from traceback import print_exc, format_exc

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def append_zero(x):
  return np.concatenate([x, np.zeros([1])])

def get_sigmas_karras(n, sigma_min, sigma_max, rho=7.):
  ramp = np.linspace(0, 1, n)
  min_inv_rho = sigma_min ** (1 / rho)
  max_inv_rho = sigma_max ** (1 / rho)
  sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
  return append_zero(sigmas)

def get_sigmas_exponential(n, sigma_min, sigma_max):
  sigmas = np.exp(np.linspace(np.log(sigma_max), np.log(sigma_min), n))
  return append_zero(sigmas)

def get_sigmas_polyexponential(n, sigma_min, sigma_max, rho=1.):
  ramp = np.linspace(1, 0, n) ** rho
  sigmas = np.exp(ramp * (np.log(sigma_max) - np.log(sigma_min)) + np.log(sigma_min))
  return append_zero(sigmas)

def get_sigmas_vp(n, beta_d=19.9, beta_min=0.1, eps_s=1e-3):
  t = np.linspace(1, eps_s, n)
  sigmas = np.sqrt(np.exp(beta_d * t ** 2 / 2 + beta_min * t) - 1)
  return append_zero(sigmas)

def get_sigmas_linear(n, sigma_min, sigma_max):
  sigmas = np.linspace(sigma_max, sigma_min, n)
  return append_zero(sigmas)


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
    wnd.title('Sigma Schedulers')
    wnd.protocol('WM_DELETE_WINDOW', wnd.quit)
    self.wnd = wnd

    # top: query
    frm1 = ttk.Label(wnd)
    frm1.pack(side=tk.TOP, anchor=tk.N, expand=tk.YES, fill=tk.X)
    if True:
      self.var_steps      = tk.IntVar   (frm1, value=50)
      self.var_denoise_w  = tk.DoubleVar(frm1, value=0.5)
      self.var_sigmax_max = tk.DoubleVar(frm1, value=10.0)
      self.var_sigmax_min = tk.DoubleVar(frm1, value=0.1)
      
      frm11 = ttk.Label(frm1)
      frm11.pack(expand=tk.YES, fill=tk.X)
      if True:
        tk.Label(frm11, text='Steps').pack(side=tk.LEFT, expand=tk.NO)
        tk.Scale(frm11, command=lambda _: self.redraw(), variable=self.var_steps, orient=tk.HORIZONTAL, from_=0, to=150, tickinterval=10, resolution=1).pack(expand=tk.YES, fill=tk.X)

      frm12 = ttk.Label(frm1)
      frm12.pack(expand=tk.YES, fill=tk.X)
      if True:
        tk.Label(frm12, text='Denoising strength').pack(side=tk.LEFT, expand=tk.NO)
        tk.Scale(frm12, command=lambda _: self.redraw(), variable=self.var_denoise_w,  orient=tk.HORIZONTAL, from_=0.0, to=1.0, tickinterval=0.1, resolution=0.01).pack(expand=tk.YES, fill=tk.X)

      frm13 = ttk.Label(frm1)
      frm13.pack(expand=tk.YES, fill=tk.X)
      if True:
        tk.Label(frm13, text='Sigma max').pack(side=tk.LEFT, expand=tk.NO)
        tk.Scale(frm13, command=lambda _: self.redraw(), variable=self.var_sigmax_max, orient=tk.HORIZONTAL, from_=0.0, to=19.9, tickinterval=1.0, resolution=0.01).pack(expand=tk.YES, fill=tk.X)
      
      frm14 = ttk.Label(frm1)
      frm14.pack(expand=tk.YES, fill=tk.X)
      if True:
        tk.Label(frm14, text='Sigma min').pack(side=tk.LEFT, expand=tk.NO)
        tk.Scale(frm14, command=lambda _: self.redraw(), variable=self.var_sigmax_min, orient=tk.HORIZONTAL, from_=0.0, to=19.9, tickinterval=1.0, resolution=0.01).pack(expand=tk.YES, fill=tk.X)

    # bottom: plot
    frm2 = ttk.Frame(wnd)
    frm2.pack(side=tk.BOTTOM, expand=tk.YES, fill=tk.BOTH)
    if True:
      fig, ax = plt.subplots()
      fig.tight_layout()
      cvs = FigureCanvasTkAgg(fig, frm2)
      cvs.get_tk_widget().pack(expand=tk.YES, fill=tk.BOTH)
      self.fig, self.ax, self.cvs = fig, ax, cvs
      self.redraw()

  def redraw(self):
    steps     = self.var_steps     .get()
    denoise_w = self.var_denoise_w .get()
    t_enc     = int(min(denoise_w, 0.999) * steps)
    sigma_max = self.var_sigmax_max.get()
    sigma_min = self.var_sigmax_min.get()
    sigma_min = max(sigma_min, 1e-3)

    if steps <= 0: return
    if sigma_max < sigma_min: return

    try:
      sigmas = {
        'karras':      get_sigmas_karras         (steps, sigma_min, sigma_max, rho=7),
        #'karras-5':    get_sigmas_karras         (steps, sigma_min, sigma_max, rho=5),  # alike polyexp <1.0
        #'karras-9':    get_sigmas_karras         (steps, sigma_min, sigma_max, rho=9),  # alike polyexp >1.0
        'exp':         get_sigmas_exponential    (steps, sigma_min, sigma_max),
        'polyexp:0.8': get_sigmas_polyexponential(steps, sigma_min, sigma_max, rho=0.8),
        'polyexp:1.2': get_sigmas_polyexponential(steps, sigma_min, sigma_max, rho=1.2),
        'vp':          get_sigmas_vp             (steps, sigma_max, sigma_min),
        'linear':      get_sigmas_linear         (steps, sigma_min, sigma_max),
      }

      self.ax.cla()
      for name, data in sigmas.items():
        self.ax.plot(data, label=name)
      self.ax.axvline(steps - t_enc, linestyle='-', c='grey')
      x_ticks = list(range(0, steps+1))
      if len(x_ticks) > 10:
        r = len(x_ticks) // 10
        x_ticks = x_ticks[::r]
      self.ax.set_xticks(x_ticks)
      self.ax.set_xlim(-1, steps+1)
      self.fig.legend()
      self.cvs.draw()
    except:
      info = format_exc()
      print(info)
      tkmsg.showerror('Error', info)


if __name__ == '__main__':
  App()
