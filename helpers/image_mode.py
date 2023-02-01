#!/usr/bin/env python3
# Author: Armit
# Create Time: 2023/02/01

# a script to quick identify image mode

from PIL import Image
from argparse import ArgumentParser

if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('img', help='path to mask image file')
  args = parser.parse_args()

  print(Image.open(args.img).mode)
