#!/usr/bin/env python3
import os

import argparse
import skimage.io
import skimage.external.tifffile
from multiprocess import Pool
from functools import partial
from tqdm import tqdm
import cv2

try:
    __import__("frame_field_learning.local_utils")
except ImportError:
    print("ERROR: The frame_field_learning package is not installed! "
          "Execute script setup.sh to install local dependencies such as frame_field_learning in develop mode.")
    exit()

from lydorn_utils import print_utils


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '--filepath',
        required=True,
        type=str,
        nargs='*',
        help='Path(s) to tiff seg RGB images to convert to single channel segmentation map (only keep the first channel).')
    argparser.add_argument(
        '--out_dirpath',
        type=str,
        help='Path to the output directory for the converted images.')

    args = argparser.parse_args()
    return args


def convert_one(filepath, out_dirpath):
    image = skimage.io.imread(filepath)
    gray_image = image[:, :, 0]

    basename = os.path.basename(filepath)
    name = basename.split(".")[0]
    out_filepath = os.path.join(out_dirpath, name + ".png")
    os.makedirs(os.path.dirname(out_filepath), exist_ok=True)

    cv2.imwrite(out_filepath, gray_image, [cv2.IMWRITE_PNG_COMPRESSION, 9])


def main():
    args = get_args()
    print_utils.print_info(f"INFO: converting {len(args.filepath)} seg images.")

    pool = Pool()
    list(tqdm(pool.imap(partial(convert_one, out_dirpath=args.out_dirpath), args.filepath), desc="RGB to Gray", total=len(args.filepath)))


if __name__ == '__main__':
    main()
