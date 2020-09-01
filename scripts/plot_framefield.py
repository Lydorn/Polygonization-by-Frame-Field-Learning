import argparse

import numpy as np
import matplotlib.pyplot as plt

from frame_field_learning import plot_utils


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-f', '--filepath',
        required=True,
        type=str,
        help='Path to the .npy to plot the framefield from.')
    argparser.add_argument(
        '-o', '--out_filepath',
        required=True,
        type=str,
        help='Path to save the image.')

    args = argparser.parse_args()
    return args


def save_plot_framefield(framefield, out_filepath):
    # Setup plot
    height = framefield.shape[0]
    width = framefield.shape[1]
    f, axis = plt.subplots(1, 1, figsize=(width // 10, height // 10))

    # axis.imshow(im)
    transparent_im = np.zeros((height, width, 4))
    axis.imshow(transparent_im)

    framefield_stride = 8
    plot_utils.plot_framefield(axis, framefield, framefield_stride, alpha=1, width=2)

    axis.autoscale(False)
    axis.axis('equal')
    axis.axis('off')

    # f.tight_layout()
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Plot without margins
    plt.savefig(out_filepath, transparent=True)
    plt.close()
    # plt.show()


def main():
    # --- Process args --- #
    args = get_args()

    # Read
    framefield = np.load(args.filepath)

    # Cut bbox
    # bbox = [2700, 2300, 3200, 2900]
    # framefield = framefield[bbox[0]:bbox[2], bbox[1]:bbox[3]]

    save_plot_framefield(framefield, args.out_filepath)


if __name__ == '__main__':
    main()
