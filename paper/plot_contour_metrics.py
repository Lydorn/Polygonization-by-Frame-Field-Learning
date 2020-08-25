import argparse
import fnmatch
import os

import random
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
from tqdm import tqdm

from functools import partial

from lydorn_utils import python_utils
from lydorn_utils import print_utils


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '--dirpath',
        default="/home/lydorn/data/mapping_challenge_dataset/eval_runs",
        type=str,
        help='Path to eval directory')

    args = argparser.parse_args()
    return args


def plot_metric(dirpath, info_list):
    legend = []
    for info in info_list:
        metrics_filepath = os.path.join(dirpath, info["metrics_filepath"])
        metrics = python_utils.load_json(metrics_filepath)
        if metrics:
            max_angle_diffs = np.array(metrics["max_angle_diffs"])
            total = len(max_angle_diffs)
            angle_thresholds = range(0, 91)
            fraction_under_threshold_list = []
            for angle_threshold in angle_thresholds:
                fraction_under_threshold = np.sum(max_angle_diffs < angle_threshold) / total
                fraction_under_threshold_list.append(fraction_under_threshold)
            # Plot
            plt.plot(angle_thresholds, fraction_under_threshold_list)

            # Compute mean
            mean_error = np.mean(max_angle_diffs)

            legend.append(f"{info['name']}: {mean_error:.1f}°")

        else:
            print_utils.print_warning("WARNING: could not open {}".format(info["metrics_filepath"]))

    plt.legend(legend, loc='lower right')
    plt.xlabel("Threshold (degrees)")
    plt.ylabel("Fraction of detections")
    axes = plt.gca()
    axes.set_xlim([0, 90])
    axes.set_ylim([0, 1])
    title = f"Cumulative max tangent angle error per detection"
    plt.title(title)
    plt.savefig(title.lower().replace(" ", "_") + ".pdf")
    plt.show()


def main():
    args = get_args()


    # Mapping challenge:
    info_list = [
        # {
        #     "name": "Baseline (no field)",
        #     "metrics_filepath": "/home/lydorn/data/mapping_challenge_dataset/eval_runs/mapping_dataset.unet16.field_off.train_val | 2020-02-28 23:51:16/test.metrics.test.annotation.poly.simple.tol_1.json"
        # },
        # {
        #     "name": "Baseline (full)",
        #     "metrics_filepath": "/home/lydorn/data/mapping_challenge_dataset/eval_runs/mapping_dataset.unet16.train_val | 2020-02-21 03:09:03/test.metrics.test.annotation.poly.simple.tol_1.json"
        # },
        # {
        #     "name": "Ours (full)",
        #     "metrics_filepath": "/home/lydorn/data/mapping_challenge_dataset/eval_runs/mapping_dataset.unet16.train_val | 2020-02-21 03:09:03/test.metrics.test.annotation.poly.acm.tol_1.json"
        # },
        #
        # {
        #     "name": "Baseline (full) Unet-ResNet101",
        #     "metrics_filepath": "/home/lydorn/data/mapping_challenge_dataset/eval_runs/mapping_dataset.unet_resnet101_pretrained | 2020-05-13 07:22:45/test.metrics.test.annotation.poly.simple.tol_1.json"
        # },
        # {
        #     "name": "Ours (full) Unet-ResNet101",
        #     "metrics_filepath": "/home/lydorn/data/mapping_challenge_dataset/eval_runs/mapping_dataset.unet_resnet101_pretrained | 2020-05-13 07:22:45/test.metrics.test.annotation.poly.acm.tol_1.json"
        # },

        {
            "name": "Baseline (no field) Unet-ResNet101 train val tol=0.125px",
            "metrics_filepath": "mapping_dataset.unet_resnet101_pretrained.field_off.train_val | 2020-05-21 08:33:20/test.metrics.test.annotation.poly.simple.tol_0.125.json"
        },
        {
            "name": "Baseline (full) Unet-ResNet101 train val tol=0.125px",
            "metrics_filepath": "mapping_dataset.unet_resnet101_pretrained.train_val | 2020-05-21 08:32:48/test.metrics.test.annotation.poly.simple.tol_0.125.json"
        },
        {
            "name": "Ours (full) Unet-ResNet101 train val tol=0.125px",
            "metrics_filepath": "mapping_dataset.unet_resnet101_pretrained.train_val | 2020-05-21 08:32:48/test.metrics.test.annotation.poly.acm.tol_0.125.json"
        },

        {
            "name": "PolyMapper",
            "metrics_filepath": "eval_runs/mapping_dataset.polymapper | 0000-00-00 00:00:00/test.metrics.test.annotation.poly.json"
        },
        {
            "name": "Li et al.",
            "metrics_filepath": "mapping_dataset.mu | 0000-00-00 00:00:00/test.metrics.test.annotation.poly.json"
        },
    ]

    # LuxCarta's Bangkok image
    # info_list = [
    #     {
    #         "name": "ACM",
    #         "metrics_filepath": "/home/lydorn/repos/lydorn/frame_field_learning/frame_field_learning/test_images/Bangkok/Bangkok3bands.poly_acm.metrics.json"
    #     },
    #     {
    #         "name": "ASM",
    #         "metrics_filepath": "/home/lydorn/repos/lydorn/frame_field_learning/frame_field_learning/test_images/Bangkok/Bangkok3bands.poly_asm.metrics.json"
    #     },
    #     {
    #         "name": "ASM regularized",
    #         "metrics_filepath": "/home/lydorn/repos/lydorn/frame_field_learning/frame_field_learning/test_images/Bangkok/Bangkok3bands.reg.metrics.json"
    #     },
    #     {
    #         "name": "Company",
    #         "metrics_filepath": "/home/lydorn/repos/lydorn/frame_field_learning/frame_field_learning/test_images/Bangkok/Luxcarta/Building_Thailand_Bangkok_pansharpened25.metrics.json"
    #     },
    # ]

    plot_metric(args.dirpath, info_list)


if __name__ == '__main__':
    main()
