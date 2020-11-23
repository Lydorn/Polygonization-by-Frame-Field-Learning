import argparse
import os

import matplotlib.pyplot as plt

import numpy as np

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
        {
            "name": "UResNet101 (no field), simple poly.",
            "metrics_filepath": "mapping_dataset.unet_resnet101_pretrained.field_off.train_val | 2020-05-21 08:33:20/test.metrics.test.annotation.poly.simple.tol_0.125.json"
        },
        {
            "name": "UResNet101 (with field), simple poly.",
            "metrics_filepath": "mapping_dataset.unet_resnet101_pretrained.train_val | 2020-05-21 08:32:48/test.metrics.test.annotation.poly.simple.tol_0.125.json"
        },
        {
            "name": "UResNet101 (with field), our poly.",
            "metrics_filepath": "mapping_dataset.unet_resnet101_pretrained.train_val | 2020-05-21 08:32:48/test.metrics.test.annotation.poly.acm.tol_0.125.json"
        },
        {
            "name": "UResNet101 (no $L_{align90}$), our poly.",
            "metrics_filepath": "mapping_dataset.unet_resnet101_pretrained.align90_off.train_val | 2020-11-02 07:34:43/test.metrics.test.annotation.poly.acm.tol_0.125.json"
        },
        {
            "name": "UResNet101 (no $L_{int edge}$), our poly.",
            "metrics_filepath": "mapping_dataset.unet_resnet101_pretrained.edge_int_off.train_val | 2020-11-02 07:34:54/test.metrics.test.annotation.poly.acm.tol_0.125.json"
        },
        {
            "name": "UResNet101 (no $L_{int align}$ and $L_{edge align}$), our poly.",
            "metrics_filepath": "mapping_dataset.unet_resnet101_pretrained.seg_framefield_off.train_val | 2020-10-29 11:27:52/test.metrics.test.annotation.poly.acm.tol_0.125.json"
        },
        {
            "name": "UResNet101 (no $L_{smooth}$), our poly.",
            "metrics_filepath": "mapping_dataset.unet_resnet101_pretrained.smooth_off.train_val | 2020-10-29 11:18:33/test.metrics.test.annotation.poly.acm.tol_0.125.json"
        },

        {
            "name": "PolyMapper",
            "metrics_filepath": "mapping_dataset.polymapper | 0000-00-00 00:00:00/test.metrics.test.annotation.poly.json"
        },
        {
            "name": "U-Net variant, ASIP poly.",
            "metrics_filepath": "mapping_dataset.asip | 0000-00-00 00:00:00/test.metrics.test.annotation.poly.json"
        },
        {
            "name": "Zorzi et al.",
            "metrics_filepath": "mapping_dataset.zorzi | 0000-00-00 00:00:00/test.metrics.test.annotation.poly.json"
        },
        {
            "name": "U-Net variant, UResNet101 poly",
            "metrics_filepath": "mapping_dataset.open_solution_full | 0000-00-00 00:00:00/test.metrics.test.annotation.seg_cleaned.poly.json"
        }
    ]

    # Inria Polygonized Dataset
    # info_list = [
    #     {
    #         "name": "UResNet101 (no field), simple poly.",
    #         "metrics_filepath": "/home/lydorn/data/AerialImageDataset/raw/test/pred_ours_leaderboard_new_losses.field_off/poly_shapefile.simple.tol_1/aggr_metrics.json"
    #     },
    #     {
    #         "name": "UResNet101 (with field), our poly.",
    #         "metrics_filepath": "/home/lydorn/data/AerialImageDataset/raw/test/pred_ours_leaderboard/poly_shapefile.acm.tol_0.125/aggr_metrics.json"
    #     },
    #     {
    #         "name": "Zorzi et al.",
    #         "metrics_filepath": "/home/lydorn/data/AerialImageDataset/raw/test/pred_zorzi/shapes/aggr_metrics.json"
    #     },
    #     {
    #         "name": "ICTNet, simple poly.",
    #         "metrics_filepath": "/home/lydorn/data/AerialImageDataset/raw/test/pred_ictnet/shp/aggr_metrics.json"
    #     },
    #     {
    #         "name": "Khvedchenya, simple poly.",
    #         "metrics_filepath": "/home/lydorn/data/AerialImageDataset/raw/test/pred_khvedchenya/shp/aggr_metrics.json"
    #     },
    # ]

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
