#!/usr/bin/env python3
import os

import fiona
import numpy as np
import shapely.geometry

try:
    __import__("frame_field_learning.local_utils")
except ImportError:
    print("ERROR: The frame_field_learning package is not installed! "
          "Execute script setup.sh to install local dependencies such as frame_field_learning in develop mode.")
    exit()


from lydorn_utils import polygon_utils, python_utils


def load_shapefile(filepath):
    shapefile = fiona.open(filepath)
    geoms = [shapely.geometry.shape(feat["geometry"]) for feat in shapefile]
    # print(geoms[0].exterior.coords[:])
    # print(geoms[-1].exterior.coords[:])
    return geoms


def eval_shapefile(gt_polygons, pred_info):
    # Compute metrics
    metrics_filepath = os.path.splitext(pred_info["shapefile_filepath"])[0] + ".metrics.json"
    metrics = python_utils.load_json(metrics_filepath)
    if not metrics:
        # Load pred shp
        pred_polygons = load_shapefile(pred_info["shapefile_filepath"])
        fixed_dt_polygons = polygon_utils.fix_polygons(pred_polygons)
        print(f"Loaded {len(fixed_dt_polygons)} pred polygons")
        max_angle_diffs = polygon_utils.compute_polygon_contour_measures(fixed_dt_polygons, gt_polygons,
                                                                         sampling_spacing=1.0, min_precision=0.5,
                                                                         max_stretch=2)
        max_angle_diffs = [value for value in max_angle_diffs if value is not None]
        max_angle_diffs = np.array(max_angle_diffs)
        max_angle_diffs = max_angle_diffs * 180 / np.pi  # Convert to degrees
        metrics = {
            "max_angle_diffs": list(max_angle_diffs)
        }
        python_utils.save_json(metrics_filepath, metrics)
    print(f"Got {len(metrics['max_angle_diffs'])} max_angle_diff values")
    return metrics


def eval_shapefiles(gt_info, pred_info_list):
    # Load gt shp
    gt_polygons = load_shapefile(gt_info["shapefile_filepath"])
    fixed_gt_polygons = polygon_utils.fix_polygons(gt_polygons, buffer=0.0001)  # Buffer adds vertices but is needed to repair some geometries
    print(f"Loaded {len(fixed_gt_polygons)} gt polygons")

    # import matplotlib.pyplot as plt
    # fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(8, 4), sharex=True, sharey=True)
    # ax = axes.ravel()
    # ax[0].set_aspect('equal', adjustable='box')
    # polygon_utils.plot_geometries(ax[0], fixed_gt_polygons)
    # # polygon_utils.plot_geometries(ax[1], target)
    # # polygon_utils.plot_geometries(ax[2], [projected_exterior, *projected_interiors])
    # fig.tight_layout()
    # plt.show()

    for pred_info in pred_info_list:
        metrics = eval_shapefile(fixed_gt_polygons, pred_info)


def main():
    gt_info = {
        "name": "Bangkok",
        "shapefile_filepath": "/home/lydorn/repos/lydorn/frame_field_learning/frame_field_learning/test_images/Bangkok/BangkokGT/Building_Thailand_Bangkok_pansharpened25.shp"
    }

    pred_info_list = [
        {
            "name": "ACM",
            "shapefile_filepath": "/home/lydorn/repos/lydorn/frame_field_learning/frame_field_learning/test_images/Bangkok/Bangkok3bands.poly_acm.shp"
        },
        {
            "name": "ASM",
            "shapefile_filepath": "/home/lydorn/repos/lydorn/frame_field_learning/frame_field_learning/test_images/Bangkok/Bangkok3bands.poly_asm.shp"
        },
        {
            "name": "ASM regularized",
            "shapefile_filepath": "/home/lydorn/repos/lydorn/frame_field_learning/frame_field_learning/test_images/Bangkok/Bangkok3bands.reg.shp"
        },
        {
            "name": "Company",
            "shapefile_filepath": "/home/lydorn/repos/lydorn/frame_field_learning/frame_field_learning/test_images/Bangkok/Luxcarta/Building_Thailand_Bangkok_pansharpened25.shp"
        },
    ]


    eval_shapefiles(gt_info, pred_info_list)


if __name__ == '__main__':
    main()