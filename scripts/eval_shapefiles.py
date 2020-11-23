#!/usr/bin/env python3
import os

import fiona
import numpy as np
import shapely.geometry
import argparse
from multiprocess import Pool
from functools import partial
from tqdm import tqdm

try:
    __import__("frame_field_learning.local_utils")
except ImportError:
    print("ERROR: The frame_field_learning package is not installed! "
          "Execute script setup.sh to install local dependencies such as frame_field_learning in develop mode.")
    exit()

from lydorn_utils import polygon_utils, python_utils, print_utils, geo_utils


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '--im_filepath',
        required=True,
        type=str,
        nargs='*',
        help='Path(s) to tiff images (to get projection info).')
    argparser.add_argument(
        '--gt_filepath',
        required=True,
        type=str,
        nargs='*',
        help='Path(s) to ground truth.')
    argparser.add_argument(
        '--pred_filepath',
        required=True,
        type=str,
        nargs='*',
        help='Path(s) to predictions.')
    argparser.add_argument(
        '--overwrite',
        action='store_true',
        help='The default behavior is to not re-compute a metric if it already has been computed. '
             'Add the --overwrite flag if you want to overwrite existing metric results.')

    args = argparser.parse_args()
    return args


def load_geom(geom_filepath, im_filepath):
    ext = os.path.splitext(geom_filepath)[-1]
    if ext == ".geojson":
        file = fiona.open(geom_filepath)
        assert len(file) == 1, "There should be only one feature per file"
        feat = file[0]
        geoms = shapely.geometry.shape(feat["geometry"])
    elif ext == ".shp":
        polygons, _ = geo_utils.get_polygons_from_shapefile(im_filepath, geom_filepath, progressbar=False)
        geoms = shapely.geometry.collection.GeometryCollection([shapely.geometry.Polygon(polygon[:, ::-1]) for polygon in polygons])
    else:
        raise ValueError(f"Geometry can not be loaded from a {ext} file.")
    return geoms


def extract_name(filepath):
    basename = os.path.basename(filepath)
    name = basename.split(".")[0]
    return name


def match_im_gt_pred(im_filepaths, gt_filepaths, pred_filepaths):
    im_names = list(map(extract_name, im_filepaths))
    gt_names = list(map(extract_name, gt_filepaths))

    im_gt_pred_filepaths = []
    for pred_filepath in pred_filepaths:
        name = extract_name(pred_filepath)
        im_index = im_names.index(name)
        gt_index = gt_names.index(name)
        im_gt_pred_filepaths.append((im_filepaths[im_index], gt_filepaths[gt_index], pred_filepath))

    return im_gt_pred_filepaths


def eval_one(im_gt_pred_filepath, overwrite=False):
    im_filepath, gt_filepath, pred_filepath = im_gt_pred_filepath
    metrics_filepath = os.path.splitext(pred_filepath)[0] + ".metrics.json"
    iou_filepath = os.path.splitext(pred_filepath)[0] + ".iou.json"

    metrics = False
    iou = False
    if not overwrite:
        # Try reading metrics and iou json
        metrics = python_utils.load_json(metrics_filepath)
        iou = python_utils.load_json(iou_filepath)

    if not metrics or not iou:
        # Have to compute at least one so load geometries
        gt_polygons = load_geom(gt_filepath, im_filepath)
        fixed_gt_polygons = polygon_utils.fix_polygons(gt_polygons,
                                                       buffer=0.0001)  # Buffer adds vertices but is needed to repair some geometries
        pred_polygons = load_geom(pred_filepath, im_filepath)
        fixed_pred_polygons = polygon_utils.fix_polygons(pred_polygons)

        if not metrics:
            # Compute and save metrics
            max_angle_diffs = polygon_utils.compute_polygon_contour_measures(fixed_pred_polygons, fixed_gt_polygons,
                                                                             sampling_spacing=1.0, min_precision=0.5,
                                                                             max_stretch=2, progressbar=True)
            max_angle_diffs = [value for value in max_angle_diffs if value is not None]
            max_angle_diffs = np.array(max_angle_diffs)
            max_angle_diffs = max_angle_diffs * 180 / np.pi  # Convert to degrees
            metrics = {
                "max_angle_diffs": list(max_angle_diffs)
            }
            python_utils.save_json(metrics_filepath, metrics)

        if not iou:
            fixed_gt_polygon_collection = shapely.geometry.collection.GeometryCollection(fixed_gt_polygons)
            fixed_pred_polygon_collection = shapely.geometry.collection.GeometryCollection(fixed_pred_polygons)
            intersection = fixed_gt_polygon_collection.intersection(fixed_pred_polygon_collection).area
            union = fixed_gt_polygon_collection.union(fixed_pred_polygon_collection).area
            iou = {
                "intersection": intersection,
                "union": union
            }
            python_utils.save_json(iou_filepath, iou)

    return {
        "metrics": metrics,
        "iou": iou,
    }


def main():
    args = get_args()
    print_utils.print_info(f"INFO: evaluating {len(args.pred_filepath)} predictions.")

    # Match files together
    im_gt_pred_filepaths = match_im_gt_pred(args.im_filepath, args.gt_filepath, args.pred_filepath)

    pool = Pool()
    metrics_iou_list = list(tqdm(pool.imap(partial(eval_one, overwrite=args.overwrite), im_gt_pred_filepaths), desc="Compute eval metrics", total=len(im_gt_pred_filepaths)))

    # Aggregate metrics and IoU
    aggr_metrics = {
        "max_angle_diffs": []
    }
    aggr_iou = {
        "intersection": 0,
        "union": 0
    }
    for metrics_iou in metrics_iou_list:
        if metrics_iou["metrics"]:
            aggr_metrics["max_angle_diffs"] += metrics_iou["metrics"]["max_angle_diffs"]
        if metrics_iou["iou"]:
            aggr_iou["intersection"] += metrics_iou["iou"]["intersection"]
            aggr_iou["union"] += metrics_iou["iou"]["union"]
    aggr_iou["iou"] = aggr_iou["intersection"] / aggr_iou["union"]

    aggr_metrics_filepath = os.path.join(os.path.dirname(args.pred_filepath[0]), "aggr_metrics.json")
    aggr_iou_filepath = os.path.join(os.path.dirname(args.pred_filepath[0]), "aggr_iou.json")
    python_utils.save_json(aggr_metrics_filepath, aggr_metrics)
    python_utils.save_json(aggr_iou_filepath, aggr_iou)


if __name__ == '__main__':
    main()
