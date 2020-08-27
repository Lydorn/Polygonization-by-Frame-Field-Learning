#!/usr/bin/env python3

###################################################################
# Use this script to convert .ply results from
# Li, M., Lafarge, F., Marlet, R.: Approximating shapes in images with low-complexitypolygons. In: CVPR (2020)
# to COCO .json format
###################################################################


import fnmatch
import os
import argparse

import skimage.io
import skimage.morphology
import numpy as np
from tqdm import tqdm
import json
from plyfile import PlyData, PlyElement
import shapely.geometry
import shapely.ops

from frame_field_learning import polygonize_utils, save_utils


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '--ply_dirpath',
        required=True,
        type=str,
        help='Path to the directory where the .ply are.')
    argparser.add_argument(
        '--mask_dirpath',
        required=True,
        type=str,
        help='Path to the directory where the masks are (used to compute probability of each polygonal partition.')
    argparser.add_argument(
        '--output_filepath',
        required=True,
        type=str,
        help='Filepath of the final .json.')
    args = argparser.parse_args()
    return args


def ply_to_json(ply_dirpath, mask_dirpath, output_filepath, mask_filename_format="{:012d}.png"):
    filenames = fnmatch.filter(os.listdir(ply_dirpath), "*.ply")

    all_annotations = []
    for filename in tqdm(filenames, desc="Process ply files:"):
        image_id = int(os.path.splitext(filename)[0])

        # Load .ply
        plydata = PlyData.read(os.path.join(ply_dirpath, filename))
        x = plydata['vertex']['x']
        y = 299 - plydata['vertex']['y']
        pos = np.stack([x, y], axis=1)
        vertex1 = plydata['edge'].data["vertex1"]
        vertex2 = plydata['edge'].data["vertex2"]
        edge_index = np.stack([vertex1, vertex2], axis=1)
        edge = pos[edge_index]
        linestrings = []
        for e in edge:
            linestrings.append(shapely.geometry.LineString(e))

        # Load mask
        mask_filename = mask_filename_format.format(image_id)
        mask = 0 < skimage.io.imread(os.path.join(mask_dirpath, mask_filename))

        # Convert to polygons
        polygons = shapely.ops.polygonize(linestrings)

        # Remove low prob polygons
        filtered_polygons = []
        filtered_polygon_probs = []
        for polygon in polygons:
            prob = polygonize_utils.compute_geom_prob(polygon, mask)
            # print("simple:", np_indicator.min(), np_indicator.mean(), np_indicator.max(), prob)
            if 0.5 < prob:
                filtered_polygons.append(polygon)
                filtered_polygon_probs.append(prob)

        annotations = save_utils.poly_coco(filtered_polygons, filtered_polygon_probs, image_id)
        all_annotations.extend(annotations)

    with open(output_filepath, 'w') as outfile:
        json.dump(all_annotations, outfile)


if __name__ == "__main__":
    args = get_args()
    ply_dirpath = args.ply_dirpath
    mask_dirpath = args.mask_dirpath
    output_filepath = args.output_filepath
    ply_to_json(ply_dirpath, mask_dirpath, output_filepath)
