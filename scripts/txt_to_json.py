#!/usr/bin/env python3

###################################################################
# Use this script to convert .ply results from
# Li, M., Lafarge, F., Marlet, R.: Approximating shapes in images with low-complexity polygons. In: CVPR (2020)
# to COCO .json format
###################################################################


# python txt_to_json.py --txt_dirpath ~/data/mapping_challenge_dataset/eval_runs/mapping_dataset.ours_asip\ \|\ 0000-00-00\ 00:00:00/asip_output/ --mask_dirpath ~/data/mapping_challenge_dataset/eval_runs/mapping_dataset.ours_asip\ \|\ 0000-00-00\ 00:00:00/seg_single_channel/ --output_filepath ~/data/mapping_challenge_dataset/eval_runs/mapping_dataset.ours_asip\ \|\ 0000-00-00\ 00:00:00/test.annotation.poly.json


import fnmatch
import functools
import os
import argparse
import multiprocessing

import skimage.io
import skimage.morphology
from tqdm import tqdm
import json
from plyfile import PlyData, PlyElement
import shapely.geometry
import shapely.ops

import numpy as np
from descartes import PolygonPatch
import matplotlib.pyplot as plt
import skimage.color

from frame_field_learning import polygonize_utils, save_utils


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '--txt_dirpath',
        required=True,
        type=str,
        help='Path to the directory where the .txt are.')
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


def plot(image_width, image_height, faces, mask):
    fig = plt.figure()

    ax = plt.subplot(111)
    ax.set_xlim((0, image_width))
    ax.set_ylim((0, image_height))

    ax.imshow(mask, cmap="gray")

    dark_blue = '#4477aa'
    for face in faces:
        poly_patch = PolygonPatch(face, fc=dark_blue, ec=dark_blue, alpha=0.5, zorder=2)
        ax.add_patch(poly_patch)

    plt.show()


def parse_txt(txt):
    vertices = []
    edges = []
    all_face_edges_list = []

    lines = txt.split("\n")
    width, height = map(int, lines[0].split(" "))
    vertex_count, edge_count, face_count = map(int, lines[1].split(" "))

    assert lines[2] == 'vertices', "3rd line should read 'vertices'"
    line_i = 3
    vertex_i = 0
    while lines[line_i] != 'edges':
        v_id, v_x, v_y = lines[line_i].split(" ")
        v_id = int(v_id)
        assert vertex_i == v_id, "Index of vertex should be correct"
        v_x = float(v_x)
        v_y = height - 1 - float(v_y)  # Change of axis orientation
        vertices.append(shapely.geometry.Point((v_x, v_y)))
        line_i += 1
        vertex_i += 1

    start_edges = line_i + 1
    end_edges = start_edges + edge_count
    for i, line in enumerate(lines[start_edges:end_edges]):
        e_id, e_v1, e_v2 = map(int, line.split(" "))
        assert i == e_id, "Index of edge should be correct"
        edges.append(shapely.geometry.LineString((vertices[e_v1], vertices[e_v2])))

    assert lines[end_edges] == 'faces', "Line before faces should read 'faces'"
    start_faces = end_edges + 1
    line_i = start_faces
    face_i = 0
    while face_i < face_count:
        f_id, contour_count = map(int, lines[line_i].split(" "))
        assert f_id == face_i, "Index of face should be correct"
        line_i += 1  # Start reading contours:
        for contour_i in range(contour_count):
            edge_info_list = lines[line_i].split("  ")[:-1]
            for edge_info in edge_info_list:
                e, placeholder = map(int, edge_info.split(" "))  # TODO: what is placeholder?
                all_face_edges_list.append(edges[e])
            line_i += 1
        face_i += 1

    merged_lines = shapely.ops.unary_union(all_face_edges_list)
    polygons = list(shapely.ops.polygonize(merged_lines))

    return polygons


def single_txt_to_json(filename, txt_dirpath, mask_dirpath, mask_filename_format):
    image_id = int(os.path.splitext(filename)[0])

    # Load .txt
    file = open(os.path.join(txt_dirpath, filename), mode='r')
    txt = file.read()
    file.close()

    try:
        polygons = parse_txt(txt)
    except ValueError as e:
        print("image_id:", image_id, e)
        return None
    except AssertionError as e:
        print("image_id:", image_id, e)
        return None
    except IndexError as e:
        print("image_id:", image_id, e)
        return None

    # Load mask
    mask_filename = mask_filename_format.format(image_id)
    mask = skimage.io.imread(os.path.join(mask_dirpath, mask_filename)) / 255

    # Remove low prob polygons
    filtered_polygons = []
    filtered_polygon_probs = []
    for polygon in polygons:
        prob = polygonize_utils.compute_geom_prob(polygon, mask)
        # print("simple:", np_indicator.min(), np_indicator.mean(), np_indicator.max(), prob)
        if 0.5 < prob:
            filtered_polygons.append(polygon)
            filtered_polygon_probs.append(prob)

    # plot(300, 300, filtered_polygons, mask)

    annotations = save_utils.poly_coco(filtered_polygons, filtered_polygon_probs, image_id)
    return annotations


def txt_to_json(txt_dirpath, mask_dirpath, output_filepath, mask_filename_format="{:012d}.png"):
    filenames = fnmatch.filter(os.listdir(txt_dirpath), "*.txt")

    with multiprocessing.Pool() as pool:
        all_annotations = list(tqdm(pool.imap(functools.partial(single_txt_to_json, txt_dirpath=txt_dirpath, mask_dirpath=mask_dirpath, mask_filename_format=mask_filename_format), filenames), desc="Process txt files", total=len(filenames)))
    all_annotations = [annotation for annotations in all_annotations if annotations is not None for annotation in annotations]

    with open(output_filepath, 'w') as outfile:
        json.dump(all_annotations, outfile)


if __name__ == "__main__":
    args = get_args()
    txt_dirpath = args.txt_dirpath
    mask_dirpath = args.mask_dirpath
    output_filepath = args.output_filepath
    txt_to_json(txt_dirpath, mask_dirpath, output_filepath)
