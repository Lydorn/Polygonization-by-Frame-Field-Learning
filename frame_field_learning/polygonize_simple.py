import argparse

import os
import torch

import numpy as np
import skimage
import skimage.measure
import skimage.io
import shapely.geometry
import shapely.ops
from PIL import Image
from multiprocess import Pool
from tqdm import tqdm

from functools import partial

from lydorn_utils import print_utils, geo_utils

from frame_field_learning import polygonize_utils, plot_utils

DEBUG = False


def debug_print(s: str):
    if DEBUG:
        print_utils.print_debug(s)


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '--seg_filepath',
        required=True,
        nargs='*',
        type=str,
        help='Filepath(s) to input segmentation/mask image.')
    argparser.add_argument(
        '--im_dirpath',
        required=True,
        type=str,
        help='Path to the directory containing the corresponding images os the segmentation/mask. '
             'Files must have the same filename as --seg_filepath.'
             'Used for vizualization or saving the shapefile with the same coordinate system as that image.')
    argparser.add_argument(
        '--out_dirpath',
        required=True,
        type=str,
        help='Path to the output directory.')
    argparser.add_argument(
        '--out_ext',
        type=str,
        default="shp",
        choices=['pdf', 'shp'],
        help="File extension of the output. "
             "'pdf': pdf visualization (requires --im_dirpath for the image),  'shp': shapefile")
    argparser.add_argument(
        '--bbox',
        nargs='*',
        type=int,
        help='Selects area in bbox for computation.')

    args = argparser.parse_args()
    return args


def simplify(polygons, probs, tolerance):
    if type(tolerance) == list:
        out_polygons_dict = {}
        out_probs_dict = {}
        for tol in tolerance:
            out_polygons, out_probs = simplify(polygons, probs, tol)
            out_polygons_dict["tol_{}".format(tol)] = out_polygons
            out_probs_dict["tol_{}".format(tol)] = out_probs
        return out_polygons_dict, out_probs_dict
    else:
        out_polygons = [polygon.simplify(tolerance, preserve_topology=True) for polygon in polygons]
        return out_polygons, probs


def shapely_postprocess(out_contours, np_indicator, config):
    height = np_indicator.shape[0]
    width = np_indicator.shape[1]

    # Handle holes:
    line_string_list = [shapely.geometry.LineString(out_contour[:, ::-1]) for out_contour in out_contours]

    # Add image boundary line_strings for border polygons
    line_string_list.append(
        shapely.geometry.LinearRing([
            (0, 0),
            (0, height - 1),
            (width - 1, height - 1),
            (width - 1, 0),
        ]))

    # Merge polylines (for border polygons):
    multi_line_string = shapely.ops.unary_union(line_string_list)

    # Find polygons:
    polygons, dangles, cuts, invalids = shapely.ops.polygonize_full(multi_line_string)

    polygons = list(polygons)

    # Remove small polygons
    polygons = [polygon for polygon in polygons if
                config["min_area"] < polygon.area]

    # Remove low prob polygons
    filtered_polygons = []
    filtered_polygon_probs = []
    for polygon in polygons:
        prob = polygonize_utils.compute_geom_prob(polygon, np_indicator)
        # print("simple:", np_indicator.min(), np_indicator.mean(), np_indicator.max(), prob)
        if config["seg_threshold"] < prob:
            filtered_polygons.append(polygon)
            filtered_polygon_probs.append(prob)

    polygons, probs = simplify(filtered_polygons, filtered_polygon_probs, config["tolerance"])

    return polygons, probs


def polygonize(seg_batch, config, pool=None, pre_computed=None):
    # tic_total = time.time()

    assert len(seg_batch.shape) == 4 and seg_batch.shape[
        1] <= 3, "seg_batch should be (N, C, H, W) with C <= 3, not {}".format(seg_batch.shape)

    # Indicator
    # tic = time.time()
    indicator_batch = seg_batch[:, 0, :, :]
    np_indicator_batch = indicator_batch.cpu().numpy()
    # toc = time.time()
    # debug_print(f"Indicator to cpu: {toc - tic}s")

    if pre_computed is None or "init_contours_batch" not in pre_computed:
        # tic = time.time()
        init_contours_batch = polygonize_utils.compute_init_contours_batch(np_indicator_batch, config["data_level"], pool=pool)
        # toc = time.time()
        # debug_print(f"Init contours: {toc - tic}s")
    else:
        init_contours_batch = pre_computed["init_contours_batch"]

    # tic = time.time()
    # Convert contours to shapely polygons to handle holes:
    if pool is not None:
        shapely_postprocess_partial = partial(shapely_postprocess, config=config)
        polygons_probs_batch = pool.starmap(shapely_postprocess_partial, zip(init_contours_batch, np_indicator_batch))
        polygons_batch, probs_batch = zip(*polygons_probs_batch)
    else:
        polygons_batch = []
        probs_batch = []
        for i, out_contours in enumerate(init_contours_batch):
            polygons, probs = shapely_postprocess(out_contours, np_indicator_batch[i], config)
            polygons_batch.append(polygons)
            probs_batch.append(probs)

    # toc = time.time()
    # debug_print(f"Shapely post-process: {toc - tic}s")

    # toc_total = time.time()
    # debug_print(f"Total: {toc_total - tic_total}s")

    return polygons_batch, probs_batch


def run_one(seg_filepath, out_dirpath, config, im_dirpath, out_ext=None, bbox=None):
    filename = os.path.basename(seg_filepath)
    name = os.path.splitext(filename)[0]

    # Load image
    image = None
    im_filepath = os.path.join(im_dirpath, name + ".tif")
    if out_ext == "pdf":
        image = skimage.io.imread(im_filepath)

    # seg = skimage.io.imread(seg_filepath) / 255
    seg_img = Image.open(seg_filepath)
    seg = np.array(seg_img)
    if seg.dtype == np.uint8:
        seg = seg / 255
    elif seg.dtype == np.bool:
        seg = seg.astype(np.float)

    # Select bbox for dev
    if bbox is not None:
        assert len(bbox) == 4, "bbox should have 4 values"
        # bbox = [1440, 210, 1800, 650]  # vienna12
        # bbox = [2808, 2393, 3124, 2772]  # innsbruck19
        if image is not None:
            image = image[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        seg = seg[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        extra_name = ".bbox_{}_{}_{}_{}".format(*bbox)
    else:
        extra_name = ""

    # Convert to torch and add batch dim
    if len(seg.shape) < 3:
        seg = seg[:, :, None]
    seg_batch = torch.tensor(np.transpose(seg[:, :, :2], (2, 0, 1)), dtype=torch.float)[None, ...]

    out_contours_batch, out_probs_batch = polygonize(seg_batch, config)

    polygons = out_contours_batch[0]

    if out_ext == "shp":
        out_filepath = os.path.join(out_dirpath, name + ".shp")
        geo_utils.save_shapefile_from_shapely_polygons(polygons, im_filepath, out_filepath)
    elif out_ext == "pdf":
        base_filepath = os.path.splitext(seg_filepath)[0]
        filepath = base_filepath + extra_name + ".poly_simple.pdf"
        # plot_utils.save_poly_viz(image, polygons, filepath, linewidths=1, draw_vertices=True, color_choices=[[0, 1, 0, 1]])
        plot_utils.save_poly_viz(image, polygons, filepath, markersize=30, linewidths=1, draw_vertices=True)
    else:
        raise ValueError(f"out_ext should be shp or pdf, not {out_ext}")


def main():
    config = {
        "data_level": 0.5,
        "tolerance": 1.0,
        "seg_threshold": 0.5,
        "min_area": 10
    }
    # --- Process args --- #
    args = get_args()

    pool = Pool()
    list(tqdm(pool.imap(partial(run_one, out_dirpath=args.out_dirpath, config=config, im_dirpath=args.im_dirpath, out_ext=args.out_ext, bbox=args.bbox), args.seg_filepath), desc="Simple poly.", total=len(args.seg_filepath)))


if __name__ == '__main__':
    main()
