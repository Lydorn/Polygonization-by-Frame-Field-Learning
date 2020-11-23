from functools import partial
from collections import Iterable

import shapely.geometry
import shapely.ops
import shapely.affinity
import numpy as np
import cv2

import skimage.measure

from lydorn_utils import print_utils


def compute_init_contours(np_indicator, level):
    assert isinstance(np_indicator, np.ndarray) and len(np_indicator.shape) == 2, "indicator should have shape (H, W)"
    # Using marching squares
    contours = skimage.measure.find_contours(np_indicator, level, fully_connected='low', positive_orientation='high')

    # Using OpenCV:
    # u, contours, _ = cv2.findContours((level < np_indicator).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # contours = [contour[:, 0, ::-1] for contour in contours]
    # contours = [np.concatenate((contour, contour[0:1, :]), axis=0) for contour in contours]

    # Using rasterio
    # import rasterio.features
    # shapes = rasterio.features.shapes((level < np_indicator).astype(np.uint8))
    # contours = []
    # for shape in shapes:
    #     for coords in shape[0]["coordinates"]:
    #         contours.append(np.array(coords)[:, ::-1] - 0.5)

    # Simplify contours a tiny bit:
    # contours = [skimage.measure.approximate_polygon(contour, tolerance=0.01) for contour in contours]
    return contours


def compute_init_contours_batch(np_indicator_batch, level, pool=None):
    post_process_partial = partial(compute_init_contours, level=level)
    if pool is not None:
        init_contours_batch = pool.map(post_process_partial, np_indicator_batch)
    else:
        init_contours_batch = list(map(post_process_partial, np_indicator_batch))
    return init_contours_batch


def split_polylines_corner(polylines, corner_masks):
    new_polylines = []
    for polyline, corner_mask in zip(polylines, corner_masks):
        splits, = np.where(corner_mask)
        if len(splits) == 0:
            new_polylines.append(polyline)
            continue
        slice_list = [(splits[i], splits[i+1] + 1) for i in range(len(splits) - 1)]
        for s in slice_list:
            new_polylines.append(polyline[s[0]:s[1]])
        # Possibly add a merged polyline if start and end vertices are not corners (or endpoints of open polylines)
        if ~corner_mask[0] and ~corner_mask[-1]:  # In fact any of those conditon should be enough
            new_polyline = np.concatenate([polyline[splits[-1]:], polyline[:splits[0] + 1]], axis=0)
            new_polylines.append(new_polyline)
    return new_polylines


def compute_geom_prob(geom, prob_map, output_debug=False):
    assert len(prob_map.shape) == 2, "prob_map should have size (H, W), not {}".format(prob_map.shape)

    if isinstance(geom, Iterable):
        return [compute_geom_prob(_geom, prob_map, output_debug=output_debug) for _geom in geom]
    elif isinstance(geom, shapely.geometry.Polygon):
        # --- Cut with geom bounds:
        minx, miny, maxx, maxy = geom.bounds
        minx = int(minx)
        miny = int(miny)
        maxx = int(maxx) + 1
        maxy = int(maxy) + 1
        geom = shapely.affinity.translate(geom, xoff=-minx, yoff=-miny)
        prob_map = prob_map[miny:maxy, minx:maxx]

        # --- Rasterize TODO: better rasterization (or sampling) of polygon ?
        raster = np.zeros(prob_map.shape, dtype=np.uint8)
        exterior_array = np.round(np.array(geom.exterior.coords)).astype(np.int32)
        interior_array_list = [np.round(np.array(interior.coords)).astype(np.int32) for interior in geom.interiors]
        cv2.fillPoly(raster, [exterior_array], color=1)
        cv2.fillPoly(raster, interior_array_list, color=0)

        raster_sum = np.sum(raster)
        if 0 < raster_sum:
            polygon_prob = np.sum(raster * prob_map) / raster_sum
        else:
            polygon_prob = 0
            if output_debug:
                print_utils.print_warning("WARNING: empty polygon raster in polygonize_tracing.compute_polygon_prob().")

        return polygon_prob
    else:
        raise NotImplementedError(f"Geometry of type {type(geom)} not implemented!")

