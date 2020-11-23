#!/usr/bin/env python3

###################################################################
# Use this script to convert shapefiles to the COCO .json format.
###################################################################

import fnmatch
import os
import argparse
import fiona
import shapely.geometry


import skimage.io
import skimage.morphology
import pycocotools.mask
import numpy as np
from tqdm import tqdm
import json
import skimage.measure


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '--shp_dirpath',
        required=True,
        type=str,
        help='Path to the directory where the shapefiles are.')
    argparser.add_argument(
        '--output_filepath',
        required=True,
        type=str,
        help='Filepath of the final .json.')
    args = argparser.parse_args()
    return args


def shp_to_json(shp_dirpath, output_filepath):
    filenames = fnmatch.filter(os.listdir(shp_dirpath), "*.shp")
    filenames = sorted(filenames)

    annotations = []
    for filename in tqdm(filenames, desc="Process shapefiles:"):
        shapefile = fiona.open(os.path.join(shp_dirpath, filename))

        polygons = []
        for feature in shapefile:
            geometry = shapely.geometry.shape(feature["geometry"])
            if geometry.type == "MultiPolygon":
                for polygon in geometry.geoms:
                    polygons.append(polygon)
            elif geometry.type == "Polygon":
                polygons.append(geometry)
            else:
                raise TypeError(f"geometry.type should be either Polygon or MultiPolygon, not {geometry.type}.")

        image_id = int(filename.split(".")[0])
        for polygon in polygons:
            bbox = np.round([polygon.bounds[0], polygon.bounds[1],
                             polygon.bounds[2] - polygon.bounds[0], polygon.bounds[3] - polygon.bounds[1]], 2)
            contour = np.array(polygon.exterior.coords)
            contour[:, 1] *= -1  # Shapefiles have inverted y axis...
            exterior = list(np.round(contour.reshape(-1), 2))
            segmentation = [exterior]
            annotation = {
                "category_id": 100,  # Building
                "bbox": list(bbox),
                "segmentation": segmentation,
                "score": 1.0,
                "image_id": image_id}
            annotations.append(annotation)

        # seg = skimage.io.imread()
        # labels = skimage.morphology.label(seg)
        # properties = skimage.measure.regionprops(labels, cache=True)
        # for i, contour_props in enumerate(properties):
        #     skimage_bbox = contour_props["bbox"]
        #     coco_bbox = [skimage_bbox[1], skimage_bbox[0],
        #                  skimage_bbox[3] - skimage_bbox[1], skimage_bbox[2] - skimage_bbox[0]]
        #
        #     image_mask = labels == (i + 1)  # The mask has to span the whole image
        #     rle = pycocotools.mask.encode(np.asfortranarray(image_mask))
        #     rle["counts"] = rle["counts"].decode("utf-8")
        #     annotation = {
        #         "category_id": 100,  # Building
        #         "bbox": coco_bbox,
        #         "segmentation": rle,
        #         "score": 1.0,
        #         "image_id": image_id}
        #     annotations.append(annotation)

    with open(output_filepath, 'w') as outfile:
        json.dump(annotations, outfile)


if __name__ == "__main__":
    args = get_args()
    shp_dirpath = args.shp_dirpath
    output_filepath = args.output_filepath
    shp_to_json(shp_dirpath, output_filepath)
