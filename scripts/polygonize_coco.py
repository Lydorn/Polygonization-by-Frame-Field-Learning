#!/usr/bin/env python3

###################################################################
# Use this script to convert .png masks from the Open Solution from the CrowdAI challenge:
# https://github.com/neptune-ai/open-solution-mapping-challenge
# to the COCO .json format
###################################################################

import fnmatch
import os
import argparse

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
        '--mask_dirpath',
        required=True,
        type=str,
        help='Path to the directory where the mask .png files are.')
    argparser.add_argument(
        '--output_filepath',
        required=True,
        type=str,
        help='Filepath of the final .json.')
    args = argparser.parse_args()
    return args


def masks_to_json(mask_dirpath, output_filepath):
    filenames = fnmatch.filter(os.listdir(mask_dirpath), "*.png")

    annotations = []
    for filename in tqdm(filenames, desc="Process masks:"):
        image_id = int(os.path.splitext(filename)[0])
        seg = skimage.io.imread(os.path.join(mask_dirpath, filename))
        labels = skimage.morphology.label(seg)
        properties = skimage.measure.regionprops(labels, cache=True)
        for i, contour_props in enumerate(properties):
            skimage_bbox = contour_props["bbox"]
            coco_bbox = [skimage_bbox[1], skimage_bbox[0],
                         skimage_bbox[3] - skimage_bbox[1], skimage_bbox[2] - skimage_bbox[0]]

            image_mask = labels == (i + 1)  # The mask has to span the whole image
            rle = pycocotools.mask.encode(np.asfortranarray(image_mask))
            rle["counts"] = rle["counts"].decode("utf-8")
            annotation = {
                "category_id": 100,  # Building
                "bbox": coco_bbox,
                "segmentation": rle,
                "score": 1.0,
                "image_id": image_id}
            annotations.append(annotation)

    with open(output_filepath, 'w') as outfile:
        json.dump(annotations, outfile)


if __name__ == "__main__":
    args = get_args()
    mask_dirpath = args.mask_dirpath
    output_filepath = args.output_filepath
    masks_to_json(mask_dirpath, output_filepath)
