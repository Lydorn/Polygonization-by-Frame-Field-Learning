#!/usr/bin/env python3

###################################################################
# Use this script to clean COCO mask detection. It merges detections that touch each other taking into account their scores.
# Example use:
# python clean_coco.py --gt_filepath /home/lydorn/data/mapping_challenge_dataset/raw/val/annotation.json --in_filepath "/home/lydorn/data/mapping_challenge_dataset/eval_runs/mapping_dataset.open_solution_full | 0000-00-00 00:00:00/test.annotation.seg.json" --out_filepath "/home/lydorn/data/mapping_challenge_dataset/eval_runs/mapping_dataset.open_solution_full | 0000-00-00 00:00:00/test.annotation.seg_cleaned.json"
###################################################################

import argparse
from pycocotools.coco import COCO
from pycocotools import mask as cocomask
from multiprocess import Pool
import numpy as np

import json
from tqdm import tqdm

import torch

from frame_field_learning import inference, polygonize_acm, save_utils


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '--gt_filepath',
        required=True,
        type=str,
        help='Filepath of the ground truth annotations in COCO format (.json file).')
    argparser.add_argument(
        '--in_filepath',
        required=True,
        type=str,
        help='Filepath of the input mask annotations in COCO format (.json file).')
    argparser.add_argument(
        '--out_filepath',
        required=True,
        type=str,
        help='Filepath of the output polygon annotations in COCO format (.json file).')
    args = argparser.parse_args()
    return args


def clean_one(im_data):
    img, dts = im_data
    seg_image = np.zeros((img["height"], img["width"]), dtype=np.float)
    weights_image = np.zeros((img["height"], img["width"]), dtype=np.float)
    for dt in dts:
        if 0 < dt["score"]:
            dt_mask = cocomask.decode(dt["segmentation"])
            dt_seg = dt_mask * dt["score"]
            seg_image += dt_seg
            weights_image += dt_mask
    weights_image[weights_image == 0] = 1  # Avoid dividing by zero:
    seg_image /= weights_image  # Avg aggr
    mask_image = 0.5 < seg_image

    sample = {
        "seg": torch.tensor(seg_image[None, :, :]),
        "seg_mask": torch.tensor(mask_image).int(),
        "image_id": torch.tensor(img["id"])
    }
    coco_ann = save_utils.seg_coco(sample)
    return coco_ann


def clean_masks(gt_filepath, in_filepath, out_filepath):
    coco_gt = COCO(gt_filepath)
    coco_dt = coco_gt.loadRes(in_filepath)

    # --- Clean input COCO mask detections --- #
    img_ids = sorted(coco_dt.getImgIds())
    im_data_list = []
    for img_id in img_ids:
        img = coco_gt.loadImgs(img_id)[0]
        dts = coco_dt.loadAnns(coco_dt.getAnnIds(imgIds=img_id))
        im_data_list.append((img, dts))

    pool = Pool()
    output_annotations_list = list(tqdm(pool.imap(clean_one, im_data_list), desc="Clean detections", total=len(im_data_list)))
    output_annotations = [output_annotation for output_annotations in output_annotations_list for output_annotation in output_annotations]

    print("Saving output...")
    with open(out_filepath, 'w') as outfile:
        json.dump(output_annotations, outfile)


if __name__ == "__main__":
    args = get_args()
    clean_masks(args.gt_filepath, args.in_filepath, args.out_filepath)
