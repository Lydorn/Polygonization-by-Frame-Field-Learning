#!/usr/bin/env python3

###################################################################
# Use this script to polygonize binary mask detection in COCO format (for example frome the Open Solution from the CrowdAI challenge:
# https://github.com/neptune-ai/open-solution-mapping-challenge)
# using the frame field polygonization method and save the output in COCO format.
# Example use:
# python polygonize_coco.py --run_dirpath "/home/lydorn/repos/lydorn/Polygonization-by-Frame-Field-Learning/frame_field_learning/runs/mapping_dataset.unet_resnet101_pretrained.train_val | 2020-09-07 11:28:51" --images_dirpath "/home/lydorn/data/mapping_challenge_dataset/raw/val/images" --gt_filepath /home/lydorn/data/mapping_challenge_dataset/raw/val/annotation.json --in_filepath "/home/lydorn/data/mapping_challenge_dataset/eval_runs/mapping_dataset.open_solution | 0000-00-00 00:00:00/test.annotation.seg.json" --out_filepath "/home/lydorn/data/mapping_challenge_dataset/eval_runs/mapping_dataset.open_solution | 0000-00-00 00:00:00/test.annotation.poly.json"
###################################################################

import argparse
from pycocotools.coco import COCO
from pycocotools import mask as cocomask
from pycocotools.cocoeval import COCOeval
import numpy as np
import skimage.io
import matplotlib.pyplot as plt
import pylab
import random
import os
import json
import sys
from tqdm import tqdm

import torch

from frame_field_learning.model import FrameFieldModel
from frame_field_learning import inference, polygonize_acm, data_transforms, save_utils, polygonize_utils
from lydorn_utils import run_utils, print_utils, python_utils
from backbone import get_backbone
import torch_lydorn

pylab.rcParams['figure.figsize'] = (8.0, 10.0)

# Image stats for the open challenge dataset:
image_mean = [0.30483739, 0.35143595, 0.3973895]
image_std = [0.16362707, 0.15187606, 0.14273278]

polygonize_config = {
    "steps": 500,
    "data_level": 0.5,
    "data_coef": 0.1,
    "length_coef": 0.4,
    "crossfield_coef": 0.5,
    "poly_lr": 0.001,
    "warmup_iters": 499,
    "warmup_factor": 0.1,
    "device": "cuda",
    "tolerance": 0.125,
    "seg_threshold": 0.5,
    "min_area": 10
}


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '--run_dirpath',
        required=True,
        type=str,
        help='Full path to the run directory to use for frame field prediction (needed for frame field polygonization).')
    argparser.add_argument(
        '--images_dirpath',
        required=True,
        type=str,
        help='Path to the images directory to use for frame field prediction (needed for frame field polygonization).')
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
    argparser.add_argument(
        '--batch_size',
        default=16,
        type=int,
        help='Batch size for running inference on the model.')
    argparser.add_argument(
        '--batch_size_mult',
        default=64,
        type=int,
        help='Multiply batch_size by this factor for polygonization.')
    args = argparser.parse_args()
    return args


def list_to_batch(sample_data_list):
    tile_data = {}
    for key in sample_data_list[0].keys():
        if isinstance(sample_data_list[0][key], list):
            tile_data[key] = [item for _tile_data in sample_data_list for item in _tile_data[key]]
        elif isinstance(sample_data_list[0][key], torch.Tensor):
            tile_data[key] = torch.cat([_tile_data[key] for _tile_data in sample_data_list], dim=0)
        else:
            raise TypeError(f"Type {type(sample_data_list[0][key])} is not handled!")
    return tile_data


def run_model(config, model, sample_data_list):
    tile_data = list_to_batch(sample_data_list)
    tile_data = inference.inference(config, model, tile_data, compute_polygonization=False)
    return tile_data


def run_polygonization(sample_data_list):
    tile_data = list_to_batch(sample_data_list)
    # Polygonize input mask with predicted frame field
    seg_batch = tile_data["mask_image"]
    crossfield_batch = tile_data["crossfield"]

    polygons_batch, _ = polygonize_acm.polygonize(seg_batch, crossfield_batch, polygonize_config)
    # Discard the probs computed by polygonize(). They will be computed next using the score_image

    # Convert to COCO format
    coco_ann_list = []
    for polygons, img_id, score_image in zip(polygons_batch, tile_data["img_id"], tile_data["score_image"]):
        scores = polygonize_utils.compute_geom_prob(polygons, score_image[0, :, :].numpy())
        coco_ann = save_utils.poly_coco(polygons, scores, image_id=img_id)
        coco_ann_list.extend(coco_ann)

    return coco_ann_list


def polygonize_masks(run_dirpath, images_dirpath, gt_filepath, in_filepath, out_filepath, batch_size, batch_size_mult):
    coco_gt = COCO(gt_filepath)
    coco_dt = coco_gt.loadRes(in_filepath)

    # --- Load model --- #
    # Load run's config file:
    config = run_utils.load_config(config_dirpath=run_dirpath)
    if config is None:
        print_utils.print_error(
            "ERROR: cannot continue without a config file. Exiting now...")
        sys.exit()

    config["backbone_params"]["pretrained"] = False  # Don't load pretrained model
    backbone = get_backbone(config["backbone_params"])
    eval_online_cuda_transform = data_transforms.get_eval_online_cuda_transform(config)
    model = FrameFieldModel(config, backbone=backbone, eval_transform=eval_online_cuda_transform)
    model.to(config["device"])
    checkpoints_dirpath = run_utils.setup_run_subdir(run_dirpath,
                                                     config["optim_params"]["checkpoints_dirname"])
    model = inference.load_checkpoint(model, checkpoints_dirpath, config["device"])
    model.eval()

    # --- Polygonize input COCO mask detections --- #
    img_ids = coco_dt.getImgIds()
    # img_ids = sorted(img_ids)[:1]  # TODO: rm limit
    output_annotations = []

    model_data_list = []  # Used to accumulate inputs and run model inference on it.
    poly_data_list = []  # Used to accumulate inputs and run polygonization on it.
    for img_id in tqdm(img_ids, desc="Polygonizing"):
        # Load image
        img = coco_gt.loadImgs(img_id)[0]
        image = skimage.io.imread(os.path.join(images_dirpath, img["file_name"]))

        # Draw mask from input COCO mask annotations
        mask_image = np.zeros((img["height"], img["width"]))
        score_image = np.zeros((img["height"], img["width"]))
        dts = coco_dt.loadAnns(coco_dt.getAnnIds(imgIds=img_id))
        for dt in dts:
            dt_mask = cocomask.decode(dt["segmentation"])
            mask_image = np.maximum(mask_image, dt_mask)
            score_image = np.maximum(score_image, dt_mask * dt["score"])

        # Accumulate inputs into the current batch
        sample_data = {
            "img_id": [img_id],
            "mask_image": torch_lydorn.torchvision.transforms.functional.to_tensor(mask_image)[None, ...].float(),
            "score_image": torch_lydorn.torchvision.transforms.functional.to_tensor(score_image)[None, ...].float(),
            "image": torch_lydorn.torchvision.transforms.functional.to_tensor(image)[None, ...],
            "image_mean": torch.tensor(image_mean)[None, ...],
            "image_std": torch.tensor(image_std)[None, ...]
        }
        # Accumulate batch for running the model
        model_data_list.append(sample_data)
        if len(model_data_list) == batch_size:
            # Run model
            tile_data = run_model(config, model, model_data_list)
            model_data_list = []  # Empty model batch

            # Accumulate batch for running the polygonization
            poly_data_list.append(tile_data)
            if len(poly_data_list) == batch_size_mult:
                coco_ann = run_polygonization(poly_data_list)
                output_annotations.extend(coco_ann)
                poly_data_list = []
    # Finish with incomplete batches
    if len(model_data_list):
        tile_data = run_model(config, model, model_data_list)
        poly_data_list.append(tile_data)
    if len(poly_data_list):
        coco_ann = run_polygonization(poly_data_list)
        output_annotations.extend(coco_ann)

    print("Saving output...")
    with open(out_filepath, 'w') as outfile:
        json.dump(output_annotations, outfile)


if __name__ == "__main__":
    args = get_args()
    polygonize_masks(args.run_dirpath, args.images_dirpath, args.gt_filepath, args.in_filepath, args.out_filepath, args.batch_size, args.batch_size_mult)
