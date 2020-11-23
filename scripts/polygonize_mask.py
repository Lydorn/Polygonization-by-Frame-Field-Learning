#!/usr/bin/env python3

###################################################################
# Use this script to extract polygons from binary masks using a model trained for this task.
# I used it to polygonize the original ground truth masks from the Inria Aerial Image Labeling Dataset.
# The first step is to train a network whose input is a binary mask and output is a segmentation + frame field.
# I did this on rasterized OSM annotation corresponding to the Inria dataset
# (so that there is a ground truth for the frame field).
###################################################################


import argparse
import sys
import os
import numpy as np
import torch_lydorn
from tqdm import tqdm
import skimage.io
import torch

try:
    __import__("frame_field_learning.local_utils")
except ImportError:
    print("ERROR: The frame_field_learning package is not installed! "
          "Execute script setup.sh to install local dependencies such as frame_field_learning in develop mode.")
    exit()

from frame_field_learning import data_transforms, polygonize_asm, save_utils, polygonize_acm, measures
from frame_field_learning.model import FrameFieldModel
from frame_field_learning import inference
from frame_field_learning import local_utils

from torch_lydorn import torchvision
from lydorn_utils import run_utils, geo_utils, polygon_utils
from lydorn_utils import print_utils

from backbone import get_backbone

# polygonize_config = {
#     "data_level": 0.5,
#     "step_thresholds": [0, 500],  # From 0 to 500: gradually go from coefs[0] to coefs[1]
#     "data_coefs": [0.9, 0.09],
#     "length_coefs": [0.1, 0.01],
#     "crossfield_coefs": [0.0, 0.05],
#     "poly_lr": 0.1,
#     "device": "cuda",
#     "tolerance": 0.001,
#     "seg_threshold": 0.5,
#     "min_area": 10,
# }
polygonize_config = {
    "steps": 500,
    "data_level": 0.5,
    "data_coef": 0.1,
    "length_coef": 0.4,
    "crossfield_coef": 0.5,
    "poly_lr": 0.01,
    "warmup_iters": 100,
    "warmup_factor": 0.1,
    "device": "cuda",
    "tolerance": 0.5,
    "seg_threshold": 0.5,
    "min_area": 10
}


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-f', '--filepath',
        required=True,
        type=str,
        nargs='*',
        help='Filepaths to the binary images to polygonize.')

    argparser.add_argument(
        '-r', '--runs_dirpath',
        default="runs",
        type=str,
        help='Directory where runs are recorded (model saves and logs).')

    argparser.add_argument(
        '--run_name',
        required=True,
        type=str,
        help='Name of the run to use for predicting the frame field needed by the polygonization algorithm.'
             'That name does not include the timestamp of the folder name: <run_name> | <yyyy-mm-dd hh:mm:ss>.')
    argparser.add_argument(
        '--eval_patch_size',
        type=int,
        help='When evaluating, patch size the tile split into.')
    argparser.add_argument(
        '--eval_patch_overlap',
        type=int,
        help='When evaluating, patch the tile with the specified overlap to reduce edge artifacts when reconstructing '
             'the whole tile')
    argparser.add_argument(
        '--out_ext',
        type=str,
        default="geojson",
        choices=['geojson', 'shp'],
        help="File extension of the output geometry. 'geojson': GeoJSON,  'shp': shapefile")

    args = argparser.parse_args()
    return args


def polygonize_mask(config, mask_filepaths, backbone, out_ext):
    """
    Reads
    @param args:
    @return:
    """

    # --- Online transform performed on the device (GPU):
    eval_online_cuda_transform = data_transforms.get_eval_online_cuda_transform(config)

    print("Loading model...")
    model = FrameFieldModel(config, backbone=backbone, eval_transform=eval_online_cuda_transform)
    model.to(config["device"])
    checkpoints_dirpath = run_utils.setup_run_subdir(config["eval_params"]["run_dirpath"],
                                                     config["optim_params"]["checkpoints_dirname"])
    model = inference.load_checkpoint(model, checkpoints_dirpath, config["device"])
    model.eval()

    rasterizer = torch_lydorn.torchvision.transforms.Rasterize(fill=True, edges=False, vertices=False)

    # Read image
    pbar = tqdm(mask_filepaths, desc="Infer images")
    for mask_filepath in pbar:
        pbar.set_postfix(status="Loading mask image")
        mask_image = skimage.io.imread(mask_filepath)

        input_image = mask_image
        if len(input_image.shape) == 2:
            # Make input_image shape (H, W, 1)
            input_image = input_image[:, :, None]
        if input_image.shape[2] == 1:
            input_image = np.repeat(input_image, 3, axis=-1)
        mean = np.array([0.5, 0.5, 0.5])
        std = np.array([1, 1, 1])
        tile_data = {
            "image": torchvision.transforms.functional.to_tensor(input_image)[None, ...],
            "image_mean": torch.from_numpy(mean)[None, ...],
            "image_std": torch.from_numpy(std)[None, ...],
            "image_filepath": [mask_filepath],
        }

        pbar.set_postfix(status="Inference")
        tile_data = inference.inference(config, model, tile_data, compute_polygonization=False)

        pbar.set_postfix(status="Polygonize")
        seg_batch = torchvision.transforms.functional.to_tensor(mask_image)[None, ...].float() / 255
        crossfield_batch = tile_data["crossfield"]
        polygons_batch, probs_batch = polygonize_acm.polygonize(seg_batch, crossfield_batch, polygonize_config)
        tile_data["polygons"] = polygons_batch
        tile_data["polygon_probs"] = probs_batch

        pbar.set_postfix(status="Saving output")
        tile_data = local_utils.batch_to_cpu(tile_data)
        tile_data = local_utils.split_batch(tile_data)[0]
        base_filepath = os.path.splitext(mask_filepath)[0]
        # save_utils.save_polygons(tile_data["polygons"], base_filepath, "polygons", tile_data["image_filepath"])
        # save_utils.save_poly_viz(tile_data["image"], tile_data["polygons"], tile_data["polygon_probs"], base_filepath, name)
        # geo_utils.save_shapefile_from_shapely_polygons(tile_data["polygons"], mask_filepath, base_filepath + "." + name + ".shp")

        if out_ext == "geojson":
            save_utils.save_geojson(tile_data["polygons"], base_filepath)
        elif out_ext == "shp":
            save_utils.save_shapefile(tile_data["polygons"], base_filepath, "polygonized", mask_filepath)
        else:
            raise ValueError(f"out_ext '{out_ext}' invalid!")

        # --- Compute IoU of mask image and extracted polygons
        polygons_raster = rasterizer(mask_image, tile_data["polygons"])[:, :, 0]
        mask = 128 < mask_image
        polygons_mask = 128 < polygons_raster
        iou = measures.iou(torch.tensor(polygons_mask).view(1, -1), torch.tensor(mask).view(1, -1), threshold=0.5)
        print("IoU:", iou.item())
        if iou < 0.9:
            print(mask_filepath)


def main():
    torch.manual_seed(0)
    # --- Process args --- #
    args = get_args()

    # --- Setup run --- #
    run_dirpath = local_utils.get_run_dirpath(args.runs_dirpath, args.run_name)
    # Load run's config file:
    config = run_utils.load_config(config_dirpath=run_dirpath)
    if config is None:
        print_utils.print_error(
            "ERROR: cannot continue without a config file. Exiting now...")
        sys.exit()

    config["eval_params"]["run_dirpath"] = run_dirpath
    if args.eval_patch_size is not None:
        config["eval_params"]["patch_size"] = args.eval_patch_size
    if args.eval_patch_overlap is not None:
        config["eval_params"]["patch_overlap"] = args.eval_patch_overlap

    backbone = get_backbone(config["backbone_params"])

    polygonize_mask(config, args.filepath, backbone, args.out_ext)


if __name__ == '__main__':
    main()
