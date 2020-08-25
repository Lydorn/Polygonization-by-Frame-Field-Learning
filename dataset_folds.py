import functools

import torch
import torch.utils.data

from frame_field_learning import data_transforms
from lydorn_utils import print_utils


def inria_aerial_train_tile_filter(tile, train_val_split_point):
    return tile["number"] <= train_val_split_point


def inria_aerial_val_tile_filter(tile, train_val_split_point):
    return train_val_split_point < tile["number"]


def get_inria_aerial_folds(config, root_dir, folds):
    from torch_lydorn.torchvision.datasets import InriaAerial

    # --- Online transform done on the host (CPU):
    online_cpu_transform = data_transforms.get_online_cpu_transform(config,
                                                                    augmentations=config["data_aug_params"]["enable"])
    mask_only = config["dataset_params"]["mask_only"]
    kwargs = {
        "pre_process": config["dataset_params"]["pre_process"],
        "transform": online_cpu_transform,
        "patch_size": config["dataset_params"]["data_patch_size"],
        "patch_stride": config["dataset_params"]["input_patch_size"],
        "pre_transform": data_transforms.get_offline_transform_patch(distances=not mask_only, sizes=not mask_only),
        "small": config["dataset_params"]["small"],
        "pool_size": config["num_workers"],
        "gt_source": config["dataset_params"]["gt_source"],
        "gt_type": config["dataset_params"]["gt_type"],
        "gt_dirname": config["dataset_params"]["gt_dirname"],
        "mask_only": mask_only,
    }
    train_val_split_point = config["dataset_params"]["train_fraction"] * 36
    partial_train_tile_filter = functools.partial(inria_aerial_train_tile_filter, train_val_split_point=train_val_split_point)
    partial_val_tile_filter = functools.partial(inria_aerial_val_tile_filter, train_val_split_point=train_val_split_point)

    ds_list = []
    for fold in folds:
        if fold == "train":
            ds = InriaAerial(root_dir, fold="train", tile_filter=partial_train_tile_filter, **kwargs)
            ds_list.append(ds)
        elif fold == "val":
            ds = InriaAerial(root_dir, fold="train", tile_filter=partial_val_tile_filter, **kwargs)
            ds_list.append(ds)
        elif fold == "train_val":
            ds = InriaAerial(root_dir, fold="train", **kwargs)
            ds_list.append(ds)
        elif fold == "test":
            ds = InriaAerial(root_dir, fold="test", **kwargs)
            ds_list.append(ds)
        else:
            print_utils.print_error("ERROR: fold \"{}\" not recognized, implement it in dataset_folds.py.".format(fold))

    return ds_list


def get_luxcarta_buildings(config, root_dir, folds):
    from torch_lydorn.torchvision.datasets import LuxcartaBuildings

    # --- Online transform done on the host (CPU):
    online_cpu_transform = data_transforms.get_online_cpu_transform(config,
                                                                    augmentations=config["data_aug_params"]["enable"])

    data_patch_size = config["dataset_params"]["data_patch_size"] if config["data_aug_params"]["enable"] else config[
        "input_patch_size"]
    ds = LuxcartaBuildings(root_dir,
                           transform=online_cpu_transform,
                           patch_size=data_patch_size,
                           patch_stride=config["dataset_params"]["input_patch_size"],
                           pre_transform=data_transforms.get_offline_transform_patch(),
                           fold="train",
                           pool_size=config["num_workers"])
    torch.manual_seed(config["dataset_params"]["seed"])  # Ensure a seed is set
    train_split_length = int(round(config["dataset_params"]["train_fraction"] * len(ds)))
    val_split_length = len(ds) - train_split_length
    train_ds, val_ds = torch.utils.data.random_split(ds, [train_split_length, val_split_length])

    ds_list = []
    for fold in folds:
        if fold == "train":
            ds_list.append(train_ds)
        elif fold == "val":
            ds_list.append(val_ds)
        elif fold == "test":
            # TODO: handle patching with multi-GPU processing
            print_utils.print_error("WARNING: handle patching with multi-GPU processing")
            ds = LuxcartaBuildings(root_dir,
                                   transform=online_cpu_transform,
                                   pre_transform=data_transforms.get_offline_transform_patch(),
                                   fold="test",
                                   pool_size=config["num_workers"])
            ds_list.append(ds)
        else:
            print_utils.print_error("ERROR: fold \"{}\" not recognized, implement it in dataset_folds.py.".format(fold))

    return ds_list


def get_mapping_challenge(config, root_dir, folds):
    from torch_lydorn.torchvision.datasets import MappingChallenge

    if "train" in folds or "val" in folds or "train_val" in folds:
        train_online_cpu_transform = data_transforms.get_online_cpu_transform(config,
                                                                              augmentations=config["data_aug_params"][
                                                                                  "enable"])
        ds = MappingChallenge(root_dir,
                              transform=train_online_cpu_transform,
                              pre_transform=data_transforms.get_offline_transform_patch(),
                              small=config["dataset_params"]["small"],
                              fold="train",
                              pool_size=config["num_workers"])
        torch.manual_seed(config["dataset_params"]["seed"])  # Ensure a seed is set
        train_split_length = int(round(config["dataset_params"]["train_fraction"] * len(ds)))
        val_split_length = len(ds) - train_split_length
        train_ds, val_ds = torch.utils.data.random_split(ds, [train_split_length, val_split_length])

    ds_list = []
    for fold in folds:
        if fold == "train":
            ds_list.append(train_ds)
        elif fold == "val":
            ds_list.append(val_ds)
        elif fold == "train_val":
            ds_list.append(ds)
        elif fold == "test":
            # The val fold from the original challenge is used as test here
            # because we don't have the ground truth for the test_images fold of the challenge:
            test_online_cpu_transform = data_transforms.get_eval_online_cpu_transform()
            test_ds = MappingChallenge(root_dir,
                                       transform=test_online_cpu_transform,
                                       pre_transform=data_transforms.get_offline_transform_patch(),
                                       small=config["dataset_params"]["small"],
                                       fold="val",
                                       pool_size=config["num_workers"])
            ds_list.append(test_ds)
        else:
            print_utils.print_error("ERROR: fold \"{}\" not recognized, implement it in dataset_folds.py.".format(fold))
            exit()

    return ds_list


def get_opencities_competition(config, root_dir, folds):
    from torch_lydorn.torchvision.datasets import RasterizedOpenCities, OpenCitiesTestDataset

    data_patch_size = config["dataset_params"]["data_patch_size"] if config["data_aug_params"]["enable"] else config[
        "input_patch_size"]

    ds_list = []
    for fold in folds:
        if fold == "train":
            train_ds = RasterizedOpenCities(tier=1, augment=False, small_subset=False, resize_size=data_patch_size,
                                            data_dir=root_dir, baseline_mode=False, val=False,
                                            val_split=config["dataset_params"]["val_fraction"])
            ds_list.append(train_ds)
        elif fold == "val":
            val_ds = RasterizedOpenCities(tier=1, augment=False, small_subset=False, resize_size=data_patch_size,
                                          data_dir=root_dir, baseline_mode=False, val=True,
                                          val_split=config["dataset_params"]["val_fraction"])
            ds_list.append(val_ds)
        elif fold == "test":
            test_ds = OpenCitiesTestDataset(root_dir + "/test/", 1024)
            ds_list.append(test_ds)
        else:
            print_utils.print_error("ERROR: fold \"{}\" not recognized, implement it in dataset_folds.py.".format(fold))

    return ds_list


def get_xview2_dataset(config, root_dir, folds):
    from torch_lydorn.torchvision.datasets import xView2Dataset

    if "train" in folds or "val" in folds or "train_val" in folds:
        train_online_cpu_transform = data_transforms.get_online_cpu_transform(config,
                                                                              augmentations=config["data_aug_params"][
                                                                                  "enable"])
        ds = xView2Dataset(root_dir, fold="train", pre_process=True,
                           patch_size=config["dataset_params"]["data_patch_size"],
                           pre_transform=data_transforms.get_offline_transform_patch(),
                           transform=train_online_cpu_transform,
                           small=config["dataset_params"]["small"], pool_size=config["num_workers"])
        torch.manual_seed(config["dataset_params"]["seed"])  # Ensure a seed is set
        train_split_length = int(round(config["dataset_params"]["train_fraction"] * len(ds)))
        val_split_length = len(ds) - train_split_length
        train_ds, val_ds = torch.utils.data.random_split(ds, [train_split_length, val_split_length])

    ds_list = []
    for fold in folds:
        if fold == "train":
            ds_list.append(train_ds)
        elif fold == "val":
            ds_list.append(val_ds)
        elif fold == "train_val":
            ds_list.append(ds)
        elif fold == "test":
            raise NotImplementedError("Test fold not yet implemented (skip pre-processing?)")
        elif fold == "hold":
            raise NotImplementedError("Hold fold not yet implemented (skip pre-processing?)")
        else:
            print_utils.print_error("ERROR: fold \"{}\" not recognized, implement it in dataset_folds.py.".format(fold))
            exit()

    return ds_list


def get_folds(config, root_dir, folds):
    assert set(folds).issubset({"train", "val", "train_val", "test"}), \
        'fold in folds should be in ["train", "val", "train_val", "test"]'

    if config["dataset_params"]["root_dirname"] == "AerialImageDataset":
        return get_inria_aerial_folds(config, root_dir, folds)

    elif config["dataset_params"]["root_dirname"] == "luxcarta_precise_buildings":
        return get_luxcarta_buildings(config, root_dir, folds)

    elif config["dataset_params"]["root_dirname"] == "mapping_challenge_dataset":
        return get_mapping_challenge(config, root_dir, folds)

    elif config["dataset_params"]["root_dirname"] == "segbuildings":
        return get_opencities_competition(config, root_dir, folds)

    elif config["dataset_params"]["root_dirname"] == "xview2_xbd_dataset":
        return get_xview2_dataset(config, root_dir, folds)

    else:
        print_utils.print_error("ERROR: config[\"data_root_partial_dirpath\"] = \"{}\" is an unknown dataset! "
                                "If it is a new dataset, add it in dataset_folds.py's get_folds() function.".format(
            config["dataset_params"]["root_dirname"]))
        exit()
