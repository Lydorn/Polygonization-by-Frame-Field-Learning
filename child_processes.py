import os

import torch

from lydorn_utils import python_utils
from lydorn_utils import print_utils

from backbone import get_backbone
from dataset_folds import get_folds


def train_process(gpu, config, shared_dict, barrier):
    from frame_field_learning.train import train

    print_utils.print_info("GPU {} -> Ready. There are {} GPU(s) available on this node.".format(gpu, torch.cuda.device_count()))

    torch.manual_seed(0)  # Ensure same seed for all processes
    # --- Find data directory --- #
    root_dir_candidates = [os.path.join(data_dirpath, config["dataset_params"]["root_dirname"]) for data_dirpath in config["data_dir_candidates"]]
    root_dir, paths_tried = python_utils.choose_first_existing_path(root_dir_candidates, return_tried_paths=True)
    if root_dir is None:
        print_utils.print_error("GPU {} -> ERROR: Data root directory amongst \"{}\" not found!".format(gpu, paths_tried))
        exit()
    print_utils.print_info("GPU {} -> Using data from {}".format(gpu, root_dir))

    # --- Get dataset splits
    # - CHANGE HERE TO ADD YOUR OWN DATASET
    # We have to adapt the config["fold"] param to the folds argument of the get_folds function
    fold = set(config["fold"])
    if fold == {"train"}:
        # Val will be used for evaluating the model after each epoch:
        train_ds, val_ds = get_folds(config, root_dir, folds=["train", "val"])
    elif fold == {"train", "val"}:
        # Both train and val are meant to be used for training
        train_ds, = get_folds(config, root_dir, folds=["train_val"])
        val_ds = None
    else:
        # Should not arrive here since main makes sure config["fold"] is either one of the above
        print_utils.print_error("ERROR: specified folds not recognized!")
        raise NotImplementedError

    # --- Instantiate backbone network
    if config["backbone_params"]["name"] in ["deeplab50", "deeplab101"]:
        assert 1 < config["optim_params"]["batch_size"], \
            "When using backbone {}, batch_size has to be at least 2 for the batchnorm of the ASPPPooling to work."\
                .format(config["backbone_params"]["name"])
    backbone = get_backbone(config["backbone_params"])

    # --- Launch training
    train(gpu, config, shared_dict, barrier, train_ds, val_ds, backbone)


def eval_process(gpu, config, shared_dict, barrier):
    from frame_field_learning.evaluate import evaluate

    torch.manual_seed(0)  # Ensure same seed for all processes
    # --- Find data directory --- #
    root_dir_candidates = [os.path.join(data_dirpath, config["dataset_params"]["root_dirname"]) for data_dirpath in
                           config["data_dir_candidates"]]
    root_dir, paths_tried = python_utils.choose_first_existing_path(root_dir_candidates, return_tried_paths=True)
    if root_dir is None:
        print_utils.print_error(
            "GPU {} -> ERROR: Data root directory amongst \"{}\" not found!".format(gpu, paths_tried))
        raise NotADirectoryError(f"Couldn't find a directory in {paths_tried} (gpu:{gpu})")
    print_utils.print_info("GPU {} -> Using data from {}".format(gpu, root_dir))
    config["data_root_dir"] = root_dir

    # --- Get dataset
    # - CHANGE HERE TO ADD YOUR OWN DATASET
    eval_ds, = get_folds(config, root_dir, folds=config["fold"])  # config["fold"] is already a list (of length 1)

    # --- Instantiate backbone network (its backbone will be used to extract features)
    backbone = get_backbone(config["backbone_params"])

    evaluate(gpu, config, shared_dict, barrier, eval_ds, backbone)

