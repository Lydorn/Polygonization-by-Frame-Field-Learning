#!/usr/bin/env python3

###################################################################
# Use this script for the main tasks of training, evaluation, computing measures on test data or just inference on an image.
###################################################################

import os
import argparse
import sys

import torch
import torch.multiprocessing

try:
    __import__("frame_field_learning.local_utils")
except ImportError:
    print("ERROR: The frame_field_learning package is not installed! "
          "Execute script setup.sh to install local dependencies such as frame_field_learning in develop mode.")
    exit()

import frame_field_learning.local_utils

from child_processes import train_process, eval_process
from backbone import get_backbone
from eval_coco import eval_coco

from lydorn_utils import run_utils, python_utils
from lydorn_utils import print_utils


# ---Examples of calling main.py: --- #
#
#
# ------------------------------------#


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '--in_filepath',
        type=str,
        nargs='*',
        help='For launching prediction on several images, use this argument to specify their paths.'
             'If --out_dirpath is specified, prediction outputs will be saved there..'
             'If --out_dirpath is not specified, predictions will be saved next to inputs.'
             'Make sure to also specify the run_name of the model to use for prediction.')
    argparser.add_argument(
        '--out_dirpath',
        type=str,
        help='Path to the output directory of prediction when using the --in_filepath option to launch prediction on several images.')

    argparser.add_argument(
        '-c', '--config',
        type=str,
        help='Name of the config file, excluding the .json file extension.')
    argparser.add_argument(
        '--dataset_params',
        type=str,
        help='Allows to overwrite the dataset_params in the config file. Accepts a path to a .json file.')

    argparser.add_argument(
        '-r', '--runs_dirpath',
        default="runs",
        type=str,
        help='Directory where runs are recorded (model saves and logs).')
    argparser.add_argument(
        '--run_name',
        type=str,
        help='Name of the run to use.'
             'That name does not include the timestamp of the folder name: <run_name> | <yyyy-mm-dd hh:mm:ss>.')
    argparser.add_argument(
        '--new_run',
        action='store_true',
        help="Train from scratch (when True) or train from the last checkpoint (when False)")
    argparser.add_argument(
        '--init_run_name',
        type=str,
        help="This is the run_name to initialize the weights from."
             "If None, weights will be initialized randomly."
             "This is a single word, without the timestamp.")
    argparser.add_argument(
        '--samples',
        type=int,
        help='Limits the number of samples to train (and validate and test) if set.')

    argparser.add_argument(
        '-b', '--batch_size',
        type=int,
        help='Batch size. Default value can be set in config file. Is doubled when no back propagation is done (while in eval mode). If a specific effective batch size is desired, set the eval_batch_size argument.')
    argparser.add_argument(
        '--eval_batch_size',
        type=int,
        help='Batch size for evaluation. Overrides the effective batch size when evaluating.')
    argparser.add_argument(
        '-m', '--mode',
        default="train",
        type=str,
        choices=['train', 'eval', 'eval_coco'],
        help='Mode to launch the script in. '
             'Train: train model on speciffied folds. '
             'Eval: eval model on specified fold. '
             'Eval_coco: measures COCO metrics of specified fold')
    argparser.add_argument(
        '--fold',
        nargs='*',
        type=str,
        choices=['train', 'val', 'test'],
        help='If training (mode=train): all folds entered here will be used for optimizing the network.'
             'If the train fold is selected and not the val fold, the val fold will be used during training to validate at each epoch.'
             'The most common scenario is to optimize on train and validate on val: select only train.'
             'When optimizing the network for the last time before test, we would like to optimize it on train + val: in that case select both train and val folds.'
             'Then for evaluation (mode=eval), we might want to evaluate on the val folds for hyper-parameter selection.'
             'And finally evaluate (mode=eval) on the test fold for the final predictions (and possibly metric) for the paper/competition')
    argparser.add_argument(
        '--max_epoch',
        type=int,
        help='Stop training when max_epoch is reached. If not set, value in config is used.')
    argparser.add_argument(
        '--eval_patch_size',
        type=int,
        help='When evaluating, patch size the tile split into.')
    argparser.add_argument(
        '--eval_patch_overlap',
        type=int,
        help='When evaluating, patch the tile with the specified overlap to reduce edge artifacts when reconstructing '
             'the whole tile')

    argparser.add_argument('--master_addr', default="localhost", type=str, help="Address of master node")
    argparser.add_argument('--master_port', default="6666", type=str, help="Port on master node")
    argparser.add_argument('-n', '--nodes', default=1, type=int, metavar='N', help="Number of total nodes")
    argparser.add_argument('-g', '--gpus', default=1, type=int, help='Number of gpus per node')
    argparser.add_argument('-nr', '--nr', default=0, type=int, help='Ranking within the nodes')

    args = argparser.parse_args()
    return args


def launch_inference_from_filepath(args):
    from frame_field_learning.inference_from_filepath import inference_from_filepath

    # --- First step: figure out what run (experiment) is to be evaluated
    # Option 1: the run_name argument is given in which case that's our run
    run_name = None
    config = None
    if args.run_name is not None:
        run_name = args.run_name
    # Else option 2: Check if a config has been given to look for the run_name
    if args.config is not None:
        config = run_utils.load_config(args.config)
        if config is not None and "run_name" in config and run_name is None:
            run_name = config["run_name"]
    # Else abort...
    if run_name is None:
        print_utils.print_error("ERROR: the run to evaluate could no be identified with the given arguments. "
                                "Please specify either the --run_name argument or the --config argument "
                                "linking to a config file that has a 'run_name' field filled with the name of "
                                "the run name to evaluate.")
        sys.exit()

    # --- Second step: get path to the run and if --config was not specified, load the config from the run's folder
    run_dirpath = frame_field_learning.local_utils.get_run_dirpath(args.runs_dirpath, run_name)
    if config is None:
        config = run_utils.load_config(config_dirpath=run_dirpath)
    if config is None:
        print_utils.print_error(f"ERROR: the default run's config file at {run_dirpath} could not be loaded. "
                                f"Exiting now...")
        sys.exit()

    # --- Add command-line arguments
    if args.batch_size is not None:
        config["optim_params"]["batch_size"] = args.batch_size
    if args.eval_batch_size is not None:
        config["optim_params"]["eval_batch_size"] = args.eval_batch_size
    else:
        config["optim_params"]["eval_batch_size"] = 2*config["optim_params"]["batch_size"]

    # --- Load params in config set as relative path to another JSON file
    config = run_utils.load_defaults_in_config(config, filepath_key="defaults_filepath")

    config["eval_params"]["run_dirpath"] = run_dirpath
    if args.eval_patch_size is not None:
        config["eval_params"]["patch_size"] = args.eval_patch_size
    if args.eval_patch_overlap is not None:
        config["eval_params"]["patch_overlap"] = args.eval_patch_overlap

    backbone = get_backbone(config["backbone_params"])
    inference_from_filepath(config, args.in_filepath, backbone, args.out_dirpath)


def launch_train(args):
    assert args.config is not None, "Argument --config must be specified. Run 'python main.py --help' for help on arguments."
    config = run_utils.load_config(args.config)
    if config is None:
        print_utils.print_error(
            "ERROR: cannot continue without a config file. Exiting now...")
        sys.exit()
    config["runs_dirpath"] = args.runs_dirpath
    if args.run_name is not None:
        config["run_name"] = args.run_name
    config["new_run"] = args.new_run
    config["init_run_name"] = args.init_run_name
    if args.samples is not None:
        config["samples"] = args.samples
    if args.batch_size is not None:
        config["optim_params"]["batch_size"] = args.batch_size
    if args.max_epoch is not None:
        config["optim_params"]["max_epoch"] = args.max_epoch

    if args.fold is None:
        if "fold" in config:
            fold = set(config["fold"])
        else:
            fold = {"train"}  # Default values for train
    else:
        fold = set(args.fold)
    assert fold == {"train"} or fold == {"train", "val"}, \
        "Argument fold when training should be either: ['train'] or ['train', 'val']"
    config["fold"] = list(fold)
    print_utils.print_info("Training on fold(s): {}".format(config["fold"]))

    config["nodes"] = args.nodes
    config["gpus"] = args.gpus
    config["nr"] = args.nr
    config["world_size"] = args.gpus * args.nodes

    # --- Load params in config set as relative path to another JSON file
    config = run_utils.load_defaults_in_config(config, filepath_key="defaults_filepath")

    # Setup num_workers per process:
    if config["num_workers"] is None:
        config["num_workers"] = int(torch.multiprocessing.cpu_count() / config["gpus"])

    # --- Distributed init:
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    manager = torch.multiprocessing.Manager()
    shared_dict = manager.dict()
    shared_dict["run_dirpath"] = None
    shared_dict["init_checkpoints_dirpath"] = None
    barrier = manager.Barrier(args.gpus)

    torch.multiprocessing.spawn(train_process, nprocs=args.gpus, args=(config, shared_dict, barrier))


def launch_eval(args):
    # --- Init: fills mode-specific default command-line arguments
    if args.fold is None:
        fold = {"test"}  # Default value for eval mode
    else:
        fold = set(args.fold)
    assert len(fold) == 1, "Argument 'fold' must be a single fold in eval mode"
    # --- First step: figure out what run (experiment) is to be evaluated
    # Option 1: the run_name argument is given in which case that's our run
    run_name = None
    config = None
    if args.run_name is not None:
        run_name = args.run_name
    # Else option 2: Check if a config has been given to look for the run_name
    if args.config is not None:
        config = run_utils.load_config(args.config)
        if config is not None and "run_name" in config and run_name is None:
            run_name = config["run_name"]
    # Else abort...
    if run_name is None:
        print_utils.print_error("ERROR: the run to evaluate could no be identified with the given arguments. "
                                "Please specify either the --run_name argument or the --config argument "
                                "linking to a config file that has a 'run_name' field filled with the name of "
                                "the run name to evaluate.")
        sys.exit()

    # --- Second step: get path to the run and if --config was not specified, load the config from the run's folder
    run_dirpath = frame_field_learning.local_utils.get_run_dirpath(args.runs_dirpath, run_name)
    if config is None:
        config = run_utils.load_config(config_dirpath=run_dirpath)
    if config is None:
        print_utils.print_error(f"ERROR: the default run's config file at {run_dirpath} could not be loaded. "
                                f"Exiting now...")
        sys.exit()

    # --- Third step: Replace parameters in config file from command-line arguments
    if args.dataset_params is not None:
        config["dataset_params"] = python_utils.load_json(args.dataset_params)
    if args.samples is not None:
        config["samples"] = args.samples
    if args.batch_size is not None:
        config["optim_params"]["batch_size"] = args.batch_size
    if args.eval_batch_size is not None:
        config["optim_params"]["eval_batch_size"] = args.eval_batch_size
    else:
        config["optim_params"]["eval_batch_size"] = 2*config["optim_params"]["batch_size"]
    config["fold"] = list(fold)
    config["nodes"] = args.nodes
    config["gpus"] = args.gpus
    config["nr"] = args.nr
    config["world_size"] = args.gpus * args.nodes

    # --- Load params in config set as relative path to another JSON file
    config = run_utils.load_defaults_in_config(config, filepath_key="defaults_filepath")

    config["eval_params"]["run_dirpath"] = run_dirpath
    if args.eval_patch_size is not None:
        config["eval_params"]["patch_size"] = args.eval_patch_size
    if args.eval_patch_overlap is not None:
        config["eval_params"]["patch_overlap"] = args.eval_patch_overlap

    # Setup num_workers per process:
    if config["num_workers"] is None:
        config["num_workers"] = int(torch.multiprocessing.cpu_count() / config["gpus"])

    # --- Distributed init:
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    manager = torch.multiprocessing.Manager()
    shared_dict = manager.dict()
    shared_dict["name_list"] = manager.list()
    shared_dict["iou_list"] = manager.list()
    shared_dict["seg_coco_list"] = manager.list()
    shared_dict["poly_coco_list"] = manager.list()
    barrier = manager.Barrier(args.gpus)

    torch.multiprocessing.spawn(eval_process, nprocs=args.gpus, args=(config, shared_dict, barrier))


def launch_eval_coco(args):
    # --- Init: fills mode-specific default command-line arguments
    if args.fold is None:
        fold = {"test"}  # Default value for eval_coco
    else:
        fold = set(args.fold)
    assert len(fold) == 1, \
        "Argument fold when evaluating with COCO should be a single fold"

    # --- Find which run and which config file to evaluate the run with
    if args.run_name is None and args.config is None:
        print_utils.print_error("ERROR: At least of one --run_name or --config has to be specified.")
        sys.exit()
    elif args.run_name is None and args.config is not None:
        # Load config
        config = run_utils.load_config(args.config)
        # Verify it has a run_name specified
        if "run_name" not in config:
            print_utils.print_error("ERROR: run_name was not found in the provided config file, you can specify it with --run_name")
            sys.exit()
        run_name = config["run_name"]
    elif args.run_name is not None and args.config is None:
        # Load run_name's config
        run_dirpath = frame_field_learning.local_utils.get_run_dirpath(args.runs_dirpath, args.run_name)
        config = run_utils.load_config(config_dirpath=run_dirpath)
        run_name = args.run_name
    else:
        # Load specified config and use specified run_name
        config = run_utils.load_config(args.config)
        run_name = args.run_name

    # --- Load params in config set as relative path to another JSON file
    config = run_utils.load_defaults_in_config(config, filepath_key="defaults_filepath")

    # --- Second step: Replace parameters in config file from command-line arguments
    config["eval_params"]["run_name"] = run_name
    if args.samples is not None:
        config["samples"] = args.samples
    config["fold"] = list(fold)

    # Setup num_workers per process:
    if config["num_workers"] is None:
        config["num_workers"] = torch.multiprocessing.cpu_count()

    eval_coco(config)


def main():
    torch.manual_seed(0)
    # --- Process args --- #
    args = get_args()

    if args.in_filepath:  # Check if in_filepath is specified, it which case run the model on that image
        launch_inference_from_filepath(args)
    elif args.mode == "train":
        launch_train(args)
    elif args.mode == "eval":
        launch_eval(args)
    elif args.mode == "eval_coco":
        launch_eval_coco(args)


if __name__ == '__main__':
    main()
