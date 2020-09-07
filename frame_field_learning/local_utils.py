import functools
import os
import subprocess
import sys
import time

from lydorn_utils import run_utils, print_utils
from lydorn_utils import python_utils


def compute_max_disp(disp_params):
    m_g_t = disp_params["max_global_translation"]
    m_g_h = disp_params["max_global_homography"]
    m_p_t = disp_params["max_poly_translation"]
    m_p_h = disp_params["max_poly_homography"]
    m_h_c = disp_params["max_homography_coef"]
    return (m_g_t + m_h_c*m_g_h) + (m_p_t + m_h_c*m_p_h)


def get_git_revision_hash():
    try:
        hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.STDOUT).decode("utf-8")[:-1]
    except subprocess.CalledProcessError:
        hash = None
    return hash


def setup_run(config):
    run_name = config["run_name"]
    new_run = config["new_run"]
    init_run_name = config["init_run_name"]

    working_dir = os.path.dirname(os.path.abspath(__file__))
    runs_dir = os.path.join(working_dir, config["runs_dirpath"])

    # setup init checkpoints directory path if one is specified:
    if init_run_name is not None:
        init_run_dirpath = run_utils.setup_run_dir(runs_dir, init_run_name)
        _, init_checkpoints_dirpath = run_utils.setup_run_subdirs(init_run_dirpath)
    else:
        init_checkpoints_dirpath = None

    # setup run directory:
    run_dirpath = run_utils.setup_run_dir(runs_dir, run_name, new_run)

    # save config in logs directory
    run_utils.save_config(config, run_dirpath)

    # save args
    args_filepath = os.path.join(run_dirpath, "args.json")
    args_to_save = {
        "run_name": run_name,
        "new_run": new_run,
        "init_run_name": init_run_name,
        "batch_size": config["optim_params"]["batch_size"],
    }
    if "samples" in config:
        args_to_save["samples"] = config["samples"]
    python_utils.save_json(args_filepath, args_to_save)

    # save current commit hash
    commit_hash = get_git_revision_hash()
    if commit_hash is not None:
        commit_hash_filepath = os.path.join(run_dirpath, "commit_history.json")
        if os.path.exists(commit_hash_filepath):
            commit_hashes = python_utils.load_json(commit_hash_filepath)
            if commit_hashes[-1] != commit_hash:
                commit_hashes.append(commit_hash)
                python_utils.save_json(commit_hash_filepath, commit_hashes)
        else:
            commit_hashes = [commit_hash]
            python_utils.save_json(commit_hash_filepath, commit_hashes)

    return run_dirpath, init_checkpoints_dirpath


def get_run_dirpath(runs_dirpath, run_name):
    working_dir = os.path.dirname(os.path.abspath(__file__))
    runs_dir = os.path.join(working_dir, runs_dirpath)
    try:
        run_dirpath = run_utils.setup_run_dir(runs_dir, run_name, check_exists=True)
    except FileNotFoundError as e:
        print_utils.print_error(f"ERROR: {e}")
        sys.exit()
    return run_dirpath


def batch_to_cuda(batch):
    # Send data to computing device:
    for key, item in batch.items():
        if hasattr(item, "cuda"):
            batch[key] = item.cuda(non_blocking=True)
    return batch


def batch_to_cpu(batch):
    # Send data to computing device:
    for key, item in batch.items():
        if hasattr(item, "cuda"):
            batch[key] = item.cpu()
    return batch


def split_batch(tile_data):
    assert len(tile_data["image"].shape) == 4, "tile_data[\"image\"] should be (N, C, H, W)"
    tile_data_list = []
    for i in range(tile_data["image"].shape[0]):
        individual_tile_data = {}
        for key, item in tile_data.items():
            if not i < len(item):
                print(key, len(item))
            individual_tile_data[key] = item[i]
        tile_data_list.append(individual_tile_data)
    return tile_data_list


def _concat_dictionaries(dict1, dict2):
    """
    Recursive concat dictionaries. Dict 1 and Dict 2 must have the same key hierarchy (this is not checked).

    :param dict1: Dictionary to add to.
    :param dict2: Dictionary to add from
    :return: Merged dictionary dict1
    """
    for key in dict1.keys():
        item1 = dict1[key]
        item2 = dict2[key]
        if isinstance(item1, dict):  # And item2 is dict too.
            dict1[key] = _concat_dictionaries(item1, item2)
        else:
            dict1[key].extend(item2)
    return dict1


def _root_concat_dictionaries(dict1, dict2):

    t0 = time.time()
    dict1 = _concat_dictionaries(dict1, dict2)
    print(f"_root_concat_dictionaries: {time.time() - t0:02}s")
    return dict1


def list_of_dicts_to_dict_of_lists(list_of_dicts):
    """
    Works recursively by using _concat_dictionaries which is recursive

    @param list_of_dicts:
    @return: dict_of_lists
    """
    return functools.reduce(_concat_dictionaries, list_of_dicts)


def flatten_dict(_dict):
    """
    Makes a hierarchy of dicts flat

    @param _dict:
    @return:
    """
    new_dict = {}
    for key, item in _dict.items():
        if isinstance(item, dict):
            item = flatten_dict(item)
            for k in item.keys():
                new_dict[key + "." + k] = item[k]
        else:
            new_dict[key] = item
    return new_dict


def _generate_list_of_dicts(list_length, methods_count, submethods_count, annotation_count, segmentation_length):
    list_of_dicts = []
    for i in range(list_length):
        d = {}
        for method_i in range(methods_count):
            d[f"method_{method_i}"] = {}
            for submethod_i in range(submethods_count):
                d[f"method_{method_i}"][f"submethod_{submethod_i}"] = []
                for annotation_i in range(annotation_count):
                    annotation = {
                        "image_id": 0,
                        "segmentation": [list(range(segmentation_length))],
                        "category_id": 100,  # Building
                        "bbox": [0, 1, 0, 1],
                        "score": 1.0
                    }
                    d[f"method_{method_i}"][f"submethod_{submethod_i}"].append(annotation)
        list_of_dicts.append(d)
    return list_of_dicts


def main():
    # list_of_dicts = [
    #     {
    #         "method1": {
    #             "submethod1": [[0, 1, 2, 3], [4, 5, 6]]
    #         }
    #     },
    #     {
    #         "method1": {
    #             "submethod1": [[7, 8, 9], [10, 11, 12, 13, 14, 15]]
    #         }
    #     },
    # ]
    t0 = time.time()
    list_of_dicts = _generate_list_of_dicts(list_length=2000, methods_count=2, submethods_count=2, annotation_count=100, segmentation_length=200)
    print(f"_generate_list_of_dicts: {time.time() - t0:02}s")

    t0 = time.time()
    dict_of_lists = list_of_dicts_to_dict_of_lists(list_of_dicts)
    print(f"list_of_dicts_to_dict_of_lists: {time.time() - t0:02}s")

    flat_dict_of_lists = flatten_dict(dict_of_lists)


if __name__ == "__main__":
    main()
