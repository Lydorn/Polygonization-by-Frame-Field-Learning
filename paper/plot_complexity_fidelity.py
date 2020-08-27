import os
import matplotlib.pyplot as plt

from lydorn_utils import python_utils
from lydorn_utils import print_utils


def get_stat_from_all(stat_filepath_format, method_info, tolerances, stat_name):
    stat_list = [0 for _ in tolerances]
    for i, tolerance in enumerate(tolerances):
        filepath = stat_filepath_format.format(method_info["name"], tolerance)
        stats = python_utils.load_json(filepath)
        if stats:
            stat_list[i] = stats[stat_name]
        else:
            print_utils.print_warning("WARNING: could not open {}".format(filepath))
    return stat_list


def plot_stat(stat_filepath_format, method_info_list, tolerances, stat_name, exp_name):
    legend = []
    for method_info in method_info_list:
        ap_list = get_stat_from_all(stat_filepath_format, method_info, tolerances, stat_name)
        legend.append(method_info["title"])

        plt.plot(tolerances, ap_list)

    plt.legend(legend, loc='lower left')
    plt.xlabel("Tolerance")
    plt.ylabel(stat_name)
    plt.title(exp_name + ": " + stat_name + " vs tolerance")
    plt.savefig(exp_name.replace(" ", "_") + "_" + stat_name + "_vs_tolerance.pdf")
    plt.show()


def main():
    method_info_list = [
        {
            "title": "Baseline polygonization",
            "name": "simple"
        },
        {
            "title": "Our polygonization",
            "name": "acm"
        },
    ]
    tolerances = [0.125, 0.25, 0.5, 1, 2, 4, 8, 16]
    eval_runs_dirpath = "/data/data/mapping_challenge_dataset/eval_runs_cluster"

    info_list = [
        {
            "exp_name": "U-Net16 full method",
            "run_dirname": "mapping_dataset.unet16.train_val | 2020-02-21 03:09:03",
        },
        {
            "exp_name": "U-Net16 field off",
            "run_dirname": "mapping_dataset.unet16.field_off.train_val | 2020-02-28 23:51:16",
        },
        {
            "exp_name": "DeepLab101 full method",
            "run_dirname": "mapping_dataset.deeplab101.train_val | 2020-02-24 23:57:19",
        },
        {
            "exp_name": "DeepLab101 field off",
            "run_dirname": "mapping_dataset.deeplab101.field_off.train_val | 2020-03-02 00:03:45",
        },
    ]
    for info in info_list:
        stat_filepath_format = os.path.join(eval_runs_dirpath, info["run_dirname"], "test.stats.test.annotation.poly.{}.tol_{}.json")

        plot_stat(stat_filepath_format, method_info_list, tolerances, "AP", info["exp_name"])
        plot_stat(stat_filepath_format, method_info_list, tolerances, "AR", info["exp_name"])


if __name__ == '__main__':
    main()
