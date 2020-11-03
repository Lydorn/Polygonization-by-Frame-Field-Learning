import os.path
import fnmatch
import argparse

from lydorn_utils import python_utils


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '--dirpath',
        default="/home/lydorn/data/mapping_challenge_dataset/eval_runs",
        type=str,
        help='Path to eval directory')

    args = argparser.parse_args()
    return args


def convert(in_filepath, stat_names):
    stats = python_utils.load_json(in_filepath)
    if stats:
        string = ""
        for stat_name in stat_names:
            string += str(round(100 * stats[stat_name], 1)) + " & "
        return string[:-2] + "\\\\"
    else:
        print("File not found!")
        return ""


def main():
    args = get_args()

    stat_names = ["AP", "AP_50", "AP_75", "AP_S", "AP_M", "AP_L", "AR", "AR_50", "AR_75", "AR_S", "AR_M", "AR_L"]

    dirname_list = next(os.walk(args.dirpath))[1]
    dirname_list = sorted(dirname_list)

    run_file_latex_list = [("Run name", "filename", "Latex")]
    run_max_len = 0
    file_max_len = 0
    for dirname in dirname_list:
        dirpath = os.path.join(args.dirpath, dirname)
        in_filename_list = fnmatch.filter(os.listdir(dirpath), "*.stats.*.annotation.*.json")
        in_filename_list = sorted(in_filename_list)
        for in_filename in in_filename_list:
            in_filepath = os.path.join(dirpath, in_filename)
            latex = convert(in_filepath, stat_names)
            run = dirname[:-len(" | 0000-00-00 00:00:00")]
            run_file_latex_list.append((run, in_filename, latex))
            run_max_len = max(run_max_len, len(run))
            file_max_len = max(file_max_len, len(in_filename))

    # print
    for run, file, latex in run_file_latex_list:
        print(run.ljust(run_max_len, ' '), file.ljust(file_max_len, ' '), latex)


if __name__ == '__main__':
    main()
