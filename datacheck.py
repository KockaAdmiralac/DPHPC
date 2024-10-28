from itertools import starmap
from timeit import timeit
import numpy as np
from typing import Literal, Optional

import argparse

ParsedOutputData = dict[str, np.ndarray]


def parsed_output_data_to_py(i: ParsedOutputData) -> dict[str, list[float]]:
    return dict(starmap(lambda k, v: (k, v.tolist()), i.items()))


def parse_dump_to_arrays(raw_str: str) -> ParsedOutputData:
    line_split = raw_str.splitlines()
    ret: ParsedOutputData = {}
    curr_arr_name: str = ""
    temp_arrays: list[np.ndarray] = []
    for line_no, line in enumerate(line_split):
        if line == "==BEGIN DUMP_ARRAYS==":
            continue
        elif line.startswith("begin dump: "):
            curr_arr_name = line.split(": ")[1]
            temp_arrays = []
        elif line == "==END   DUMP_ARRAYS==":
            continue
        elif line.startswith("end   dump: "):
            if line.split(": ")[1] != curr_arr_name:
                raise ValueError(
                    f"Regex found an unmatched begin dump string at line {line_no}"
                )
            ret[curr_arr_name] = np.concatenate(temp_arrays)
        else:
            temp_arrays.append(np.fromstring(line, sep=" "))
    return ret


def compare_arrays_and_get_deviation(
    max_deviation, arr_name, np_cand_arr, np_truth_arr
):
    np_truth_upper_bound = np.multiply(np_truth_arr, 1 + max_deviation)
    np_truth_lower_bound = np.multiply(np_truth_arr, 1 - max_deviation)
    np_cand_exceeds_upper = np.greater(np_cand_arr, np_truth_upper_bound)
    np_cand_below_lower = np.less(np_cand_arr, np_truth_lower_bound)
    np_cand_oob = np.logical_or(np_cand_exceeds_upper, np_cand_below_lower)
    violating_indices = np.argwhere(np_cand_oob)
    if len(violating_indices):
        raise ValueError(
            f"Candidate element in {arr_name}[{violating_indices[0][0]}] is {np_cand_arr[violating_indices[0]]} while the true value is {np_truth_arr[violating_indices[0]]}, >{max_deviation*100}% different."
        )

    np_truth_eq0 = np.equal(np_truth_arr, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        np_ratio = np.divide(np_cand_arr, np_truth_arr)
    np_div0_fixed = np.where(np_truth_eq0, 1.0, np_ratio)
    np_percent = np.subtract(np_div0_fixed, 1.0)
    return np_percent


# max_deviation is a percentage, not absolute
def compare_results(
    ground_truth: ParsedOutputData,
    candidate: ParsedOutputData,
    mode: Literal["strict", "fuzzy"],
    max_deviation: Optional[float] = None,
) -> np.ndarray:
    if max_deviation is None:
        max_deviation = 0.015
    diff_keys = set(ground_truth.keys()).difference(candidate.keys())
    if len(diff_keys):
        raise IndexError(
            f"Keys {diff_keys} are not output by both the ground truth and the candidate"
        )

    deviation_runs = []
    for arr_name in ground_truth.keys():
        cand_arr = candidate[arr_name]
        truth_arr = ground_truth[arr_name]
        if len(truth_arr) != len(cand_arr):
            raise IndexError(
                f"ground truth array length {len(truth_arr)} != candidate length {len(cand_arr)}"
            )

        if mode == "fuzzy":
            deviation_runs.append(
                compare_arrays_and_get_deviation(
                    max_deviation, arr_name, cand_arr, truth_arr
                )
            )
        else:
            if np.not_equal(truth_arr, cand_arr).any():
                raise ValueError("Ground truth and candidate aren't 100% identical")
            return np.array((1,))
    return np.concatenate(deviation_runs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare two data files to check they match"
    )

    parser.add_argument(
        "--truth", type=str, required=True, help="The data against which to compare"
    )
    parser.add_argument(
        "--against",
        type=str,
        required=True,
        help="The data of interest, usually a failure",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="fuzzy",
        help="Whether to do a fuzzy or strict match",
    )
    parser.add_argument(
        "--max_deviation", type=float, help="Relative deviation to tolerate"
    )

    args = parser.parse_args()
    true_f = open(args.truth, "r")
    against_f = open(args.against, "r")

    parsed_true = parse_dump_to_arrays(true_f.read())
    parsed_against = parse_dump_to_arrays(against_f.read())

    true_f.close()
    against_f.close()

    dev = compare_results(parsed_true, parsed_against, args.mode, args.max_deviation)
