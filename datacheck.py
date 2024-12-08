from itertools import starmap
import struct
import numpy as np
from typing import Literal, Optional
import marshmallow_numpy

import argparse

ParsedOutputData = dict[str, marshmallow_numpy.NumpyArray]


def parsed_output_data_to_py(i: ParsedOutputData) -> dict[str, list[float]]:
    return dict(starmap(lambda k, v: (k, v.tolist()), i.items()))


def parse_str_dump_to_arrays(raw_str: str) -> ParsedOutputData:
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
    if not len(ret):
        raise IndexError("There are no arrays to be parsed, something's wrong")
    return ret


def parse_binary_dump_to_arrays(binary_data: bytes) -> ParsedOutputData:
    curr_pos = 0
    ret: ParsedOutputData = {}
    while curr_pos < len(binary_data):
        # Format: uint64_t for length of upcoming array in doubles, one char for array name, doubles data
        header_format = "<Qc"
        unp = struct.unpack_from(header_format, binary_data, offset=curr_pos)
        (arr_len, arr_name) = unp
        curr_pos += struct.calcsize(header_format)
        arr_data = np.frombuffer(
            binary_data, count=arr_len, offset=curr_pos, dtype="d"
        )  # https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.double
        ret[arr_name.decode()] = arr_data
        curr_pos += 8 * arr_len
    return ret


def parse_dump_to_arrays(
    raw_binary_data: bytes, is_human_readable: Optional[bool] = None
) -> ParsedOutputData:
    # if human readable is not provided, this will guess based on start of binary data
    if is_human_readable is None:
        is_human_readable = raw_binary_data.startswith(b"==BEGIN DUMP_ARRAYS==")

    if is_human_readable:
        return parse_str_dump_to_arrays(raw_binary_data.decode())
    else:
        return parse_binary_dump_to_arrays(raw_binary_data)


def compare_arrays_and_get_deviation(
    max_deviation, arr_name, np_cand_arr, np_truth_arr
):
    np_truth_upper_bound = np.multiply(np_truth_arr, 1 + max_deviation)
    np_truth_lower_bound = np.multiply(np_truth_arr, 1 - max_deviation)
    np_cand_exceeds_upper = np.greater(np_cand_arr, np_truth_upper_bound)
    np_cand_below_lower = np.less(np_cand_arr, np_truth_lower_bound)
    np_cand_oob = np.logical_or(np_cand_exceeds_upper, np_cand_below_lower)
    if np_cand_oob.any():
        show_where_arrays_mismatch(
            "Candidate element in {violating_idx} is {violating_elem} while the true value is {good_elem}, >"
            + str(max_deviation * 100)
            + "% different.",
            arr_name,
            np.argwhere(np_cand_oob)[0],
            np_cand_arr,
            np_truth_arr,
        )

    np_truth_eq0 = np.equal(np_truth_arr, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        np_ratio = np.divide(np_cand_arr, np_truth_arr)
    np_div0_fixed = np.where(np_truth_eq0, 1.0, np_ratio)
    np_percent = np.subtract(np_div0_fixed, 1.0)
    return np_percent


def show_where_arrays_mismatch(
    fstr: str,
    arr_name: str,
    fault_idx: int,
    faulty_arr: np.ndarray,
    good_arr: np.ndarray,
):
    raise ValueError(
        fstr.format(
            violating_idx=f"{arr_name}[{fault_idx}]",
            good_elem=good_arr[fault_idx],
            violating_elem=faulty_arr[fault_idx],
            div_diff=faulty_arr[fault_idx] / good_arr[fault_idx],
        )
    )


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
            if not np.allclose(truth_arr, cand_arr):
                cand_mismatch = np.not_equal(truth_arr, cand_arr)
                show_where_arrays_mismatch(
                    "Candidate element in {violating_idx} is {violating_elem} while the true value is {good_elem}, failed exact match.",
                    arr_name,
                    np.argwhere(cand_mismatch)[0][0],
                    cand_arr,
                    truth_arr,
                )
    return np.concatenate(deviation_runs) if len(deviation_runs) else np.empty((1,))


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
    true_f = open(args.truth, "rb")
    against_f = open(args.against, "rb")

    parsed_true = parse_dump_to_arrays(true_f.read())
    parsed_against = parse_dump_to_arrays(against_f.read())

    true_f.close()
    against_f.close()

    dev = compare_results(parsed_true, parsed_against, args.mode, args.max_deviation)
