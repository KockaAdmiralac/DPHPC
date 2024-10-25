import re
from typing import Literal


def parse_dump_to_arrays(raw_str: str) -> dict[str, list[float]]:
    initial_split = re.compile(
        r"(?:==BEGIN DUMP_ARRAYS==\nbegin dump: ([^\n]+)\n((?:\d+\.\d+ ?\n?)+)\nend   dump: ([^\n]+)\n==END   DUMP_ARRAYS==)+",
        re.MULTILINE | re.DOTALL,
    )
    split_dumps = initial_split.findall(raw_str)
    results = {}
    for res in split_dumps:
        if res[0] != res[2]:
            raise ValueError("Regex found an unmatched begin dump string")
        split_nums = map(
            lambda n: float(n.rstrip("\n")), res[1].rstrip("\n").rstrip(" ").split(" ")
        )
        results[res[0]] = list(split_nums)
    return results


# max_deviation is a percentage, not absolute
def compare_results_fuzzy(
    ground_truth: dict[str, list[float]],
    candidate: dict[str, list[float]],
    max_deviation: float = None,
) -> list[float]:
    if max_deviation is None:
        max_deviation = 0.015
    diff_keys = set(ground_truth.keys()).difference(candidate.keys())
    if len(diff_keys):
        raise IndexError(
            f"Keys {diff_keys} are not output by both the ground truth and the candidate"
        )

    for arr_name in ground_truth.keys():
        cand_arr = candidate[arr_name]
        truth_arr = ground_truth[arr_name]
        if len(truth_arr) != len(cand_arr):
            raise IndexError(
                f"ground truth array length {len(truth_arr)} != candidate length {len(cand_arr)}"
            )

        for i, (truth_elem, cand_elem) in enumerate(zip(truth_arr, cand_arr)):
            if cand_elem < truth_elem * (
                1 - max_deviation
            ) or cand_elem > truth_elem * (1 + max_deviation):
                raise ValueError(
                    f"Candidate element in {arr_name}[{i}] is {cand_elem} while the true value is {truth_elem}, >{max_deviation*100}% different."
                )
            yield cand_elem / truth_elem - 1.0  # Useful to get info on deviations.


def compare_results_from_raw(
    ground_truth: str,
    candidate: str,
    mode: Literal["strict", "fuzzy"],
    max_deviation: float = None,
) -> list[float]:
    if mode == "fuzzy":
        return compare_results_fuzzy(
            parse_dump_to_arrays(ground_truth),
            parse_dump_to_arrays(candidate),
            max_deviation=max_deviation,
        )
    else:
        if ground_truth != candidate:
            raise ValueError("Ground truth and candidate aren't 100%% identical")
        return (0,)


if __name__ == "__main__":
    with open("results/gemver/N20_truth", "r") as f:
        print(parse_dump_to_arrays(f.read()))
