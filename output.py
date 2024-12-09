import argparse
import itertools
import json
from pathlib import Path
import pprint
from typing import Iterable, List, Literal, Tuple
import marshmallow_dataclass
import numpy as np
import tabulate

from structures import ProcessedResult, SingleBenchmark


def load_results(results_file: Path) -> List[ProcessedResult]:
    procres_schema = marshmallow_dataclass.class_schema(ProcessedResult)()
    with open(results_file, "r") as f:
        results_pyobj = json.load(f)

    ret: List[ProcessedResult] = list(map(procres_schema.load, results_pyobj))  # type: ignore
    return ret


def keep_valid_runs(results: Iterable[ProcessedResult]) -> Iterable[ProcessedResult]:
    for res in results:
        if res.data_checked and res.data_valid:
            yield res


ResultTimings = dict[str, List[float]]
PreprocessedResultPair = Tuple[SingleBenchmark, ResultTimings]
PreprocessedResults = List[PreprocessedResultPair]


def group_runs(results: Iterable[ProcessedResult]) -> PreprocessedResults:
    existing_results: PreprocessedResults = []
    for res in results:
        curr_tuple = next(
            (t for t in existing_results if t[0] == res.referenced_run), None
        )
        if curr_tuple is None:
            curr_tuple = (res.referenced_run, {})
            existing_results.append(
                curr_tuple
            )  # note curr_tuple will keep the reference to this
        curr_ret = curr_tuple[1]
        for t, val in res.timings.items():
            if t not in curr_ret:
                curr_ret[t] = []
            curr_ret[t].append(val)
    return existing_results


def preprocess_results(
    results: Iterable[ProcessedResult],
) -> PreprocessedResults:
    return group_runs(keep_valid_runs(results))


def output_table(results: Iterable[PreprocessedResultPair]) -> None:
    header = [
        "Variant",
        "Threads",
        "Mean",
        "Min",
        "Max",
        "Median",
        "Stdev",
        "Run count",
    ]
    bm_key = lambda res: res[0].benchmark
    results_by_bm = itertools.groupby(sorted(results, key=bm_key), key=bm_key)
    for benchmark, bm_res in results_by_bm:
        print(f"For benchmark {benchmark}:")
        print(
            tabulate.tabulate(
                sorted(
                    [
                        (
                            sb.variant,
                            sb.run_options.threads,
                            np.mean(timings["kernel_time"]),
                            np.min(timings["kernel_time"]),
                            np.max(timings["kernel_time"]),
                            np.median(timings["kernel_time"]),
                            np.std(timings["kernel_time"]),
                            len(timings["kernel_time"]),
                        )
                        for (sb, timings) in bm_res
                    ],
                    key=lambda row: list(row)[2],
                ),
                headers=header,
            )
        )


def output_graphs(results: Iterable[PreprocessedResultPair]) -> None:
    pass


output_modes = {"table": output_table, "graphs": output_graphs}


def run_output(
    results: Iterable[PreprocessedResultPair],
    methods: List[Literal["table", "graphs"]],
    filt=lambda res, method: True,
) -> None:
    for method in methods:
        filt_res = filter(lambda res: filt(res, method), results)
        output_modes[method](filt_res)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process collected output results and display/graph them"
    )
    parser.add_argument(
        "--from",
        dest="from_files",
        type=str,
        help="Files to load results from",
        action="append",
    )
    parser.add_argument(
        "--output",
        type=str,
        choices=list(output_modes.keys()),
        help="How you would like to output the results",
        action="append",
        required=True,
    )

    args = parser.parse_args()

    all_results = itertools.chain(*map(load_results, args.from_files))

    preproc_results = preprocess_results(all_results)
    # pprint.pprint(preproc_results)

    run_output(preproc_results, args.output)
