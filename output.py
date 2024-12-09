import argparse
import itertools
import json
from pathlib import Path
import pprint
from typing import Iterable, List, Literal, Tuple
import marshmallow_dataclass
import numpy as np
import tabulate

import result_processing
from structures import ProcessedResult, SingleBenchmark


def load_results(results_file: Path) -> List[ProcessedResult]:
    procres_schema = marshmallow_dataclass.class_schema(ProcessedResult)()
    with open(results_file, "r") as f:
        results_pyobj = json.load(f)

    ret: List[ProcessedResult] = list(map(procres_schema.load, results_pyobj))  # type: ignore
    return ret


def output_table(results: Iterable[result_processing.PreprocessedResultPair]) -> None:
    header = [
        "Variant",
        "Threads",
        "Mean",
        "Min",
        "Max",
        "Median",
        "Stdev",
        "95% CI of avg",
        "Run count",
    ]
    bm_key = lambda res: res[0].benchmark
    results_by_bm = itertools.groupby(sorted(results, key=bm_key), key=bm_key)
    for benchmark, bm_res in results_by_bm:
        print(f"For benchmark {benchmark}:")
        N2_key = lambda res: int(
            res[0].variant_config.compile_options.extra_defines["N2"]
        )
        results_by_N2 = itertools.groupby(sorted(bm_res, key=N2_key), key=N2_key)
        for N2, N2_res in results_by_N2:
            print(f"For N2={N2}:")
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
                                result_processing.get_ci(
                                    np.array(timings["kernel_time"]), np.mean
                                ),
                                len(timings["kernel_time"]),
                            )
                            for (sb, timings) in N2_res
                        ],
                        key=lambda row: list(row)[2],
                    ),
                    headers=header,
                )
            )
            print()
        print()


def output_graphs(results: Iterable[result_processing.PreprocessedResultPair]) -> None:
    pass


output_modes = {"table": output_table, "graphs": output_graphs}


def run_output(
    results: Iterable[result_processing.PreprocessedResultPair],
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

    preproc_results = result_processing.preprocess_results(all_results)
    # pprint.pprint(preproc_results)

    run_output(preproc_results, args.output)
