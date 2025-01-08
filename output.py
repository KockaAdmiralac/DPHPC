import argparse
import itertools
import json
from pathlib import Path
import pprint
from typing import Iterable, List, Literal, Tuple
import marshmallow_dataclass
import numpy as np
import tabulate
import pandas as pd
import os
import json
import result_processing
from structures import ProcessedResult, SingleBenchmark
import plotting


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
                                (
                                    result_processing.get_ci(
                                        np.array(timings["kernel_time"]), np.mean
                                    )
                                    if len(timings["kernel_time"]) >= 2
                                    else None
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
    bm_key = lambda res: res[0].benchmark
    results_by_bm = itertools.groupby(sorted(results, key=bm_key), key=bm_key)
    adi = []
    gemver = []
    gemver1 = []
    gemver2 = []
    gemver3 = []
    for benchmark, bm_res in results_by_bm:
        data = []
        N2_key = lambda res: int(
            res[0].variant_config.compile_options.extra_defines["N2"]
        )
        results_by_N2 = itertools.groupby(sorted(bm_res, key=N2_key), key=N2_key)
        for N2, N2_res in results_by_N2:
            sort_list = sorted(
                [
                    (
                        sb.benchmark,
                        sb.variant,
                        sb.run_options.threads,
                        np.mean(timings["kernel_time"]),
                        np.min(timings["kernel_time"]),
                        np.max(timings["kernel_time"]),
                        np.median(timings["kernel_time"]),
                        np.std(timings["kernel_time"]),
                        (
                            result_processing.get_ci(
                                np.array(timings["kernel_time"]), np.mean
                            )
                            if len(timings["kernel_time"]) >= 2
                            else None
                        ),
                        len(timings["kernel_time"]),
                    )
                    for (sb, timings) in N2_res
                ],
                key=lambda row: list(row)[2],
            )
            for val in sort_list:
                temp_data = {}
                temp_data["threads"] = val[2]
                temp_data["N"] = N2
                temp_data["N2"] = N2
                temp_data["tr"] = val[2]
                temp_data["speedup"] = val[3]
                temp_data["implementation"] = val[1]
                temp_data["algorithm"] = val[0]
                temp_data["mean"] = val[3]
                temp_data["deviation"] = val[7]
                temp_data["deviation_window"] = val[8]
                data.append(temp_data)
        if benchmark == "adi":
            adi = data
        elif benchmark == "gemver_k1":
            gemver1 = data
        elif benchmark == "gemver_k2":
            gemver2 = data
        elif benchmark == "gemver_k3":
            gemver3 = data
        else:
            gemver = data

    p = os.path.join("plot", "adi.json")
    with open(p) as f:
        adi_dictionary = json.load(f)
    plotting.plotting_fun(
        adi,
        adi_dictionary["mpi_implementations"],
        adi_dictionary["cuda_implementations"],
        adi_dictionary["serial_implementations"],
        adi_dictionary["open_implementations"],
        adi_dictionary["mpi_implementations_names"],
        adi_dictionary["cuda_implementations_names"],
        adi_dictionary["serial_implementations_names"],
        adi_dictionary["open_implementations_names"],
        adi_dictionary["threads"],
        adi_dictionary["N2"],
        adi_dictionary["N2_c"],
        adi_dictionary["filename_list"],
        adi_dictionary["title_list"],
        adi_dictionary["plot_path"],
        adi_dictionary["set_threads"],
        adi_dictionary["set_n2"],
        adi_dictionary["runtime"],
        False
    )
    # p = os.path.join("plot", "gemver.json")
    # with open(p) as f:
    #     gem_dictionary = json.load(f)
    # plotting.plotting_fun(
    #     gemver,
    #     gem_dictionary["mpi_implementations"],
    #     gem_dictionary["cuda_implementations"],
    #     gem_dictionary["serial_implementations"],
    #     gem_dictionary["open_implementations"],
    #     gem_dictionary["mpi_implementations_names"],
    #     gem_dictionary["cuda_implementations_names"],
    #     gem_dictionary["serial_implementations_names"],
    #     gem_dictionary["open_implementations_names"],
    #     gem_dictionary["threads"],
    #     gem_dictionary["N2"],
    #     gem_dictionary["N2_c"],
    #     gem_dictionary["filename_list"],
    #     gem_dictionary["title_list"],
    #     gem_dictionary["plot_path"],
    #     gem_dictionary["set_threads"],
    #     gem_dictionary["set_n2"],
    #     gem_dictionary["runtime"],
    #     False
    # )
    # p = os.path.join("plot", "gemver2.json")
    # with open(p) as f:
    #     gem_dictionary = json.load(f)
    # plotting.plotting_fun(
    #     gemver2,
    #     gem_dictionary["mpi_implementations"],
    #     gem_dictionary["cuda_implementations"],
    #     gem_dictionary["serial_implementations"],
    #     gem_dictionary["open_implementations"],
    #     gem_dictionary["mpi_implementations_names"],
    #     gem_dictionary["cuda_implementations_names"],
    #     gem_dictionary["serial_implementations_names"],
    #     gem_dictionary["open_implementations_names"],
    #     gem_dictionary["threads"],
    #     gem_dictionary["N2"],
    #     gem_dictionary["N2_c"],
    #     gem_dictionary["filename_list"],
    #     gem_dictionary["title_list"],
    #     gem_dictionary["plot_path"],
    #     gem_dictionary["set_threads"],
    #     gem_dictionary["set_n2"],
    #     gem_dictionary["runtime"],
    # )
    # p = os.path.join("plot", "gemver.json")
    # with open(p) as f:
    #     gem_dictionary = json.load(f)
    # plotting.plotting_fun(
    #     gemver,
    #     gem_dictionary["mpi_implementations"],
    #     gem_dictionary["cuda_implementations"],
    #     gem_dictionary["serial_implementations"],
    #     gem_dictionary["open_implementations"],
    #     gem_dictionary["mpi_implementations_names"],
    #     gem_dictionary["cuda_implementations_names"],
    #     gem_dictionary["serial_implementations_names"],
    #     gem_dictionary["open_implementations_names"],
    #     gem_dictionary["threads"],
    #     gem_dictionary["N2"],
    #     gem_dictionary["N2_c"],
    #     gem_dictionary["filename_list"],
    #     gem_dictionary["title_list"],
    #     gem_dictionary["plot_path"],
    #     gem_dictionary["set_threads"],
    #     gem_dictionary["set_n2"],
    #     gem_dictionary["runtime"],
    #     True
    # )


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
