#!/usr/bin/env python3
from argparse import ArgumentParser, Action
from dataclasses import dataclass
import datetime
import itertools
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Literal, Optional, List, get_args

import numpy as np
from tabulate import tabulate


ParallelisationScheme = Literal["serial", "openmp", "mpi", "cuda"]


class SplitStrArgs(Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if isinstance(values, str):
            setattr(namespace, self.dest, [value for value in values.split(",")])


class SplitIntArgs(Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if isinstance(values, str):
            setattr(namespace, self.dest, [int(value) for value in values.split(",")])


@dataclass
class Binary:
    path: Path
    benchmark: str
    variant: str
    n: int
    scheme: ParallelisationScheme


@dataclass
class Result:
    variant: int
    scheme: ParallelisationScheme
    threads: int
    mean: float
    min: float
    max: float
    median: float
    std: float
    data: List[str]
    used_cached_results: bool


def get_script_dir() -> Path:
    return Path(os.path.realpath(__file__)).parent


def get_benchmark_dir(benchmark: str) -> Path:
    return get_script_dir() / benchmark


def get_variant_dir(benchmark: str, variant: str) -> Path:
    return get_benchmark_dir(benchmark) / variant


def get_results_dir() -> Path:
    return get_script_dir() / "results"


def get_results_variant_dir(benchmark: str, variant: str) -> Path:
    return get_results_dir() / benchmark / variant


def compile(benchmark: str, variant: str, cached_bins: bool, n: int, tsteps: int) -> Binary:
    script_dir = get_script_dir()
    os.makedirs(script_dir / "bin", exist_ok=True)
    bin_path = script_dir / "bin" / f"{benchmark}_{variant}_{n}"

    pb_generic_compunits = (script_dir / "polybench.c",)
    benchmark_dir = get_benchmark_dir(benchmark)
    assert benchmark_dir.exists()

    scheme = variant.split("_")[0]
    assert scheme in get_args(ParallelisationScheme)

    variant_dir = get_variant_dir(benchmark, variant)
    assert variant_dir.exists()

    compunit_filter = lambda files: filter(lambda fp: fp.suffix in (".c", ".cu"), files)
    benchmark_files = compunit_filter(benchmark_dir.iterdir())
    variant_files = compunit_filter(variant_dir.iterdir())
    compunits = itertools.chain(pb_generic_compunits, benchmark_files, variant_files)

    if cached_bins and bin_path.exists():
        return Binary(bin_path, benchmark, variant, n, scheme)
    args = [
        {
            "serial": "gcc",
            "openmp": "gcc",
            "mpi": "mpicc",
            "cuda": "nvcc",
        }[scheme],
        # '-fsanitize=address',
        # '-fsanitize=undefined',
        # '-fsanitize=leak',
        "-O3",
        "-o",
        str(bin_path),
        "-I",
        str(script_dir),
        "-I",
        str(benchmark_dir),
        "-DPOLYBENCH_TIME",
        "-DPOLYBENCH_DUMP_ARRAYS",
        f"-DN={n}",
        f"-DTSTEPS={tsteps}",
    ] + list(map(str, compunits))
    if scheme != "cuda":
        args.extend(("-Wall","-Wextra"))
    if scheme == "openmp":
        args.append("-fopenmp")
    if scheme != "cuda":
        args.append("-ffast-math")
        args.append("-march=native")
    print(' '.join(args))
    subprocess.check_call(args)
    return Binary(bin_path, benchmark, variant, n, scheme)


def format_fstr(loc: dict[str, Any], fstr: str) -> str:
    return fstr.format(
        **loc
        | {
            "ts": time.time(),
            "iso8601": datetime.datetime.now().isoformat(),
            "script_dir": get_script_dir(),
            "benchmark_dir": get_benchmark_dir(loc["binary"].benchmark),
            "variant_dir": get_variant_dir(
                loc["binary"].benchmark, loc["binary"].variant
            ),
            "results_variant_dir": get_results_variant_dir(
                loc["binary"].benchmark, loc["binary"].variant
            ),
        }
    )


def run(
    binary: Binary,
    threads: int,
    runs: int,
    cached_results: bool,
    result_fp_fstring: Optional[str] = None,
    latest_results_fstr: Optional[str] = None,
    ground_truth: Optional[str] = None,
) -> Result:

    # Find arguments for running the benchmark
    if binary.scheme == "mpi":
        args = ["mpiexec", "-n", str(threads), str(binary.path)]
        env = {}
    elif binary.scheme == "openmp":
        args = [str(binary.path)]
        env = {"OMP_NUM_THREADS": str(threads)}
    else:
        args = [str(binary.path)]
        env = {}

    # Run the benchmark
    timing_results = []
    data_outputs = []

    # Read results from cache, if specified
    use_cached_results = latest_results_fstr is not None
    if use_cached_results:  # done to prevent giving None to format_fstr
        cached_results_path = Path(format_fstr(locals(), latest_results_fstr))

    used_cached_results = False

    if use_cached_results and cached_results and cached_results_path.exists():
        with open(cached_results_path, "r") as results_file:
            timing_results = json.load(results_file)["timing"]
            used_cached_results = True
    else:
        # Run the benchmark

        for _ in range(runs):
            process = subprocess.Popen(
                args, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            exit_code = process.wait()
            if exit_code != 0:
                raise ValueError(f"Exit code {exit_code} for {binary.path}")
            if process.stdout is None:
                raise ValueError("No process timing output found")
            if process.stderr is None:
                raise ValueError("No process data output found")
            timing_results.append(float(process.stdout.read()))
            data_outputs.append(process.stderr.read())
            if ground_truth is not None and ground_truth != data_outputs[-1]:
                print(data_outputs[-1])
                raise ValueError(f"Discrepancy between results - {binary.path}")

        if result_fp_fstring is not None:
            curr_result_fp = Path(format_fstr(locals(), result_fp_fstring))
            result_path_dir = curr_result_fp.parent
            if not result_path_dir.exists():
                os.makedirs(result_path_dir, exist_ok=True)
            with open(
                curr_result_fp,
                "w+",
            ) as result_json_file:
                dump_contents = {"timing": timing_results, "data": data_outputs}
                json.dump(dump_contents, result_json_file)

            latest_results_parent = cached_results_path.parent
            if not latest_results_parent.exists():
                os.makedirs(cached_results_path)
            if cached_results_path.exists():
                cached_results_path.unlink()
            cached_results_path.symlink_to(curr_result_fp, target_is_directory=False)

    return Result(
        binary.variant,
        binary.scheme,
        threads,
        float(np.mean(timing_results)),
        float(np.min(timing_results)),
        float(np.max(timing_results)),
        float(np.median(timing_results)),
        float(np.std(timing_results)),
        data_outputs,
        used_cached_results,
    )


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Testing infrastructure for the DPHPC project, HS2024"
    )
    parser.add_argument(
        "--benchmark", type=str, required=True, help="The benchmark to run"
    )
    parser.add_argument(
        "--variants", action=SplitStrArgs, required=True, help="The variants to run"
    )
    parser.add_argument(
        "--threads",
        action=SplitIntArgs,
        default=[1, 2, 4, 8],
        help="Numbers of threads to run with (tries each)",
    )
    parser.add_argument(
        "--runs", type=int, default=10, help="The number of runs to perform"
    )
    parser.add_argument(
        "--n", type=int, default=400, help="The number passed as N to the kernel"
    )
    parser.add_argument(
        "--tsteps", type=int, default=100, help="The number passed as TSTEPS to the kernel"
    )
    parser.add_argument(
        "--cached_bins",
        action="store_true",
        default=False,
        help="Do not compile binaries if compiled previously",
    )
    parser.add_argument(
        "--cached_results",
        action="store_true",
        default=False,
        help="Do not run benchmarks if ran previously",
    )

    parser.add_argument(
        "--results_fstr",
        type=str,
        default="{results_variant_dir}/{iso8601}_{binary.n}_{threads}.json",
        help="Format string for raw timing result JSON filepath",
    )

    parser.add_argument(
        "--latest_results_fstr",
        type=str,
        default="{results_variant_dir}/latest_{binary.n}_{threads}.json",
        help="Format string for raw timing result JSON filepath",
    )

    args = parser.parse_args()
    variants = set(args.variants)

    ground_truth_bin = compile(args.benchmark, "serial_base", False, args.n, args.tsteps)
    ground_truth = run(ground_truth_bin, 1, 1, False)
    ground_truth_data = ground_truth.data[0]

    binaries = [
        compile(args.benchmark, variant, args.cached_bins, args.n, args.tsteps)
        for variant in variants
    ]

    results = [
        run(
            binary,
            thread_num,
            args.runs,
            args.cached_results,
            args.results_fstr,
            args.latest_results_fstr,
            ground_truth_data,
        )
        for thread_num in args.threads
        for binary in binaries
    ]

    print(
        tabulate(
            [
                (
                    result.variant,
                    result.scheme,
                    result.threads,
                    result.mean,
                    result.min,
                    result.max,
                    result.median,
                    result.std,
                    result.used_cached_results,
                )
                for result in [ground_truth] + results
            ],
            headers=[
                "Variant",
                "Parallelisation Scheme",
                "Threads",
                "Mean",
                "Min",
                "Max",
                "Median",
                "Stdev",
                "Used cached results",
            ],
        )
    )
