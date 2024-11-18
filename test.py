#!/usr/bin/env python3
from argparse import Action, ArgumentParser, Namespace
from copy import deepcopy
from dataclasses import dataclass
import datetime
import itertools
import json
import os
from pathlib import Path
from shutil import which
import subprocess
import sys
import time
from typing import Any, Literal, Optional, get_args

import numpy as np
from tabulate import tabulate

import datacheck
import options

ParallelisationScheme = Literal["serial", "openmp", "mpi", "cuda"]

mpiexec_path = None


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
    opts: options.Options
    scheme: ParallelisationScheme
    defines: dict[str, str]


@dataclass
class Result:
    variant: str
    scheme: ParallelisationScheme
    threads: int
    mean: float
    min: float
    max: float
    median: float
    std: float
    mean_deviation: float
    min_deviation: float
    max_deviation: float
    median_deviation: float
    std_deviation: float
    data: datacheck.ParsedOutputData
    used_cached_results: bool


def get_script_dir() -> Path:
    return Path(os.path.realpath(__file__)).parent


def get_benchmark_dir(benchmark: str) -> Path:
    return get_script_dir() / benchmark


def get_variant_dir(benchmark: str, variant: str) -> Path:
    return get_benchmark_dir(benchmark) / variant


def get_results_dir() -> Path:
    return get_script_dir() / "results"


def get_results_benchmark_dir(benchmark: str) -> Path:
    return get_results_dir() / benchmark


def get_results_variant_dir(benchmark: str, variant: str) -> Path:
    return get_results_benchmark_dir(benchmark) / variant


def check_defines_constraints(
    defines_constraints: options.DefinesConstraints, merged_defines: dict[str, str]
) -> None:
    for constraint in defines_constraints:
        k = constraint["define"]
        assert type(k) == str
        if k in merged_defines:
            val = merged_defines[k]
            if "max" in constraint or "min" in constraint:
                val_int = int(val)
                if "max" in constraint:
                    assert type(constraint["max"]) == int
                    if val_int > constraint["max"]:
                        raise ValueError(
                            f"Value {val} for define {k} is beyond {constraint['max']}"
                        )
                if "min" in constraint:
                    assert type(constraint["min"]) == int
                    if val_int < constraint["min"]:
                        raise ValueError(
                            f"Value {val} for define {k} is below {constraint['min']}"
                        )


# https://stackoverflow.com/a/5656097
def intersperse_elem_in_list(iterable, delimiter):
    try:
        it = iter(iterable)
        for x in it:
            yield delimiter
            yield x
    except StopIteration:
        return


def compile(
    benchmark: str,
    variant: str,
    cached_bins: bool,
    defines: dict[str, str],
    human_readable_output: bool = False,
    disable_checking: bool = False,
) -> Binary:
    script_dir = get_script_dir()

    pb_generic_compunits = (script_dir / "polybench.c",)
    benchmark_dir = get_benchmark_dir(benchmark)
    assert benchmark_dir.exists()

    scheme = variant.split("_")[0]
    assert scheme in get_args(ParallelisationScheme)

    variant_dir = get_variant_dir(benchmark, variant)
    if not variant_dir.exists():
        raise AssertionError(f"{variant_dir} doesn't exist")

    opt = load_options(benchmark, variant)

    merged_defines = deepcopy(opt.defines)
    merged_defines.update(
        defines
    )  # manually provided defines via CLI override ones in JSON

    if human_readable_output:
        merged_defines["DUMP_DATA_HUMAN_READABLE"] = ""

    if disable_checking:
        merged_defines["DISABLE_CHECKING"] = ""

    check_defines_constraints(opt.defines_constraints, merged_defines)

    os.makedirs(script_dir / "bin", exist_ok=True)
    bin_path = (
        script_dir / "bin" / f"{benchmark}_{variant}_{serialise_defines(defines)}"
    )

    def compunit_filter(files):
        return filter(lambda fp: fp.suffix in (".c", ".cu", ".cpp"), files)

    source_dirs = itertools.chain(
        [benchmark_dir, variant_dir], get_abs_paths(opt.extra_source_dirs)
    )
    raw_source_files = itertools.chain(*map(lambda d: d.iterdir(), source_dirs))
    compunits_unfiltered = itertools.chain(
        pb_generic_compunits, map(str, compunit_filter(raw_source_files))
    )

    exclude_sources_abs = list(map(str, get_abs_paths(opt.exclude_sources)))
    compunits = filter(lambda src: src not in exclude_sources_abs, compunits_unfiltered)

    includes = itertools.chain(
        [str(variant_dir), str(script_dir), str(benchmark_dir)],
        map(str, get_abs_paths(opt.extra_includes)),
    )
    includes_args = intersperse_elem_in_list(includes, "-I")

    if cached_bins and bin_path.exists():
        return Binary(bin_path, benchmark, variant, opt, scheme, defines)

    defines_args = map(
        lambda it: f"-D{it[0]}={it[1]}".rstrip("="), merged_defines.items()
    )  # rstrip makes sure defines without values don't have =
    extra_options = (
        opt.extra_compile_options if opt.extra_compile_options is not None else []
    )
    args = [
        *{
            "serial": ("gcc",),
            "openmp": ("gcc",),
            "mpi": ("mpicc",),
            "cuda": (
                "docker",
                "run",
                "-it",
                "--rm",
                "--mount",
                "type=bind,src=/home/paolo/repos/ethz/dphpc/dphpc-project,target=/project",
                "dphpc-cuda:8.0-devel-ubuntu16.04",
                "nvcc",
                "-Wno-deprecated-gpu-targets",
            ),
        }[scheme],
        "-std=c++11" if scheme == "cuda" else "-std=c11",
        "-O3",
        "-o",
        str(bin_path),
        *includes_args,
        *defines_args,
        *extra_options,
    ] + list(map(str, compunits))
    if scheme == "cuda":
        args.extend(
            (
                "--use_fast_math",
                "-DCUDA_MODE",
            )
        )
    else:
        args.extend(("-Wall", "-Wextra", "-ffast-math", "-march=native"))
    if scheme == "openmp":
        args.append("-fopenmp")

    print(" ".join(args))
    subprocess.check_call(args)
    return Binary(bin_path, benchmark, variant, opt, scheme, defines)


def get_abs_paths(paths):
    return map(lambda p: Path(p).resolve(), paths)


def load_options(benchmark: str, variant: str) -> options.Options:
    return options.options_from_multiple_files(
        filter(
            Path.exists,
            (
                get_script_dir() / "dphpc_md.json",
                get_benchmark_dir(benchmark) / "dphpc_md.json",
                get_variant_dir(benchmark, variant) / "dphpc_md.json",
            ),
        )
    )


def format_fstr(
    loc: dict[str, Any],
    fstr: str,
    benchmark: Optional[str] = None,
    variant: Optional[str] = None,
    defines: Optional[dict[str, str]] = None,
) -> str:
    helper_dict = {
        "ts": time.time(),
        "iso8601": datetime.datetime.now().isoformat(),
        "script_dir": get_script_dir(),
    }
    if benchmark is not None:
        helper_dict["benchmark_dir"] = get_benchmark_dir(benchmark)
        helper_dict["results_benchmark_dir"] = get_results_benchmark_dir(benchmark)
        if variant is not None:
            helper_dict["variant_dir"] = get_variant_dir(benchmark, variant)
            helper_dict["results_variant_dir"] = get_results_variant_dir(
                benchmark, variant
            )

    if defines is not None:
        helper_dict["ser_defines"] = serialise_defines(defines)

    return fstr.format(**loc | helper_dict)


def serialise_defines(defines: dict[str, str]) -> str:
    return "".join(map(lambda it: it[0] + it[1], defines.items()))


def lowlevel_run(binary: Binary, threads: int) -> tuple[float, bytes]:
    global mpiexec_path
    # Find arguments for running the benchmark
    external_env = dict(os.environ)
    if binary.scheme == "mpi":
        if mpiexec_path is None:
            mpiexec_path = which("mpiexec")
        if mpiexec_path is None:
            raise ValueError("mpiexec not found! Please install an MPI library.")
        args = [mpiexec_path, "-n", str(threads), str(binary.path)]
        env = {}
    elif binary.scheme == "openmp":
        args = [str(binary.path)]
        env = {"OMP_NUM_THREADS": str(threads)}
    else:
        args = [str(binary.path)]
        env = {}
    env.update(external_env)
    process = subprocess.Popen(
        args,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False,
        shell=False,
    )
    (stdout, stderr) = process.communicate()
    exit_code = process.returncode
    if exit_code != 0:
        print(stderr, file=sys.stderr)
        raise ValueError(f"Exit code {exit_code} for {binary.path}")
    time_taken = float(stdout)
    binary_data = stderr
    return (time_taken, binary_data)


def check_results_or_log_failure(
    binary: Binary,
    threads: int,
    ground_truth: datacheck.ParsedOutputData,
    failed_data_out_fstr: str,
    candidate_data: datacheck.ParsedOutputData,
    deviations_log: np.ndarray,
    raw_binary_data: bytes,
) -> None:
    try:
        deviations_log = np.append(
            deviations_log,
            datacheck.compare_results(
                ground_truth,
                candidate_data,
                mode=binary.opts.data_check,
                max_deviation=binary.opts.max_deviation,
            ),
        )
    except Exception as match_err:
        if failed_data_out_fstr is not None:
            failed_out_fp = format_and_provide_outpath(
                locals(),
                failed_data_out_fstr,
                benchmark=binary.benchmark,
                variant=binary.variant,
                defines=binary.defines,
            )
            with open(failed_out_fp, "wb+") as failed_out:
                failed_out.write(raw_binary_data)
            print(f"Wrote bad data to {failed_out_fp}")
        else:
            print(raw_binary_data)
        print(f"Discrepancy between results - {binary.path}")
        raise match_err


def log_results(
    binary: Binary,
    threads: int,
    result_fp_fstring: str,
    timing_results: list[float],
    deviations: np.ndarray,
    cached_results_path: Path,
) -> None:
    curr_result_fp = format_and_provide_outpath(
        locals(),
        result_fp_fstring,
        benchmark=binary.benchmark,
        variant=binary.variant,
        defines=binary.defines,
    )
    with open(
        curr_result_fp,
        "w+",
    ) as result_json_file:
        dump_contents = {
            "timing": timing_results,
            "deviations": deviations.tolist(),
        }
        json.dump(dump_contents, result_json_file)

    latest_results_parent = cached_results_path.parent
    if not latest_results_parent.exists():
        os.makedirs(cached_results_path)
    if cached_results_path.exists():
        cached_results_path.unlink()
    cached_results_path.symlink_to(curr_result_fp, target_is_directory=False)


def run(
    binary: Binary,
    threads: int,
    runs: int,
    cached_results: bool,
    result_fp_fstring: Optional[str] = None,
    latest_results_fstr: Optional[str] = None,
    ground_truth: Optional[datacheck.ParsedOutputData] = None,
    ground_truth_out_fstr: Optional[str] = None,
    failed_data_out_fstr: Optional[str] = None,
    disable_checking: bool = False,
    human_readable_output: bool = False,
) -> Result:

    # Run the benchmark
    timing_results = []
    data_output: datacheck.ParsedOutputData = {}
    deviations = np.empty(1)

    # Read results from cache, if specified
    did_use_cached_results = False
    cached_results_path = None
    if latest_results_fstr is not None:  # done to prevent giving None to format_fstr
        cached_results_path = format_and_provide_outpath(
            locals(),
            latest_results_fstr,
            benchmark=binary.benchmark,
            variant=binary.variant,
            defines=binary.defines,
        )
        did_use_cached_results = cached_results and cached_results_path.exists()
        if did_use_cached_results:
            with open(cached_results_path, "r") as results_file:
                timing_results = json.load(results_file)["timing"]

    if not did_use_cached_results:
        # Run the benchmark

        for _ in range(runs):

            (time_taken, raw_binary_data) = lowlevel_run(binary, threads)

            timing_results.append(time_taken)
            if not disable_checking:
                data_output = datacheck.parse_dump_to_arrays(
                    raw_binary_data, is_human_readable=human_readable_output
                )

                if (
                    binary.variant == "serial_base"
                    and ground_truth_out_fstr is not None
                ):
                    truth_out_fp = format_and_provide_outpath(
                        locals(),
                        ground_truth_out_fstr,
                        benchmark=binary.benchmark,
                        variant=binary.variant,
                        defines=binary.defines,
                    )
                    with open(truth_out_fp, "wb+") as truth_out:
                        truth_out.write(raw_binary_data)
                    print(f"Wrote ground truth data to {truth_out_fp}")

                if ground_truth is not None and failed_data_out_fstr is not None:
                    check_results_or_log_failure(
                        binary,
                        threads,
                        ground_truth,
                        failed_data_out_fstr,
                        data_output,
                        deviations,
                        raw_binary_data,
                    )

        if result_fp_fstring is not None and cached_results_path is not None:
            log_results(
                binary,
                threads,
                result_fp_fstring,
                timing_results,
                deviations,
                cached_results_path,
            )

    if deviations == np.empty(1):
        deviations = np.empty(
            1
        )  # just here to prevent np complaining when running base case aka without ground truth yet.
    return Result(
        binary.variant,
        binary.scheme,
        threads,
        float(np.mean(timing_results)),
        float(np.min(timing_results)),
        float(np.max(timing_results)),
        float(np.median(timing_results)),
        float(np.std(timing_results)),
        float(np.mean(deviations)),
        float(np.min(deviations)),
        float(np.max(deviations)),
        float(np.median(deviations)),
        float(np.std(deviations)),
        data_output,
        did_use_cached_results,
    )


def format_and_provide_outpath(
    loc: dict[str, Any],
    result_fp_fstring: str,
    benchmark: Optional[str] = None,
    variant: Optional[str] = None,
    defines: Optional[dict[str, str]] = None,
) -> Path:
    fp = Path(
        format_fstr(
            loc,
            result_fp_fstring,
            benchmark=benchmark,
            variant=variant,
            defines=defines,
        )
    )
    parent_dir = fp.parent
    if not parent_dir.exists():
        os.makedirs(parent_dir, exist_ok=True)
    return fp


def main_run(args: Namespace) -> None:
    variants = set(args.variants)

    defines = dict([x.split("=")[:2] for x in args.set_defines.split(",")])

    if args.disable_checking:
        print("\033[93mWARNING: all checking disabled\033[0m", file=sys.stderr)

    ground_truth_data = None
    if not args.disable_checking:
        truth_fp = format_and_provide_outpath(
            locals(),
            args.ground_truth_out_fstr,
            benchmark=args.benchmark,
            defines=defines,
        )
        if args.require_recompute_ground_truth or not truth_fp.exists():
            ground_truth_bin = compile(
                args.benchmark,
                "serial_base",
                False,
                defines,
                human_readable_output=args.human_readable_output,
            )
            ground_truth = run(
                ground_truth_bin,
                1,
                1,
                False,
                ground_truth_out_fstr=args.ground_truth_out_fstr,
                human_readable_output=args.human_readable_output,
            )
            ground_truth_data = ground_truth.data
        else:
            with open(truth_fp, "rb") as truth_f:
                ground_truth_data = datacheck.parse_dump_to_arrays(truth_f.read())

    binaries = [
        compile(
            args.benchmark,
            variant,
            args.cached_bins,
            defines,
            human_readable_output=args.human_readable_output,
            disable_checking=args.disable_checking,
        )
        for variant in variants
    ]

    results = [
        run(
            binary,
            thread_num,
            args.runs,
            args.cached_results,
            args.results_fstr if args.results_fstr != "" else None,
            args.latest_results_fstr if args.latest_results_fstr != "" else None,
            ground_truth_data,
            failed_data_out_fstr=(
                args.failed_data_out_fstr if args.failed_data_out_fstr != "" else None
            ),
            disable_checking=args.disable_checking,
            human_readable_output=args.human_readable_output,
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
                    result.mean_deviation,
                    result.min_deviation,
                    result.max_deviation,
                    result.median_deviation,
                    result.std_deviation,
                    result.used_cached_results,
                )
                for result in results
            ],
            headers=[
                "Variant",
                "Parallelisation Scheme",
                "Threads",
                "Mean time",
                "Min time",
                "Max time",
                "Median time",
                "Stdev time",
                "Mean dev",
                "Min dev",
                "Max dev",
                "Median dev",
                "Stdev dev",
                "Used cached results",
            ],
        )
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
        default=[1],
        help="Numbers of threads to run with (tries each)",
    )
    parser.add_argument(
        "--runs", type=int, default=10, help="The number of runs to perform"
    )
    parser.add_argument(
        "--set_defines",
        type=str,
        required=True,
        help="Set defines yourself, notably N2/TSTEPS.  Comma-separated and mandatory =",
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
        default="{results_variant_dir}/{iso8601}_{ser_defines}_{threads}.json",
        help="Format string for raw timing result JSON filepath",
    )

    parser.add_argument(
        "--latest_results_fstr",
        type=str,
        default="{results_variant_dir}/latest_{ser_defines}_{threads}.json",
        help="Format string for raw timing result JSON filepath",
    )

    parser.add_argument(
        "--ground_truth_out_fstr",
        type=str,
        default="{results_benchmark_dir}/{ser_defines}_truth",
        help="Format string for raw output from the ground truth case",
    )

    parser.add_argument(
        "--failed_data_out_fstr",
        type=str,
        default="{results_variant_dir}/failed_{ser_defines}_{threads}",
        help="Format string for raw output from a failed run",
    )

    parser.add_argument(
        "--require_recompute_ground_truth",
        action="store_true",
        help="Do not try using an existing ground truth dataset if one exists",
    )

    parser.add_argument(
        "--disable_checking",
        action="store_true",
        help="Disable all ouput data checking",
    )

    parser.add_argument(
        "--human_readable_output",
        action="store_true",
        help="Dump data output in human-readable format",
    )

    args = parser.parse_args()
    main_run(args)
