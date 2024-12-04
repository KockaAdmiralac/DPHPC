from collections.abc import Callable
import copy
import os
from shutil import which
import subprocess
from typing import List

import numpy as np
import datacheck
from structures import *


def lowlevel_run(
    arg_maker: Callable[[], List[str]],
    env_maker: Callable[[dict[str, str]], dict[str, str]],
) -> RawResult:
    external_env = dict(os.environ)
    args = arg_maker()
    env = env_maker(external_env)
    print(f"Running benchmark with args {' '.join(args)}")
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
    return RawResult(raw_stdout=stdout, raw_stderr=stderr, exit_code=exit_code)


def check_results_or_log_failure(
    b: SingleBenchmark,
    ground_truth: datacheck.ParsedOutputData,
    candidate_data: datacheck.ParsedOutputData,
) -> Tuple[bool, Optional[np.ndarray]]:
    try:
        return (
            True,
            datacheck.compare_results(
                ground_truth,
                candidate_data,
                mode=b.compile_settings.dphpc_opts.data_check,
                max_deviation=b.compile_settings.dphpc_opts.max_deviation,
            ),
        )
    except Exception as match_err:
        print(f"Discrepancy between results - {b}")
        print(match_err)
        # raise match_err
        return (False, None)


def parse_run_output(
    raw_result: RawResult,
    parse_data: bool = True,
    is_data_human_readable: bool | None = None,
) -> Tuple[dict[str, float], ParsedOutputData | None]:
    timings = {"kernel_time": float(raw_result.raw_stdout)}
    if parse_data:
        data_output = datacheck.parse_dump_to_arrays(
            raw_result.raw_stderr, is_human_readable=is_data_human_readable
        )
    else:
        data_output = None
    return (timings, data_output)


def run_benchmark(b: SingleBenchmark, prep: PreparationResult) -> ProcessedResult:
    scheme = b.compile_settings.scheme
    if scheme == "mpi":
        mpiexec_path = which("mpiexec")
        if mpiexec_path is None:
            raise ValueError("mpiexec not found! Please install an MPI library.")
        args = [
            mpiexec_path,
            "-n",
            str(b.run_options.threads),
            str(b.compile_settings.binary_path),
        ]
        env = {}
    elif scheme == "openmp":
        args = [str(b.compile_settings.binary_path)]
        env = {"OMP_NUM_THREADS": str(b.run_options.threads)}
    else:
        args = [str(b.compile_settings.binary_path)]
        env = {}
    arg_maker = lambda: args

    def env_maker(prior_env: dict[str, str]) -> dict[str, str]:
        new_env = copy.deepcopy(env)
        new_env.update(prior_env)
        return new_env

    raw_res = lowlevel_run(arg_maker, env_maker)

    res = ProcessedResult()

    if prep.save_raw_outputs:
        res.raw_result = raw_res

    res.timings, data_output = parse_run_output(
        raw_res,
        parse_data=not b.variant_config.compile_options.disable_checking
        or prep.save_parsed_output_data,
        is_data_human_readable=b.variant_config.compile_options.human_readable_output,
    )
    if prep.save_parsed_output_data:
        res.output_data = data_output

    if not b.variant_config.compile_options.disable_checking:
        res.data_valid, res.deviations = check_results_or_log_failure(
            b, prep.ground_truth_results[b.ground_truth_bin_path], data_output
        )

        res.data_checked = True
    return res
