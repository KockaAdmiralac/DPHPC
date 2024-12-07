from collections.abc import Callable
from dataclasses import dataclass
import itertools
from pathlib import Path
import random
from typing import Iterable, List
import compat
import datacheck
import runner
import single_benchmark
from structures import *


def prepare_compilation(conf: BenchmarkConfiguration):
    benchmark_choices: List[single_benchmark.SingleBenchmark] = []
    compilations: dict[Path, CompilationSettings] = {}
    ground_truth_compilations: dict[Path, CompilationSettings] = {}
    for benchmark, variants_in_benchmark in conf.benchmarks.items():
        for variant, variant_configs in variants_in_benchmark.items():
            for compile_variant in variant_configs:
                compsettings = compat.prepare_single_compilation(
                    benchmark,
                    variant,
                    compile_variant.compile_options,
                )

                ground_truth_compsettings = compat.prepare_single_compilation(
                    benchmark, "serial_base", compile_variant.compile_options
                )
                ground_truth_bin_path = ground_truth_compsettings.binary_path

                if ground_truth_bin_path not in ground_truth_compilations:
                    ground_truth_compilations[ground_truth_bin_path] = (
                        ground_truth_compsettings
                    )

                compilations[compsettings.binary_path] = compsettings
                for run_subvariant in compile_variant.run_options:
                    benchmark_choices.append(
                        single_benchmark.SingleBenchmark(
                            compile_variant,
                            run_subvariant,
                            compsettings,
                            ground_truth_bin_path,
                        )
                    )

    return (benchmark_choices, compilations, ground_truth_compilations)


ground_truth_run_func = Callable[[CompilationSettings], Tuple[Path, ParsedOutputData]]


def compile_run_ground_truth(
    gt_comps: Iterable[CompilationSettings],
) -> Iterable[Tuple[ground_truth_run_func, List[CompilationSettings]]]:
    def single_gt(gt: CompilationSettings) -> ParsedOutputData:
        compat.compile(gt)
        raw_res = single_benchmark.lowlevel_run(
            lambda: [
                str(gt.binary_path),
            ],
            lambda prior_env: prior_env,
            user_msg_fstr="Collecting ground truth with args {joined_args}",
        )
        assert (
            raw_res.exit_code == 0
        ), f"Ground truth failed to run, for binary {gt.binary_path} got exit code {raw_res.exit_code}"
        parsed = datacheck.parse_dump_to_arrays(
            raw_res.raw_stderr, is_human_readable=gt.orig_options.human_readable_output
        )
        return (gt.binary_path, parsed)

    for gt in gt_comps:
        yield (
            single_gt,
            [
                gt,
            ],
        )


def all_prepare(r: runner.Runner, conf: BenchmarkConfiguration) -> PreparationResult:
    benchmark_choices, compilations, ground_truth_compilations = prepare_compilation(
        conf
    )
    # print(ground_truth_compilations.values())
    print([c.binary_path for c in ground_truth_compilations.values()])
    compilation_tasks = [
        (
            compat.compile,
            [
                c,
            ],
        )
        for c in compilations.values()
    ]
    all_jobs = r.run_tasks(
        list(
            itertools.chain(
                compilation_tasks,
                compile_run_ground_truth(ground_truth_compilations.values()),
            )
        )
    )
    ground_truth_results: dict[Path, ParsedOutputData] = dict(
        itertools.islice(all_jobs, len(compilations), None)
    )
    print(ground_truth_results)
    prep = PreparationResult(
        benchmark_choices,
        compilations,
        benchmark_choices * conf.min_runs,
        ground_truth_results,
        conf.keep_going,
        conf.check_results_between_runs,
        conf.save_raw_outputs,
        conf.save_parsed_output_data,
    )
    random.shuffle(prep.must_completes)
    return prep
