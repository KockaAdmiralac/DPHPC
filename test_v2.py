from argparse import ArgumentParser
import json
import random
import traceback
from typing import List

import marshmallow_dataclass
import output
import preparation
import runner
import single_benchmark
from structures import *
import result_processing


class BenchmarkRunner:
    r: runner.Runner

    benchmark_config: BenchmarkConfiguration
    prep: PreparationResult
    notified_finished_must_completes: bool = False
    results: List[ProcessedResult] = []
    ci_not_yet_tight: List[SingleBenchmark] = []
    notified_finished_all_cis: bool = False

    def __init__(self, benchmark_config: BenchmarkConfiguration) -> None:
        self.benchmark_config = benchmark_config

    def disqualify_benchmark(self, sb: SingleBenchmark):
        if sb in self.ci_not_yet_tight:
            self.ci_not_yet_tight.remove(sb)
        while sb in self.prep.must_completes:
            self.prep.must_completes.remove(sb)
        if sb in self.prep.benchmark_choices:
            self.prep.benchmark_choices.remove(sb)

    def select_next_benchmark(self) -> SingleBenchmark | None:
        grouped_by_sb = result_processing.group_runs(self.results)
        if len(self.prep.must_completes):
            idx = random.randint(0, len(self.prep.must_completes) - 1)
            item = self.prep.must_completes[idx]
            prior_runs = self.get_runs_by(grouped_by_sb, item)
            del self.prep.must_completes[idx]
            times_ran = len(prior_runs["kernel_time"]) if prior_runs is not None else 0
            print(f"Ran {times_ran} times so far")
            return item

        if not self.notified_finished_must_completes:
            print("Completed minimum required runs")
            self.notified_finished_must_completes = True

        # If I'm to follow the CI tightness requirement then sample from among these
        if self.prep.also_require_ci:
            while True:
                if (
                    not len(self.ci_not_yet_tight)
                    and not self.notified_finished_all_cis
                ):
                    print("All benchmarks have satisfactory CIs")
                    self.notified_finished_all_cis = True
                    break
                elif self.notified_finished_all_cis:
                    break
                else:
                    candidate_sb = random.choice(self.ci_not_yet_tight)
                    existing_runs = self.get_runs_by(grouped_by_sb, candidate_sb)
                    if existing_runs is None or len(existing_runs["kernel_time"]) < 2:
                        # nothing to compute statistics over, so run the benchmark to collect results
                        # bootstrap method needs at least two values
                        return candidate_sb
                    else:
                        # now check current CI because a previous run may have made this tight enough
                        existing_runs_timings = existing_runs["kernel_time"]
                        stat_func = getattr(np, self.prep.ci_statistic)
                        np_existing_timings = np.array(existing_runs_timings)
                        ci = result_processing.get_ci(
                            np_existing_timings, statistic=stat_func
                        )
                        measured_val = stat_func(np_existing_timings)
                        ci_lower_permitted = measured_val * (
                            1 - self.prep.ci_max_dev_from_plain_stat
                        )
                        ci_upper_permitted = measured_val * (
                            1 + self.prep.ci_max_dev_from_plain_stat
                        )
                        print(
                            f"CI is ({ci.low}, {ci.high}), measured stat {measured_val}, permitted CI is ({ci_lower_permitted}, {ci_upper_permitted})"
                        )
                        if ci.low < ci_lower_permitted or ci_upper_permitted < ci.high:
                            # this candidate still has a too-broad CI
                            prior_runs = self.get_runs_by(grouped_by_sb, candidate_sb)
                            times_ran = (
                                len(prior_runs["kernel_time"])
                                if prior_runs is not None
                                else 0
                            )
                            print(f"Ran {times_ran} times so far")
                            return candidate_sb
                        else:
                            # this one is already satisfying the CI tightness
                            print(f"CI tight enough")
                            self.ci_not_yet_tight.remove(candidate_sb)

        if self.prep.keep_going:
            item = random.choice(self.prep.benchmark_choices)
            prior_runs = self.get_runs_by(grouped_by_sb, item)
            times_ran = len(prior_runs["kernel_time"]) if prior_runs is not None else 0
            print(f"Ran {times_ran} times so far")
            return item
        else:
            return None

    def get_runs_by(
        self,
        grouped_by_sb: result_processing.PreprocessedResults,
        sb_searched: SingleBenchmark,
    ):
        return next((sb[1] for sb in grouped_by_sb if sb[0] == sb_searched), None)

    def run_single_benchmark(self, b: SingleBenchmark) -> ProcessedResult:
        try:
            return single_benchmark.run_benchmark(b, self.prep)
        except Exception as e:
            print(f"Couldn't run {b.compile_settings.binary_path}")
            raise e

    def main_run_loop(self, r: runner.Runner) -> List[ProcessedResult]:
        while True:
            try:
                next_benchmark = self.select_next_benchmark()
                if next_benchmark is None:
                    break
                res = self.run_single_benchmark(next_benchmark)

                if (res.data_checked and not res.data_valid) or (
                    res.raw_result is not None and res.raw_result.exit_code != 0
                ):
                    self.disqualify_benchmark(next_benchmark)
                    print(f"Disqualified {next_benchmark}")

                self.results.append(res)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(
                    traceback.format_exc()
                )  # intended to minimise damage of test_v2.py implementation bugs

        return self.results

    def main_run(self, args):
        r = runner.Runner()
        self.prep = preparation.all_prepare(r, self.benchmark_config)
        if self.prep.also_require_ci:
            self.ci_not_yet_tight = [c for c in self.prep.benchmark_choices]
        results = self.main_run_loop(r)
        if args.results_output is not None:
            procres_schema = marshmallow_dataclass.class_schema(ProcessedResult)()
            results_pyobj = list(map(procres_schema.dump, results))
            with open(args.results_output, "w+") as f:
                json.dump(results_pyobj, f, indent=4)
        if args.output is not None:
            print(args.output)
            preproc_results = result_processing.preprocess_results(results)
            output.run_output(preproc_results, args.output)
        return results


if __name__ == "__main__":
    # So far you start the program by copying the output of gen_benchmark_config.py in place of BenchmarkConfiguration here.
    # In future you have the option of loading this configuration from a JSON file.

    parser = ArgumentParser(
        description="Runs a set of benchmarks following a provided configuration."
    )

    parser.add_argument(
        "--config",
        required=True,
        help="A configuraion file in JSON format, normally generated by gen_benchmark_config.py",
    )

    parser.add_argument(
        "--results-output", help="Where to save the JSON list of ProcessedResults"
    )

    parser.add_argument(
        "--output",
        type=str,
        choices=list(output.output_modes.keys()),
        help="How you would like to output the results",
        action="append",
    )

    args = parser.parse_args()

    with open(args.config, "r") as f:
        raw_benchmark_config = json.load(f)

    benchmark_config_schema = marshmallow_dataclass.class_schema(
        BenchmarkConfiguration
    )()
    bc: BenchmarkConfiguration = benchmark_config_schema.load(raw_benchmark_config)  # type: ignore

    br = BenchmarkRunner(bc)
    br.main_run(args)
