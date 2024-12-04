from dataclasses import asdict, dataclass
import pprint
import random
from typing import List
import preparation
import runner
import single_benchmark
from structures import *


class BenchmarkRunner:
    r: runner.Runner

    benchmark_config: BenchmarkConfiguration
    prep: PreparationResult
    notified_finished_must_completes: bool = False

    def __init__(self, benchmark_config: BenchmarkConfiguration) -> None:
        self.benchmark_config = benchmark_config

    def select_next_benchmark(self) -> SingleBenchmark | None:
        if len(self.prep.must_completes):
            idx = random.randint(0, len(self.prep.must_completes) - 1)
            item = self.prep.must_completes[idx]
            del self.prep.must_completes[idx]
            return item

        if not self.notified_finished_must_completes:
            print("Completed minimum required runs")
            self.notified_finished_must_completes = True
        if self.prep.keep_going:
            return random.choice(self.prep.benchmark_choices)
        else:
            return None

    def run_single_benchmark(self, b: SingleBenchmark) -> ProcessedResult:
        try:
            return single_benchmark.run_benchmark(b, self.prep)
        except Exception as e:
            print(f"Couldn't run {b.compile_settings.binary_path}")
            raise e

    def main_run_loop(self, r: runner.Runner) -> List[ProcessedResult]:
        results: List[ProcessedResult] = []
        while True:
            try:
                next_benchmark = self.select_next_benchmark()
                if next_benchmark is None:
                    break
                res = self.run_single_benchmark(next_benchmark)

                results.append(res)
            except KeyboardInterrupt:
                break

        print(results)
        return results

    def main_run(self):
        r = runner.Runner()
        self.prep = preparation.all_prepare(r, self.benchmark_config)
        self.main_run_loop(r)


if __name__ == "__main__":
    # So far you start the program by copying the output of gen_benchmark_config.py in place of BenchmarkConfiguration here.
    # In future you have the option of loading this configuration from a JSON file.
    br = BenchmarkRunner(
        BenchmarkConfiguration(
            benchmarks={
                "adi": {
                    "mpi_1": [
                        VariantConfiguration(
                            compile_options=VariantCompilationOptions(
                                extra_defines={"N2": "10", "TSTEPS": "10"},
                                human_readable_output=False,
                                disable_checking=False,
                            ),
                            run_options=[
                                SubvariantRunOptions(subvariant_name=None, threads=1),
                                SubvariantRunOptions(subvariant_name=None, threads=2),
                            ],
                            variant_name="second " "variant " "name",
                        ),
                        VariantConfiguration(
                            compile_options=VariantCompilationOptions(
                                extra_defines={"N2": "10", "TSTEPS": "11"},
                                human_readable_output=False,
                                disable_checking=False,
                            ),
                            run_options=[
                                SubvariantRunOptions(subvariant_name=None, threads=1),
                                SubvariantRunOptions(subvariant_name=None, threads=2),
                            ],
                            variant_name="second " "variant " "name",
                        ),
                        VariantConfiguration(
                            compile_options=VariantCompilationOptions(
                                extra_defines={"N2": "10", "TSTEPS": "12"},
                                human_readable_output=False,
                                disable_checking=False,
                            ),
                            run_options=[
                                SubvariantRunOptions(subvariant_name=None, threads=1),
                                SubvariantRunOptions(subvariant_name=None, threads=2),
                            ],
                            variant_name="second " "variant " "name",
                        ),
                        VariantConfiguration(
                            compile_options=VariantCompilationOptions(
                                extra_defines={"N2": "10", "TSTEPS": "13"},
                                human_readable_output=False,
                                disable_checking=False,
                            ),
                            run_options=[
                                SubvariantRunOptions(subvariant_name=None, threads=1),
                                SubvariantRunOptions(subvariant_name=None, threads=2),
                            ],
                            variant_name="second " "variant " "name",
                        ),
                        VariantConfiguration(
                            compile_options=VariantCompilationOptions(
                                extra_defines={"N2": "10", "TSTEPS": "14"},
                                human_readable_output=False,
                                disable_checking=False,
                            ),
                            run_options=[
                                SubvariantRunOptions(subvariant_name=None, threads=1),
                                SubvariantRunOptions(subvariant_name=None, threads=2),
                            ],
                            variant_name="second " "variant " "name",
                        ),
                        VariantConfiguration(
                            compile_options=VariantCompilationOptions(
                                extra_defines={"N2": "11", "TSTEPS": "10"},
                                human_readable_output=False,
                                disable_checking=False,
                            ),
                            run_options=[
                                SubvariantRunOptions(subvariant_name=None, threads=1),
                                SubvariantRunOptions(subvariant_name=None, threads=2),
                            ],
                            variant_name="second " "variant " "name",
                        ),
                        VariantConfiguration(
                            compile_options=VariantCompilationOptions(
                                extra_defines={"N2": "11", "TSTEPS": "11"},
                                human_readable_output=False,
                                disable_checking=False,
                            ),
                            run_options=[
                                SubvariantRunOptions(subvariant_name=None, threads=1),
                                SubvariantRunOptions(subvariant_name=None, threads=2),
                            ],
                            variant_name="second " "variant " "name",
                        ),
                        VariantConfiguration(
                            compile_options=VariantCompilationOptions(
                                extra_defines={"N2": "11", "TSTEPS": "12"},
                                human_readable_output=False,
                                disable_checking=False,
                            ),
                            run_options=[
                                SubvariantRunOptions(subvariant_name=None, threads=1),
                                SubvariantRunOptions(subvariant_name=None, threads=2),
                            ],
                            variant_name="second " "variant " "name",
                        ),
                        VariantConfiguration(
                            compile_options=VariantCompilationOptions(
                                extra_defines={"N2": "11", "TSTEPS": "13"},
                                human_readable_output=False,
                                disable_checking=False,
                            ),
                            run_options=[
                                SubvariantRunOptions(subvariant_name=None, threads=1),
                                SubvariantRunOptions(subvariant_name=None, threads=2),
                            ],
                            variant_name="second " "variant " "name",
                        ),
                        VariantConfiguration(
                            compile_options=VariantCompilationOptions(
                                extra_defines={"N2": "11", "TSTEPS": "14"},
                                human_readable_output=False,
                                disable_checking=False,
                            ),
                            run_options=[
                                SubvariantRunOptions(subvariant_name=None, threads=1),
                                SubvariantRunOptions(subvariant_name=None, threads=2),
                            ],
                            variant_name="second " "variant " "name",
                        ),
                        VariantConfiguration(
                            compile_options=VariantCompilationOptions(
                                extra_defines={"N2": "12", "TSTEPS": "10"},
                                human_readable_output=False,
                                disable_checking=False,
                            ),
                            run_options=[
                                SubvariantRunOptions(subvariant_name=None, threads=1),
                                SubvariantRunOptions(subvariant_name=None, threads=2),
                            ],
                            variant_name="second " "variant " "name",
                        ),
                        VariantConfiguration(
                            compile_options=VariantCompilationOptions(
                                extra_defines={"N2": "12", "TSTEPS": "11"},
                                human_readable_output=False,
                                disable_checking=False,
                            ),
                            run_options=[
                                SubvariantRunOptions(subvariant_name=None, threads=1),
                                SubvariantRunOptions(subvariant_name=None, threads=2),
                            ],
                            variant_name="second " "variant " "name",
                        ),
                        VariantConfiguration(
                            compile_options=VariantCompilationOptions(
                                extra_defines={"N2": "12", "TSTEPS": "12"},
                                human_readable_output=False,
                                disable_checking=False,
                            ),
                            run_options=[
                                SubvariantRunOptions(subvariant_name=None, threads=1),
                                SubvariantRunOptions(subvariant_name=None, threads=2),
                            ],
                            variant_name="second " "variant " "name",
                        ),
                        VariantConfiguration(
                            compile_options=VariantCompilationOptions(
                                extra_defines={"N2": "12", "TSTEPS": "13"},
                                human_readable_output=False,
                                disable_checking=False,
                            ),
                            run_options=[
                                SubvariantRunOptions(subvariant_name=None, threads=1),
                                SubvariantRunOptions(subvariant_name=None, threads=2),
                            ],
                            variant_name="second " "variant " "name",
                        ),
                        VariantConfiguration(
                            compile_options=VariantCompilationOptions(
                                extra_defines={"N2": "12", "TSTEPS": "14"},
                                human_readable_output=False,
                                disable_checking=False,
                            ),
                            run_options=[
                                SubvariantRunOptions(subvariant_name=None, threads=1),
                                SubvariantRunOptions(subvariant_name=None, threads=2),
                            ],
                            variant_name="second " "variant " "name",
                        ),
                        VariantConfiguration(
                            compile_options=VariantCompilationOptions(
                                extra_defines={"N2": "13", "TSTEPS": "10"},
                                human_readable_output=False,
                                disable_checking=False,
                            ),
                            run_options=[
                                SubvariantRunOptions(subvariant_name=None, threads=1),
                                SubvariantRunOptions(subvariant_name=None, threads=2),
                            ],
                            variant_name="second " "variant " "name",
                        ),
                        VariantConfiguration(
                            compile_options=VariantCompilationOptions(
                                extra_defines={"N2": "13", "TSTEPS": "11"},
                                human_readable_output=False,
                                disable_checking=False,
                            ),
                            run_options=[
                                SubvariantRunOptions(subvariant_name=None, threads=1),
                                SubvariantRunOptions(subvariant_name=None, threads=2),
                            ],
                            variant_name="second " "variant " "name",
                        ),
                        VariantConfiguration(
                            compile_options=VariantCompilationOptions(
                                extra_defines={"N2": "13", "TSTEPS": "12"},
                                human_readable_output=False,
                                disable_checking=False,
                            ),
                            run_options=[
                                SubvariantRunOptions(subvariant_name=None, threads=1),
                                SubvariantRunOptions(subvariant_name=None, threads=2),
                            ],
                            variant_name="second " "variant " "name",
                        ),
                        VariantConfiguration(
                            compile_options=VariantCompilationOptions(
                                extra_defines={"N2": "13", "TSTEPS": "13"},
                                human_readable_output=False,
                                disable_checking=False,
                            ),
                            run_options=[
                                SubvariantRunOptions(subvariant_name=None, threads=1),
                                SubvariantRunOptions(subvariant_name=None, threads=2),
                            ],
                            variant_name="second " "variant " "name",
                        ),
                        VariantConfiguration(
                            compile_options=VariantCompilationOptions(
                                extra_defines={"N2": "13", "TSTEPS": "14"},
                                human_readable_output=False,
                                disable_checking=False,
                            ),
                            run_options=[
                                SubvariantRunOptions(subvariant_name=None, threads=1),
                                SubvariantRunOptions(subvariant_name=None, threads=2),
                            ],
                            variant_name="second " "variant " "name",
                        ),
                        VariantConfiguration(
                            compile_options=VariantCompilationOptions(
                                extra_defines={"N2": "14", "TSTEPS": "10"},
                                human_readable_output=False,
                                disable_checking=False,
                            ),
                            run_options=[
                                SubvariantRunOptions(subvariant_name=None, threads=1),
                                SubvariantRunOptions(subvariant_name=None, threads=2),
                            ],
                            variant_name="second " "variant " "name",
                        ),
                        VariantConfiguration(
                            compile_options=VariantCompilationOptions(
                                extra_defines={"N2": "14", "TSTEPS": "11"},
                                human_readable_output=False,
                                disable_checking=False,
                            ),
                            run_options=[
                                SubvariantRunOptions(subvariant_name=None, threads=1),
                                SubvariantRunOptions(subvariant_name=None, threads=2),
                            ],
                            variant_name="second " "variant " "name",
                        ),
                        VariantConfiguration(
                            compile_options=VariantCompilationOptions(
                                extra_defines={"N2": "14", "TSTEPS": "12"},
                                human_readable_output=False,
                                disable_checking=False,
                            ),
                            run_options=[
                                SubvariantRunOptions(subvariant_name=None, threads=1),
                                SubvariantRunOptions(subvariant_name=None, threads=2),
                            ],
                            variant_name="second " "variant " "name",
                        ),
                        VariantConfiguration(
                            compile_options=VariantCompilationOptions(
                                extra_defines={"N2": "14", "TSTEPS": "13"},
                                human_readable_output=False,
                                disable_checking=False,
                            ),
                            run_options=[
                                SubvariantRunOptions(subvariant_name=None, threads=1),
                                SubvariantRunOptions(subvariant_name=None, threads=2),
                            ],
                            variant_name="second " "variant " "name",
                        ),
                        VariantConfiguration(
                            compile_options=VariantCompilationOptions(
                                extra_defines={"N2": "14", "TSTEPS": "14"},
                                human_readable_output=False,
                                disable_checking=False,
                            ),
                            run_options=[
                                SubvariantRunOptions(subvariant_name=None, threads=1),
                                SubvariantRunOptions(subvariant_name=None, threads=2),
                            ],
                            variant_name="second " "variant " "name",
                        ),
                    ]
                },
                "gemver": {},
            },
            keep_going=True,
            min_runs=10,
            check_results_between_runs=True,
            save_raw_outputs=False,
            save_parsed_output_data=False,
        )
    )
    br.main_run()
