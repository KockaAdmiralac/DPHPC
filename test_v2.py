from dataclasses import dataclass
import random
from typing import Iterable, List
import runner
import single_benchmark
import test


@dataclass
class BenchmarkConfiguration:
    pass


class BenchmarkRunner:
    r: runner.Runner

    must_run_benchmarks: List[single_benchmark.SingleBenchmark]
    benchmark_choices: List[single_benchmark.SingleBenchmark]

    benchmark_config: BenchmarkConfiguration

    def select_next_benchmark(self) -> single_benchmark.SingleBenchmark:
        if len(self.must_run_benchmarks):
            idx = random.randint(0, len(self.must_run_benchmarks) - 1)
            item = self.must_run_benchmarks[idx]
            del self.must_run_benchmarks[idx]
            return item

        return random.choice(self.benchmark_choices)

    def run_single_benchmark(b: single_benchmark.SingleBenchmark) -> test.Result:

        pass

    def main_run_loop(r: runner.Runner):
        results: List[test.Result] = []
        while True:
            try:
                pass
            except KeyboardInterrupt:
                break

    def main_run():
        r = runner.Runner()
