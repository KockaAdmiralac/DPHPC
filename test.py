#!/usr/bin/env python3
from argparse import ArgumentParser, Action
from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
from typing import Literal, Optional

import numpy as np
from tabulate import tabulate


Mode = Literal['serial', 'openmp', 'mpi', 'cuda']


class SplitArgs(Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if isinstance(values, str):
            setattr(namespace, self.dest,
                    [int(value) for value in values.split(',')])


@dataclass
class Binary:
    path: Path
    benchmark: str
    variant: int
    mode: Mode


@dataclass
class Result:
    variant: int
    mode: Mode
    threads: int
    mean: float
    std: float
    data: str


def get_script_dir() -> Path:
    return Path(os.path.realpath(__file__)).parent


def compile(benchmark: str, variant: int, cached_bins: bool) -> Binary:
    script_dir = get_script_dir()
    os.makedirs(script_dir / 'bin', exist_ok=True)
    bin_path = script_dir / 'bin' / f'{benchmark}_{variant}'
    benchmark_dir = script_dir / benchmark
    variant_path = benchmark_dir / f'v{variant}.c'
    if not variant_path.exists():
        variant_path = benchmark_dir / f'v{variant}.cu'
    with open(variant_path, 'r') as variant_file:
        variant_code = variant_file.read()
        if '#pragma omp' in variant_code:
            mode = 'openmp'
        elif '__global__' in variant_code:
            mode = 'cuda'
        elif 'MPI_Init' in variant_code:
            mode = 'mpi'
        else:
            mode = 'serial'
    if cached_bins and bin_path.exists():
        return Binary(bin_path, benchmark, variant, mode)
    args = [
        {
            'serial': 'gcc',
            'openmp': 'gcc',
            'mpi': 'mpicc',
            'cuda': 'nvcc',
        }[mode],
        '-Wall',
        '-Wextra',
        # '-fsanitize=address',
        # '-fsanitize=undefined',
        # '-fsanitize=leak',
        '-O3',
        '-o', str(bin_path),
        '-I', str(script_dir),
        '-I', str(benchmark_dir),
        '-DPOLYBENCH_TIME',
        '-DPOLYBENCH_DUMP_ARRAYS',
        str(script_dir / 'polybench.c'),
        str(script_dir / ''),
        str(variant_path)
    ]
    if mode == 'openmp':
        args.append('-fopenmp')
    if mode != 'cuda':
        args.append('-ffast-math')
        args.append('-march=native')
    subprocess.check_call(args)
    return Binary(bin_path, benchmark, variant, mode)


def run(
    binary: Binary,
    threads: int,
    runs: int,
    cached_results: bool,
    ground_truth: Optional[Result] = None
) -> Result:
    script_dir = get_script_dir()
    results_dir = script_dir / 'results'
    os.makedirs(results_dir, exist_ok=True)
    if binary.mode == 'mpi':
        args = ['mpiexec', '-n', str(threads), str(binary.path)]
        env = {}
    elif binary.mode == 'openmp':
        args = [str(binary.path)]
        env = {
            'OMP_NUM_THREADS': str(threads)
        }
    else:
        args = [str(binary.path)]
        env = {}

    results = []
    output = ''
    for _ in range(runs):
        if cached_results:
            results.append(0)
            continue
        process = subprocess.Popen(args, env=env)
        exit_code = process.wait()
        if exit_code != 0:
            raise ValueError(f'Exit code {exit_code} for {binary.path}')
        if process.stdout is None:
            raise ValueError('No process timing output found')
        if process.stderr is None:
            raise ValueError('No process data output found')
        results.append(float(process.stdout.read().decode('utf-8')))
        output = process.stderr.read().decode('utf-8')
        if ground_truth is not None and ground_truth.data != output:
            raise ValueError(f'Discrepancy between results - {binary.path}')
    return Result(binary.variant, binary.mode, threads,
                  float(np.mean(results)), float(np.std(results)), output)


if __name__ == '__main__':
    parser = ArgumentParser(
        description='Testing infrastructure for the DPHPC project, HS2024')
    parser.add_argument('--benchmark', type=str, required=True,
                        help='The benchmark to run')
    parser.add_argument('--variants', action=SplitArgs, required=True,
                        help='The variants to run')
    parser.add_argument('--threads', action=SplitArgs, default=[1, 2, 4, 8],
                        help='Numbers of threads to run with (tries each)')
    parser.add_argument('--runs', type=int, default=10,
                        help='The number of runs to perform')
    parser.add_argument('--cached_bins', action='store_true', default=False,
                        help='Do not compile binaries if compiled previously')
    parser.add_argument('--cached_results', action='store_true', default=False,
                        help='Do not run benchmarks if ran previously')

    args = parser.parse_args()
    variants = [variant for variant in args.variants if variant != 0]
    if len(variants) == 0:
        raise ValueError('Variant 0 is the baseline')

    binaries = [
        compile(args.benchmark, variant, args.cached_bins)
        for variant in [0] + args.variants
    ]

    ground_truth = run(binaries.pop(0), 1, args.runs, args.cached_results)
    results = [
        run(binary, thread_num, args.runs, args.cached_results, ground_truth)
        for thread_num in args.threads
        for binary in binaries
    ]

    print(tabulate([
        (result.variant, result.mode, result.threads, result.mean, result.std)
        for result in [ground_truth] + results
    ], headers=['Variant', 'Mode', 'Threads', 'Runtime (mean)',
                'Runtime (std)']))
