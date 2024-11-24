from collections.abc import Callable, Iterable
from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
from typing import List

import test


@dataclass
class SingleBenchmark:
    scheme: test.ParallelisationScheme
    binary_path: Path


@dataclass
class RawResult:
    raw_stdout: bytes
    raw_stderr: bytes
    exit_code: int


def lowlevel_run(
    arg_maker: Callable[[SingleBenchmark, Path], List[str]],
    env_maker: Callable[[SingleBenchmark, dict[str, str]], dict[str, str]],
    b: SingleBenchmark,
) -> RawResult:
    external_env = dict(os.environ)
    args = arg_maker(b, b.binary_path)
    env = env_maker(b, external_env)
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
        raise ValueError(f"Exit code {exit_code} for {b.binary_path}")
    return RawResult(raw_stdout=stdout, raw_stderr=stderr, exit_code=exit_code)


def run_benchmark(b: SingleBenchmark) -> test.Result:
    pass
