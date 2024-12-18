from copy import deepcopy
import itertools
import os
from pathlib import Path
import subprocess
from typing import get_args
import options
import preparation
from structures import *


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


def get_abs_paths(paths):
    return map(lambda p: Path(p).resolve(), paths)


def load_options(benchmark: str, variant: str) -> options.Options:
    if not get_benchmark_dir(benchmark).exists():
        raise FileNotFoundError("Couldn't open benchmark directory")
    if not get_variant_dir(benchmark, variant).exists():
        raise FileNotFoundError("Couldn't open variant directory")
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


def serialise_defines(defines: dict[str, str]) -> str:
    return "".join(map(lambda it: it[0] + it[1], defines.items()))


def prepare_single_compilation(
    benchmark: str,
    variant: str,
    user_options: VariantCompilationOptions,
) -> CompilationSettings:
    script_dir = get_script_dir()

    pb_generic_compunits = (script_dir / "polybench.c",)
    benchmark_dir = get_benchmark_dir(benchmark)
    assert benchmark_dir.exists()

    scheme = get_scheme_from_variant_name(variant)

    variant_dir = get_variant_dir(benchmark, variant)
    if not variant_dir.exists():
        raise AssertionError(f"{variant_dir} doesn't exist")

    dphpc_opts = load_options(benchmark, variant)

    merged_defines = deepcopy(dphpc_opts.defines)
    merged_defines.update(
        user_options.extra_defines
    )  # manually provided defines via CLI override ones in JSON

    if user_options.human_readable_output:
        merged_defines["DUMP_DATA_HUMAN_READABLE"] = ""

    if user_options.disable_checking:
        merged_defines["DISABLE_CHECKING"] = ""

    check_defines_constraints(dphpc_opts.defines_constraints, merged_defines)

    os.makedirs(script_dir / "bin", exist_ok=True)
    bin_path = get_bin_path(benchmark, variant, user_options, script_dir)

    def compunit_filter(files):
        return filter(lambda fp: fp.suffix in (".c", ".cu", ".cpp"), files)

    source_dirs = itertools.chain(
        [benchmark_dir, variant_dir], get_abs_paths(dphpc_opts.extra_source_dirs)
    )
    raw_source_files = itertools.chain(*map(lambda d: d.iterdir(), source_dirs))
    compunits_unfiltered = itertools.chain(
        pb_generic_compunits, compunit_filter(raw_source_files)
    )

    exclude_sources_abs = list(get_abs_paths(dphpc_opts.exclude_sources))
    compunits = list(
        filter(lambda src: src not in exclude_sources_abs, compunits_unfiltered)
    )

    includes = list(
        itertools.chain(
            [variant_dir, script_dir, benchmark_dir],
            get_abs_paths(dphpc_opts.extra_includes),
        )
    )
    includes_args = intersperse_elem_in_list(includes, "-I")

    defines_args = map(
        lambda it: f"-D{it[0]}={it[1]}".rstrip("="), merged_defines.items()
    )  # rstrip makes sure defines without values don't have =
    extra_options = (
        dphpc_opts.extra_compile_options
        if dphpc_opts.extra_compile_options is not None
        else []
    )
    extra_options_post = (
        dphpc_opts.extra_compile_options_post
        if dphpc_opts.extra_compile_options_post is not None
        else []
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
        *map(str, includes_args),
        *map(str, defines_args),
        *map(str, extra_options),
        *map(str, compunits),
        *map(str, extra_options_post),
    ]
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

    # print(args)

    return CompilationSettings(
        scheme,
        merged_defines,
        user_options.extra_defines,
        args,
        dphpc_opts,
        user_options,
        bin_path,
        includes,
        source_files=compunits,
    )


def get_scheme_from_variant_name(variant):
    scheme = variant.split("_")[0]
    assert scheme in get_args(ParallelisationScheme)
    return scheme


def get_bin_path(
    benchmark: str,
    variant: str,
    user_options: VariantCompilationOptions,
    script_dir: Path,
):
    return (
        script_dir
        / "bin"
        / f"{benchmark}_{variant}_{serialise_defines(user_options.extra_defines)}{'_NOPRINT' if user_options.disable_checking else ''}"
    )


def compile(compsettings: CompilationSettings):
    bin_path = compsettings.binary_path
    should_recompile = not bin_path.exists()
    if not should_recompile:
        bin_change_time = bin_path.stat().st_mtime
        for dep in compsettings.source_files:
            if dep.stat().st_mtime >= bin_change_time:
                should_recompile = True
                break
    if should_recompile:
        print("Recompiling " + str(bin_path))
        print(" ".join(compsettings.compile_raw_args))
        subprocess.check_call(compsettings.compile_raw_args)
    else:
        # print("Not recompiling " + str(bin_path))
        pass  # Change me back if your terminal is big enough
