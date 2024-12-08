from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Literal, Optional

import marshmallow
import numpy as np
from datacheck import ParsedOutputData

valid_benchmark = Literal["adi", "gemver"]
ParallelisationScheme = Literal["serial", "openmp", "mpi", "cuda"]

DefinesConstraints = list[dict[str, int | str]]


class PathField(marshmallow.fields.Field):
    def _serialize(self, value, attr, obj, **kwargs):
        return str(value)

    def _deserialize(self, value, attr, data, **kwargs):
        return Path(value)


@dataclass
class Unionable:
    def __or__(self, other):
        return self.__class__(**self.__dict__ | other.__dict__)


@dataclass
class VariantCompilationOptions:
    extra_defines: dict[str, str] = field(default_factory=lambda: {})
    human_readable_output: bool = False
    disable_checking: bool = False


@dataclass
class SubvariantRunOptions:
    subvariant_name: Optional[str] = None
    threads: int = 1


@dataclass
class VariantConfiguration:
    compile_options: VariantCompilationOptions
    run_options: List[SubvariantRunOptions] = field(
        default_factory=lambda: [],
        metadata={"validate": marshmallow.validate.Length(min=1)},
    )
    variant_name: Optional[str] = None


@dataclass
class Options:
    defines: dict[str, str] = field(default_factory=lambda: {})
    extra_compile_options: List[str] = field(default_factory=lambda: [])
    data_check: Literal["strict", "fuzzy"] = "strict"
    max_deviation: Optional[float] = None
    defines_constraints: DefinesConstraints = field(default_factory=lambda: [])
    extra_source_dirs: List[str] = field(default_factory=lambda: [])
    extra_includes: List[str] = field(default_factory=lambda: [])
    exclude_sources: List[str] = field(default_factory=lambda: [])
    variant_configurations: List[VariantConfiguration] = field(
        default_factory=lambda: [
            VariantConfiguration(
                VariantCompilationOptions(),
                [SubvariantRunOptions("default")],
                variant_name="default",
            )
        ]
    )  # this one is special: only after reading this from Options does it get converted to VariantConfiguration

    def __getitem__(self, item: str) -> Any:
        return getattr(self, item)

    def __setitem__(self, item: str, value: Any) -> None:
        setattr(self, item, value)


@dataclass
class CompilationSettings:
    scheme: ParallelisationScheme
    all_defines: dict[str, str]
    defines_in_bin_path: dict[str, str]
    compile_raw_args: List[str]
    dphpc_opts: Options
    orig_options: VariantCompilationOptions
    binary_path: Path = field(metadata=dict(marshmallow_field=PathField()))
    include_dirs: List[Path] = field(
        metadata=dict(marshmallow_field=marshmallow.fields.List(PathField()))
    )
    source_files: List[Path] = field(
        metadata=dict(marshmallow_field=marshmallow.fields.List(PathField()))
    )


@dataclass
class BenchmarkConfiguration:
    benchmarks: dict[
        valid_benchmark,
        dict[str, List[VariantConfiguration]],
    ]  # benchmark name ->[variant1->[subvariant1,subvariant2]]
    keep_going: bool  # whether or not to continue after must completes
    min_runs: int
    check_results_between_runs: bool
    save_raw_outputs: bool
    save_parsed_output_data: bool
    save_deviations: bool
    generated_by: Optional[str] = None


@dataclass
class SingleBenchmark:
    variant_config: VariantConfiguration
    run_options: SubvariantRunOptions
    compile_settings: CompilationSettings
    ground_truth_bin_path: Path = field(metadata=dict(marshmallow_field=PathField()))


@dataclass
class PreparationResult:
    benchmark_choices: List[SingleBenchmark]
    compilations: dict[str, CompilationSettings]
    must_completes: List[SingleBenchmark]  # should be indices of benchmark_choices
    ground_truth_results: dict[Path, ParsedOutputData]
    keep_going: bool  # whether or not to continue after must completes
    check_results_between_runs: bool
    save_raw_outputs: bool
    save_parsed_output_data: bool
    save_deviations: bool


@dataclass
class RawResult:
    raw_stdout: bytes
    raw_stderr: bytes
    exit_code: int


@dataclass
class ProcessedResult:
    referenced_run: SingleBenchmark
    raw_result: Optional[RawResult] = None
    output_data: Optional[ParsedOutputData] = None
    timings: dict[str, float] = field(default_factory=lambda: {})
    data_checked: bool = False
    data_valid: Optional[bool] = None
    deviations: List[float] = field(default_factory=lambda: [])
