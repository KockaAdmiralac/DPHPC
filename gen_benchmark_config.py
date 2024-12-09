from argparse import Action, ArgumentParser, Namespace
import argparse
import copy
import json
import pprint
import sys

import marshmallow_dataclass
from structures import *
import compat


def get_template_benchmark_config(args: Namespace) -> BenchmarkConfiguration:
    template_bc = BenchmarkConfiguration(
        {},
        args.keep_going,
        args.min_runs,
        args.check_results_between_runs,
        args.save_raw_outputs,
        args.save_parsed_output_data,
        args.save_deviations,
    )
    variant_configuration_schema = marshmallow_dataclass.class_schema(
        VariantConfiguration
    )()
    for benchmark_arg in args.benchmark:
        benchmark = benchmark_arg[0]
        if benchmark not in template_bc.benchmarks:
            template_bc.benchmarks[benchmark] = {}
        for variant in benchmark_arg[1:]:
            opt = compat.load_options(benchmark, variant)
            template_bc.benchmarks[benchmark][variant] = opt.variant_configurations
    return template_bc


def bc_inject_args(
    args: Namespace, template_benchmark_config: BenchmarkConfiguration
) -> BenchmarkConfiguration:
    bc = copy.deepcopy(template_benchmark_config)
    # handles most of the arguments from original test.py, notably things like threads, human readable output, disable checking
    for benchmark, variants in bc.benchmarks.items():
        for variant, variant_configs in variants.items():
            for var_conf in variant_configs:
                if args.disable_checking is not None:
                    var_conf.compile_options.disable_checking = args.disable_checking
                if args.human_readable_output is not None:
                    var_conf.compile_options.human_readable_output = (
                        args.human_readable_output
                    )
                new_run_options = []
                for run_opt in var_conf.run_options:
                    if args.threads is None:
                        new_run_options.append(run_opt)
                    else:
                        if compat.get_scheme_from_variant_name(variant) in [
                            "serial",
                            "cuda",
                        ]:
                            thread_replicas = [
                                1,
                            ]
                        else:
                            thread_replicas = args.threads
                        for thread_inst in thread_replicas:
                            local_run = copy.deepcopy(run_opt)
                            local_run.threads = thread_inst
                            new_run_options.append(local_run)
                var_conf.run_options = new_run_options

    # this pass is about updating extra_defines according to the extra-defines argument
    for benchmark, variants in bc.benchmarks.items():
        for variant, variant_configs in variants.items():
            for var_conf in variant_configs:
                for extra_defines_updates in args.extra_defines:
                    if extra_defines_updates["benchmark"] in (
                        benchmark,
                        "*",
                    ) and extra_defines_updates["variant"] in (variant, "*"):
                        new_values = extra_defines_updates["values"]
                        new_defines = dict([x.split("=")[:2] for x in new_values])
                        action = extra_defines_updates["action"]
                        if action == "add":
                            var_conf.compile_options.extra_defines.update(new_defines)
                        elif action == "replace":
                            var_conf.compile_options.extra_defines = new_defines
    # this part replicates compilation options multiple times to account for sweeps
    for benchmark, variants in bc.benchmarks.items():
        for variant, variant_configs in variants.items():
            prev_configs = variant_configs
            for sweep in args.sweep_define:
                new_variant_configs = []
                for var_conf in prev_configs:
                    if (
                        sweep["benchmark"] in (benchmark, "*")
                        and sweep["variant"] in (variant, "*")
                        and sweep["compile_variant_name"]
                        in (var_conf.variant_name, "*")
                    ):
                        if "values" in sweep:
                            vals = sweep["values"]
                        else:
                            vals = map(str, range(sweep["min"], sweep["max"] + 1))
                        for val in vals:
                            new_var_conf = copy.deepcopy(var_conf)
                            new_var_conf.compile_options.extra_defines[
                                sweep["name"]
                            ] = str(val)
                            new_variant_configs.append(new_var_conf)
                    else:
                        new_variant_configs.append(var_conf)
                prev_configs = new_variant_configs
                # prev_configs and new_variant_configs now point to same new configs
            variants[variant] = prev_configs
    return bc


class SplitIntArgs(Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if isinstance(values, str):
            setattr(namespace, self.dest, [int(value) for value in values.split(",")])


class OptionsSetter(argparse._AppendAction):
    def __call__(self, parser, namespace, raw_val, option_string=None):
        values = raw_val.split(",")
        # values should look like benchmark variant action item1 item2
        if len(values) < 4:
            raise argparse.ArgumentError(
                self,
                "Less than three parameters were provided for the option setter action",
            )
        if values[2] not in ("add", "replace"):
            raise argparse.ArgumentError(
                self, "The option action was none of add/replace/delete"
            )
        vals_to_write = values[3:]
        items = getattr(namespace, self.dest, None)
        items = argparse._copy_items(items)  # type: ignore
        items.append(
            {
                "benchmark": values[0],
                "variant": values[1],
                "values": vals_to_write,
                "action": values[2],
            }
        )
        setattr(namespace, self.dest, items)


class SweepProcessor(argparse._AppendAction):
    def __call__(self, parser, namespace, raw_str, option_string=None):
        # values should look like benchmark variant compilevariantname name min max
        values = raw_str.split(",")
        if len(values) != 6:
            raise argparse.ArgumentError(
                self,
                "Must pass all arguments",
            )
        if not values[4].isdigit() or not values[5].isdigit():
            raise argparse.ArgumentError(self, "min and max must be integers")
        min_int = int(values[4])
        max_int = int(values[5])
        if min_int > max_int:
            raise argparse.ArgumentError(
                self, "Can't have min value be higher than max"
            )

        items = getattr(namespace, self.dest, None)
        items = argparse._copy_items(items)  # type: ignore
        items.append(
            {
                "benchmark": values[0],
                "variant": values[1],
                "compile_variant_name": values[2],
                "name": values[3],
                "min": min_int,
                "max": max_int,
            }
        )
        setattr(namespace, self.dest, items)


class SweepProcessor2(argparse._AppendAction):
    def __call__(self, parser, namespace, raw_str, option_string=None):
        # values should look like benchmark variant compilevariantname name min max
        values = raw_str.split(",")
        if len(values) < 5:
            raise argparse.ArgumentError(
                self,
                "Must pass all arguments, at least one value",
            )

        items = getattr(namespace, self.dest, None)
        items = argparse._copy_items(items)  # type: ignore
        items.append(
            {
                "benchmark": values[0],
                "variant": values[1],
                "compile_variant_name": values[2],
                "name": values[3],
                "values": values[4:],
            }
        )
        setattr(namespace, self.dest, items)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Generator of BenchmarkConfiguration from dphpc_md.json files and old-style parameters"
    )

    parser.add_argument(
        "--benchmark",
        nargs="+",
        action="append",
        type=str,
        help="A benchmark to run, where the first item is the benchmark name and the remaining ones are variants of that benchmark.",
    )

    parser.add_argument(
        "--threads",
        action=SplitIntArgs,
        # default=[1],
        help="Numbers of threads to run with (note this is forced to 1 for serial+CUDA)",
    )

    parser.add_argument(
        "--min-runs", type=int, default=5, help="The minimum number of runs to perform"
    )

    parser.add_argument(
        "--keep-going",
        action="store_true",
        default=False,
        help="Keep running once the minimum runs have been completed, until a keyboard interrupt arrives",
    )

    parser.add_argument(
        "--disable-checking",
        action="store_true",
        help="Disable all ouput data checking",
    )

    parser.add_argument(
        "--human-readable-output",
        action="store_true",
        help="Dump data output in human-readable format",
    )

    parser.add_argument(
        "--extra-defines",
        help="""Replace/add extra defines for one or more benchmarks/variants.  Specify as "benchmark,variant,action,item1=val1,item2=", where benchmark and variant can be * for wildcard, action is add/replace and the items are to go in the dict (=-separated)""",
        default=[],
        action=OptionsSetter,
    )

    parser.add_argument(
        "--sweep-define",
        help="""Sweep a preprocessor definition.  Specify as "benchmark,variant,compilevariantname,name,min,max", where benchmark and variant can be * for wildcard, compilevariantname is the name of a particular compilation or *, name is the preprocessor definition, min/max are the range to sweep over""",
        default=[],
        action=SweepProcessor,
    )

    parser.add_argument(
        "--multiple-define",
        help="""Sweep a preprocessor definition across a provided set of values, mostly inteded for N2/TSTEPS.  Specify as "benchmark,variant,compilevariantname,name,val1,val2,...", where benchmark and variant can be * for wildcard, compilevariantname is the name of a particular compilation or *, name is the preprocessor definition, values are what to sweep over""",
        default=[],
        dest="sweep_define",
        action=SweepProcessor2,
    )

    parser.add_argument(
        "--no-check-results-between-runs",
        help="Normally the script checks a run's result right after, pass this to check after all runs instead",
        dest="check_results_between_runs",
        action="store_false",
    )  # at the moment checking after all runs finished is not implemented, this option only disables checking between runs

    parser.add_argument(
        "--save-raw-outputs",
        action="store_true",
        help="Save raw stdout and stderr beyond the scope of one run.  Mostly used if you want to print everything at the end.",
    )

    parser.add_argument(
        "--save-parsed-output-data",
        action="store_true",
        help="Save parsed output data beyond the scope of one run.  Necessary if you want to do all checking after all runs have completed.",
    )

    parser.add_argument(
        "--save-deviations",
        action="store_true",
        help="Save info about deviations for each output data value.",
    )

    parser.add_argument(
        "config_file",
        default=None,
        help="Where to write the configuration to, optional and if not specified it dumps to stdout.",
    )

    args = parser.parse_args()
    template_bc = get_template_benchmark_config(args)
    bc = bc_inject_args(args, template_bc)
    bc.generated_by = " ".join(sys.argv)
    if args.config_file is None:
        pprint.pprint(bc)
    else:
        benchmark_config_schema = marshmallow_dataclass.class_schema(
            BenchmarkConfiguration
        )()
        py_obj = benchmark_config_schema.dump(bc)
        with open(args.config_file, "w+") as f:
            json.dump(py_obj, f, indent=4)

# Example run: python3 gen_benchmark_config.py --benchmark adi mpi_1 --benchmark gemver --keep-going --min-runs 10 --threads 1,2 --sweep-define \* \* \* N2 10 15 --sweep-define adi \* \* TSTEPS 10 15 --no-check-results-between-runs
