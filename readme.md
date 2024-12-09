# PolyBench parallelization
This project aims to parallelize [PolyBench](https://github.com/MatthiasJReisinger/PolyBenchC-4.2.1) using OpenMP, MPI and CUDA, as a part of the [Design of Parallel and High-Performance Computing](https://spcl.inf.ethz.ch/Teaching/2024-dphpc/) at [ETH Zürich](https://ethz.ch/), in Winter Semester 2024. Students working on this project are:

- [George Manos](https://github.com/WatchmeDoc)
- [Giorgos Drosos](https://gdrosos.github.io/)
- [Maritina Tsanta](https://github.com/mctsnt)
- [Paolo Celati](https://github.com/pcelati)
- [Luka Simić](https://kocka.tech/)

## Benchmark
You can benchmark your code by using the `test_v2.py` script. It can compile your code with the appropriate options, run it with the appropriate runtime, check whether its output is correct and compare it with other implementations.

To run a benchmark campaign, test_v2 is split into `gen_benchmark_config.py`, `test_v2.py` and `output.py`.  You must first generate a benchmark configuration for `test_v2.py` to tell it what to run exactly.  This is done with `gen_benchmark_config.py`, which accepts most of the arguments for the original `test.py` but with a few changes.  It will save the benchmark configuration to a JSON file you specify.  Afterwards, run `test_v2.py`, passing it the file with the configuration.  If you give `test_v2.py` an output filepath it will save the results there.  If you provide the `--output` option and specify a valid `output.py` output format (currently graph/table), the results will also be output directly.  If you want to process multiple runs' results together you can run `output.py` and pass it `--from` one or more times to have it output multiple files' results.

### Generating a Benchmark Configuration
To generate the benchmark configuration file you run `gen_benchmark_config.py`.  The program can make a single large configuration file for multiple benchmarks, multiple variants, multiple compilation configurations per variant (for sweeps of compile time preprocessor definitions for instance like N2 or block size) and multiple run configurations per compilation configuration (notably for running the same MPI/OpenMP binary with varying numbers of threads).  These are all represented in the `BenchmarkConfiguration` dataclass, which gets serialised and dumped to a JSON file.

The configuration file is very verbose because it means `test_v2.py` needs to account for far less business logic, and you can make changes to the configuration file by hand if needed before running `test_v2.py` itself.  The latter can be useful especially if you have a small change where the configuration generator's interface is too coarse.

The generator extends dphpc_md a little.  You can provide a `variant_configurations` key in `dphpc_md.json` to provide one or more variant configurations.  A variant configuration indicates how to compile the variant (with the `compile_options` key) and has one or more `run_options` to specify different ways to run the binary.  The `variant_name` in the variant configuration and `subvariant_name` in the run configuration are supposed to be display names for things like nicer graphs.  The configuration generator goes through `variant_configurations` and the child data to get defaults for each benchmark/variant.  `variant_configurations` is meant to be used to provide different configurations for the same variant without having to duplicate the code a bunch of times.  The toplevel `dphpc_md.json` already implements a basic default for running everything so the `dphpc_md.json` options don't need to be provided in the default case.

The configuration generator supports:
- `--benchmark <benchmark name> <variant 1> <variant 2> ...`: Which benchmark to run, and what variants of that benchmark.

  Examples:
  - `--benchmark adi serial_opt serial_block openmp_base --benchmark gemver mpi_cols cuda_improved6`
  - `--benchmark gemver cuda_cublas serial_block_first_loop`
- `--threads`: How many threads to test with for OpenMP/MPI, as a comma-separated list.
- `--min-runs`: How many runs `test_v2.py` should run.  If `--keep-going` is provided, this specifies the minimum runs to perform for each configuration.  Every configuration will be run this many times before switching to random resampling.  The default is 5 run.  Note: when the confidence interval needs to be calculated there need to be at least two runs so the program will automatically perform a second run in that case even if `--min-runs` is 1.
- `--keep-going`: Useful when you have a good amount of time but are unsure how long you can allocate, specify this if you'd like `test_v2.py` to randomly resample configurations to get more trials for each.  Interrupt the benchmark run with Ctrl+C, wait for the program to finish running the current run and it will run the rest of the program like saving to a results file.  If this is not specified, `test_v2.py` will only run exactly as many runs as given by `--min-runs`.
  
  Example: `--keep-going --min-runs 5` will run five of each configuration, then will print the "must completes" are done and will switch to randomly sampling among the possible configurations and running those.  It will keep running until it gets Ctrl+C.
- `--disable-checking`: Does what it says, provides `DISABLE_CHECKING` to everyone and does not check or save results.
- `--human-readable-output`: Makes all configurations output data in human readable form.
- `--extra-defines <benchmark>,<variant>,<action>,<name1>=<val1>,<name2>=,<name3>=<val3>,...`: Depending on `<action>` being `add` or `replace`, determines whether to add more preprocessor definitions or change existing preprocessor definitions.  Note this is processed before `--sweep-define`, so avoid specifying the same preprocessor definitions here and there.  `<benchmark>,<variant>` are used to filter which benchmarks and variants this should apply to.  Specify `*` if you want wildcard.

  Example: `--extra-defines adi,*,add,BLOCK_SIZE=64` adds BLOCK_SIZE=64 to all adi variants
- `--sweep-define <benchmark>,<variant>,<compilevariantname>,<name>,<min>,<max>`: Generates multiple compilation configurations, with the `<name>` preprocessor definition for each min<=x<=max (both inclusive).  `<benchmark>,<variant>,<compilevariantname>` are used to filter which benchmarks, variants and compilation configurations this should apply to.  Specify `*` if you want a wildcard.  Passing this optio multiple times will effectively iterate over the product of the passed options.

  Example: `--sweep-define gemver,cuda_cublas,*,JS_IN_SM,1,8 --sweep-define *,*,N2,6,10` will make 5 versions of each configuration with N2 between 6 and 10, except gemver/cuda_cublas which will have 5x8=40 versions, iterating N2 between 6 and 10 and JS_IN_SM between 1 and 8
- `--multiple-define <benchmark>,<variant>,<compilervariantname>,<name>,<val1>,...`: Sweeps very much like `--sweep-define` but you provide the individual values you'd like.  Use this to specify N2 especially, possibly CUDA threads per block.

  Example: `--multiple-define *,*,*,N2,70,80,100,150,200`
- `--no-check-results-between-runs`: Not recommended.  Intended for being able to check results only after all runs' results have been collected, but this isn't implemented.  At the moment, checking is performed between runs instead.
- `--save-raw-outputs`: Whether to store the raw stderr/stdout/exit code after each run.  By default this is not done because it needs lots of RAM.  A non-zero exit code will store the raw stderr/stdout/exit code even if this option is not provided.
- `--save-parsed-output-data`: Whether to store the parsed data at the end of a run or not.  Normally this doesn't happen because it needs lots of RAM.
- `--save-deviations`: Saves deviation data after a run, possibly useful data if we were to implement variants using lower precision floating point formats like fp32.  Normally this is not done because it needs lots of RAM, exactly as much as the data arrays themselves.
- `--ground-truths-dir`: Where to save all results.  If not passed, ground truths will be kept in RAM.
-- `<config file>`: Where to dump the configuration to.  If not provided, the configuration is printed to the stdout.

This is how an invocation of `gen_benchmark_config.py` might look like:
```console
python3 gen_benchmark_config.py \
    --benchmark gemver serial_base openmp_tasks mpi_cache_efficiency \
    --threads 1,2,4,8 \
    --min-runs 100 \
    --disable_checking \
    --extra-defines *,*,add,N2=12000
```

Examples using `adi`:
```console
python3 gen_benchmark_config.py \
    --benchmark adi serial_base \
    --min-runs 10 \
    --save-parsed-output-data \
    --extra-defines *,*,replace,SMALL_DATASET=1
```

```console
python3 gen_benchmark_config.py \
    --benchmark adi serial_base mpi_1 openmp_block cuda_multithreaded5 \
    --min-runs 10 \
    --keep-going \
    --threads 1,2,4,8
    --disable-checking \
    --extra-defines *,*,replace,SMALL_DATASET=1
    --sweep-define *,*,*,TSTEPS,10,15
```

Running all useful, working benchmarks so far for a long run:
```console
python3 gen_benchmark_config.py \
    --keep-going \
    --threads 1,2,4,8,16 \
    --benchmark gemver cuda_cublas cuda_improved2 cuda_improved3 cuda_improved4 cuda_improved5 cuda_improved6 mpi_basic mpi_cols mpi_non_block openmp_basic openmp_basic_all openmp_block_second_loop openmp_merged serial_base serial_block_first_loop serial_block_second_loop serial_extracted_alpha serial_extracted_beta serial_merge_loops serial_neater_c serial_reorder_second_loop serial_vectorized_first_loop serial_vectorized_second_loop \
    --benchmark adi cuda_multithreaded4 cuda_multithreaded5 cuda_multithreaded6 cuda_multithreaded7 cuda_multithreaded8 cuda_multithreaded9 openmp_base openmp_base_opt openmp_block serial_base serial_block cuda_minimise_spare_arr_pop cuda_multithreaded12 \
    --multiple-define gemver,*,*,N2,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000,11000,12000,13000,14000,1024,1536,2048,3172,4096,6144,8192,10240,12288,14336,1023,1535,2047,3171,4095,6143,8191,10239,12287,14335,1025,1537,2049,3173,4097,6145,8193,10241,12289 \
    --multiple-define adi,*,*,N2,1000,2000,3000,4000,5000,6000,1024,1536,2048,3172,4096,6144,1023,1535,2047,3171,4095,6143,1025,1537,2049,3173,4097,6145 \
    --extra-defines adi,*,add,TSTEPS=15 \
    --min-runs 25 \
    --ground-truths-dir /storage/bulk2024b/dphpc_results \
    config2.json
```

You can read about the other options in `./gen_benchmark_config.py --help`.

### Running `test_v2.py`
The only things to pass `test_v2.py` itself are the benchmark configuration file (`--config`), where to save results (`--results-output`) and the output format (`--output`).  Only the configuration file option is mandatory.  The program will go through the benchmark configuration file and run everything, then dump the results at the end.

Example: `./test_v2.py --config config2.json --results-output results.json --output table`

### Postprocessing and outputting data
There's a third program called `output.py` that is useful when you have a good collection of results already and want to process the data and graph it.  You can pass `--output` to specify whether you want table or graph output (graph not yet implemented), and `--from` one or more times to indicate which files to load results from.

### Specialization
Your code will sometimes need special options that `test.py` does not set. For this purpose, `test.py` reads files named `dphpc_md.json` at root level, benchmark level and variant level to determine which options should it use to compile the code. For example, if your benchmark is `gemver` and you are creating a variant named `openmp_tasks`, then `test.py` will read the following files:

- `dphpc_md.json`
- `gemver/dphpc_md.json`
- `gemver/openmp_tasks/dphpc_md.json`

Deeper files will override options from parent directories, with command-line
arguments such as `--set_defines` overriding all of them.

These files are structured in the following way:

```json
{
    // Dictionary of additional definitions to pass to the compiler.
    "defines": {
        // #define VARIABLE value
        "VARIABLE": "value",
        // #define VARIABLE2 1234
        "VARIABLE2": 1234,
        // #define FLAG3
        "FLAG3": ""
    },
    // Array of additional options to pass to the compiler.
    "extra_compile_options": [
        // These two can be useful if you're looking for invalid memory accesses.
        "-fsanitize=address",
        "-fsanitize=undefined"
    ],
    // Can be either "strict" or "fuzzy" depending on if you want the results to
    // exactly match the baseline, or only to a certain deviation.
    // Default is strict. If you're getting minor differences in results, try
    // overriding this to fuzzy in the appropriate benchmark/variant.
    "data_check": "fuzzy",
    // How much can the obtained result deviate from the ground truth, relative
    // to the obtained result.
    // Default is 0.015, meaning that the obtained result can be up to 1.5%
    // different from the ground truth.
    "max_deviation": 0.015,
    // If you want to constrain what values can certain defines have.
    "defines_constraints": [
        {
            "define": "VARIABLE2",
            "max": 100000,
            "min": 1000
        }
    ]
}
```

### Rerunning faster
Aside the aforementioned `--disable_checking` option, which disables checking of whether the output data matches the ground truth in case you already know your code is correct, the benchmark provides `--cached_bins` to run the code without recompiling the binaries, as well as `--cached_results` to reuse latest timing results for the same parameters (benchmark, variant, number of threads, defines), if they exist.

### Binary and result formats
The `bin` directory contains previously compiled binaries for a given benchmark, variant and defines. For example, running the `serial_base` variant of the `gemver` benchmark with `N2=12000` generates a binary under `bin/gemver_serial_base_N212000`. This can be useful to know if you're running the binary externally with a program such as Intel vTune or a debugger.

The `results` directory contains subdirectories for each variant, which contain all timing results of previous runs that have been generated so far. They are stored as JSON files with a `timing` array that indicates how long each of the runs (the amount of which has been specified in `--runs`) have been running (allowing for analysis of mean time and standard deviation of these runs), as well as a `deviations` array which contains how much the output data deviates from the ground truth. Output data itself is not stored in these JSON files. Aside from timing results, it contains symbolic links to the latest obtained timing result, if a future script needs to only grab the latest timing result.

Together with timing results, `test.py` stores ground truth data in the benchmark directories. If a ground truth file is found, it reads the ground truth data from it instead of rerunning the serial case to obtain it. The option `--require_recompute_ground_truth` allows you to recompute this ground truth, if you want.

### Finding discrepancies
When `test.py` outputs a discrepancy between the output data and ground truth, it might not be very clear where the discrepancy is. For this reason, it outputs a file with the output data of that failed run into the directory with the variant's timing results, by default in a binary format which consists of 8 bytes that indicates array (or matrix) size, one character that indicates array (or matrix) name, followed by an array of doubles with the data, for each array being printed.

If you'd like to see the numbers in a humanly-readable format, you can pass the option `--human_readable_output`. This can be significantly slower, but helps with debugging on smaller datasets.

### Changing result paths
If you'd like to change where timing results and output data are stored, the following options are available:

- `--results_fstr`: filename to store the timing results in
- `--latest_results_fstr`: filename of the symlink to the latest timing results
- `--ground_truth_out_fstr`: filename to store the ground truth in
- `--failed_data_out_fstr`: filename to store the output data after a failed run in

These options are passed as Python format strings, with the following variables available to use in the format string:

- **Directory paths:**
    - `{script_dir}`: absolute path to the repository root
    - `{results_benchmark_dir}`: absolute path to the benchmark results directory
    - `{results_variant_dir}`: absolute path to the variant results directory (unavailable for ground truth)
    - `{benchmark_dir}`: absolute path to the benchmark code directory
    - `{variant_dir}`: absolute path to the variant code directory (unavailable for ground truth)
    - `{binary.path}`: absolute path to the program's binary (unavailable for ground truth)
- **Path components:**
    - `{ts}`: current Unix timestamp
    - `{iso8601}`: current time formatted as an ISO 8601 timestamp
    - `{benchmark}`: benchmark name
    - `{variant}`: variant name (unavailable for ground truth)
    - `{threads}`: amount of threads used
    - `{ser_defines}`: defines serialized into a string
    - `{binary.scheme}`: parallelization scheme used (unavailable for ground truth)

For example, if you passed an option such as `--results_fstr {script_dir}/results/{benchmark}_{variant}_{threads}_{ser_defines}.json`, you would end up with a file such as `results/gemver_serial_base_1_N212000.json`.

### Specifying GPU to use for CUDA variants
Set the environment variables like `CUDA_VISIBLE_DEVICES=<gpuidx> CUDA_DEVICE_ORDER=PCI_BUS_ID`.  To get the gpuidx for the desired GPU use `lspci -nn | grep -E "VGA.*NVIDIA"` and pick the right index.  Note this is not necessarily the same index as that in `nvidia-smi`.

## Running on Euler
To obtain final results of the project, we can use the [Euler](https://scicomp.ethz.ch/wiki/Euler) cluster provided by ETH. To access it as an ETH student, you can read the [wiki page on getting started with clusters](https://scicomp.ethz.ch/wiki/Getting_started_with_clusters). Once you get access to the Euler login node, you can use the `sbatch` command to run a Bash script on the cluster.

Before running `test.py`, we need to install Python on the relevant nodes. The following command installs it:

```
module load stack/2024-06 python/3.11.6
```

For example, you can save the shell script below as `run_gemver_serial_base.sh`:

```console
#!/bin/sh
python3 test.py \
    --benchmark gemver \
    --variants serial_base \
    --threads 1 \
    --runs 1000 \
    --set_defines N2=12000
```

Then, to run this script on the CPU hosts, you can use:

```
sbatch run_gemver_serial_base.sh
```

You can read more about [loadable application stacks](https://scicomp.ethz.ch/wiki/Applications) and [submitting batch jobs](https://scicomp.ethz.ch/wiki/Using_the_batch_system) on the [Scientific Computing Wiki](https://scicomp.ethz.ch/).

## Linting
This repository uses a [GitHub Action](.github/workflows/lint.yml) to check whether all files are linted adequately. If you're modifying C/CUDA/Python code, you can use `make format` to format the code consistently with other files, and `pyright .` to check for Python type errors. Alternatively, you can install adequate extensions for black, Pyright and clang-format in your IDE. You can also install an extension for `isort` to keep the imports ordered correctly, but this is not required to pass the linting.

Linting checks do not prevent you from merging your code, but in case you want to skip the GitHub Action anyways, you can use one of the [skipping words](https://docs.github.com/en/actions/managing-workflow-runs-and-deployments/managing-workflow-runs/skipping-workflow-runs) in your commit message or commit description, such as `[skip ci]`.

## MPI Profiling

To profile MPI code use [Caliper](http://software.llnl.gov/Caliper/index.html) -- it can also be used for OpenMP and Cuda it simply needs the correct flags when running cmake.

### Slurm setup:

First, set up `slurm`. To download run:
```
sudo apt update -y
sudo apt install slurmd slurmctld -y
```
Then we need to create the `slurm.conf` file which configures how the slurm queue is set up. Run `slurmd -C` to see your node resources and adjust the `Compute Nodes` region below accordingly.
```
$ sudo chmod 777 /etc/slurm
$ sudo cat << EOF > /etc/slurm/slurm.conf
# slurm.conf file generated by configurator.html.
# Put this file on all nodes of your cluster.
# See the slurm.conf man page for more information.
#
ClusterName=localcluster
SlurmctldHost=localhost
MpiDefault=none
ProctrackType=proctrack/linuxproc
ReturnToService=2
SlurmctldPidFile=/var/run/slurmctld.pid
SlurmctldPort=6817
SlurmdPidFile=/var/run/slurmd.pid
SlurmdPort=6818
SlurmdSpoolDir=/var/lib/slurm/slurmd
SlurmUser=slurm
StateSaveLocation=/var/lib/slurm/slurmctld
SwitchType=switch/none
TaskPlugin=task/none
#
# TIMERS
InactiveLimit=0
KillWait=30
MinJobAge=300
SlurmctldTimeout=120
SlurmdTimeout=300
Waittime=0
# SCHEDULING
SchedulerType=sched/backfill
SelectType=select/cons_tres
SelectTypeParameters=CR_Core
#
#AccountingStoragePort=
AccountingStorageType=accounting_storage/none
JobCompType=jobcomp/none
JobAcctGatherFrequency=30
JobAcctGatherType=jobacct_gather/none
SlurmctldDebug=info
SlurmctldLogFile=/var/log/slurm/slurmctld.log
SlurmdDebug=info
SlurmdLogFile=/var/log/slurm/slurmd.log
#
# COMPUTE NODES
NodeName=localhost CPUs=1 RealMemory=500 State=UNKNOWN
PartitionName=LocalQ Nodes=ALL Default=YES MaxTime=INFINITE State=UP
EOF$ sudo chmod 755 /etc/slurm/
```
Then start slurm:
```
sudo systemctl start slurmctld
sudo systemctl start slurmd
```
And set your machine as idle so you can start queuing jobs with Caliper later.
```
sudo scontrol update nodename=localhost state=idle
sinfo
```
### Caliper:
#### Install:

```
$ git clone https://github.com/LLNL/Caliper.git
$ cd Caliper
$ mkdir build && cd build
$ cmake -DWITH_MPI=yes ..
$ make
$ make install
```
Here in the `cmake` command, there are more build flags in the documentation page that can be added to have support for OpenMP and Cuda as well.

#### Profile a region:
```
#include <caliper/cali.h>
```
In the areas of code you want to profile simply add:
```
cali_begin_region("appropriate name for easy analysis");
...
cali_end_region("appropriate name for easy analysis");
```
#### Building:
Run the following command to build the implementation you want to test. In this example I am testing the `mpi_cols` implementation for gemver.
```
mpicc -std=c11 -o test -I /home/mctsanta/DPHPC/gemver/mpi_cols -I /home/mctsanta/DPHPC -I /home/mctsanta/DPHPC/gemver -DPOLYBENCH_TIME -DN2=2000 /home/mctsanta/DPHPC/polybench.c /home/mctsanta/DPHPC/gemver/gemver.c /home/mctsanta/DPHPC/gemver/mpi_cols/mpi_v3.c -L<path to caliper installation>/lib -lcaliper 
```
#### Running:
Run the program with the following command:

```
CALI_CONFIG=runtime-report,profile.mpi srun -n 8 ./test
```
Again in the documentation you can see more report options for Caliper but these are the ones they suggest for MPI.
Your results should look something like this:
```
Path          Min time/rank Max time/rank Avg time/rank Time %    
kernel             0.035150      0.035150      0.035150  3.862603 
    MPI_Gatherv      0.874397      0.874397      0.874397 96.085520 
    MPI_Reduce       0.000039      0.000039      0.000039  0.004232 
MPI_Finalize       0.000005      0.000005      0.000005  0.000585 
MPI_Comm_dup       0.000027      0.000027      0.000027  0.003002 
```
