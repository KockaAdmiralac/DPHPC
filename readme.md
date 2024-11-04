# PolyBench parallelization
This project aims to parallelize [PolyBench](https://github.com/MatthiasJReisinger/PolyBenchC-4.2.1) using OpenMP, MPI and CUDA, as a part of the [Design of Parallel and High-Performance Computing](https://spcl.inf.ethz.ch/Teaching/2024-dphpc/) at [ETH Zürich](https://ethz.ch/), in Winter Semester 2024. Students working on this project are:

- [George Manos](https://github.com/WatchmeDoc)
- [Giorgos Drosos](https://gdrosos.github.io/)
- [Maritina Tsanta](https://github.com/mctsnt)
- [Paolo Celati](https://github.com/pcelati)
- [Luka Simić](https://kocka.tech/)

## Benchmark
You can benchmark your code by using the `test.py` script. It can compile your code with the appropriate options, run it with the appropriate runtime, check whether its output is correct and compare it with other implementations. The most important options you can provide are:

- `--benchmark`: Which benchmark are you running (either `gemver` or `adi`)
- `--variants`: Which variant of the benchmark are you running
    - Variants are named starting with the type of parallelization performed in them: `serial`, `openmp`, `mpi` or `cuda`.
        - Depending on this, your program will be compiled with the appropriate compiler, and run with the appropriate runtime.
    - Example:
        - You have developed an MPI parallelization of a benchmark with cache efficiency, and named it `mpi_cache_efficiency`.
        - Your colleague has developed a parallelization of that benchmark using OpenMP's tasks, and named it `openmp_tasks`.
        - You can compare the baseline, your parallelization and your colleague's parallelization with `--variant serial_base,openmp_tasks,mpi_cache_efficiency`.
- `--threads`: Comma-separated list of how many threads should the benchmark be run in
    - For example, `--threads 1,2,4,8` runs the benchmark with 1, 2, 4 and 8 threads.
- `--runs`: How many times to run the benchmark and average the timing results
    - When testing whether your benchmark works, you can pass a lower number here as checking whether your output data matches the ground truth can take some time.
    - When testing only the performance of your code, you can pass a higher number together with the `--disable_checking` option to make the benchmark finish faster.
- `--set_defines`: A comma-separated list of preprocessor defines to pass to the compiler
    - For example, `--set_defines N2=12000` acts just like a `#define N2 12000` would. It increases the input size to 12000, so you can use this option to give larger input sizes while testing performance.
    - Alternatively, you can specify one of `MINI_DATASET`, `SMALL_DATASET`, `MEDIUM_DATASET`, `LARGE_DATASET` and `EXTRALARGE_DATASET` using `--set_defines SMALL_DATASET=1` to set the relevant define which automatically determines the `N2` (and `TSTEPS`, for the `adi` benchmark) values.

This is how an invocation of `test.py` might look like:
```console
python3 test.py \
    --benchmark gemver \
    --variants serial_base,openmp_tasks,mpi_cache_efficiency \
    --threads 1,2,4,8 \
    --runs 100 \
    --disable_checking \
    --set_defines N2=12000
```

Another example, using `adi`:
```console
python3 test.py \
    --benchmark adi \
    --variants serial_base \
    --runs 10 \
    --set_defines SMALL_DATASET=1
```

You can read about the other options in `./test.py --help`.

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
