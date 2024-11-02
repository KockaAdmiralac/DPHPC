# Running test.py on Euler
Save the shell script below to run_gemver_serial_base.sh.
```
#!/bin/sh
python3 test.py --benchmark gemver --variants serial_base --runs 1000 --n 100
```

First run this on the login node:

    module load stack/2024-06 python/3.11.6

Now run the shell script on the CPU hosts with:

    sbatch run_gemver_serial_base.sh

# Specifying specialised options
At the root, benchmark and variant levels you can add a `dphpc_md.json` file to specify options test.py should follow.  Files in deeper directory levels override their parents, where lists and dictionaries are extended/updated while other types are replaced.  You can specify the dictionary `defines` to set preprocessor definitions (definitions without a value should be given the empty string as value) and the list `extra_compile_options` to pass extra options to the compiler.

## Linting
This repository uses a [GitHub Action](.github/workflows/lint.yml) to check whether all files are linted adequately. If you're modifying C/CUDA/Python code, you can use `make format` to format the code consistently with other files, and `pyright .` to check for Python type errors. Alternatively, you can install adequate extensions for black, Pyright and clang-format in your IDE. You can also install an extension for `isort` to keep the imports ordered correctly, but this is not required to pass the linting.

Linting checks do not prevent you from merging your code, but in case you want to skip the GitHub Action anyways, you can use one of the [skipping words](https://docs.github.com/en/actions/managing-workflow-runs-and-deployments/managing-workflow-runs/skipping-workflow-runs) in your commit message or commit description, such as `[skip ci]`.
