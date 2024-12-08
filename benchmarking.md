# Comparing the project with other benchmarks
There is a number of other benchmarks that you can use to compare this project with, some of which are listed below.

## OpenBLAS/MKL
Variants of `gemver` have been implemented using BLAS routines from OpenBLAS and Intel MKL. You can run them as `openmp_mkl` and `openmp_openblas` variants in this repository.

## Polly
The [Polly](https://polly.llvm.org/) optimizer can be used to optimize the code. This procedure can be used to run a Polly-optimized baseline:

```bash
# Compile Polly
git clone --depth 1 https://github.com/llvm/llvm-project.git ~/llvm && cd ~/llvm
mkdir build && cd build
sudo apt-get install -y cmake
cmake '-DLLVM_ENABLE_PROJECTS=clang;polly' '-DCMAKE_BUILD_TYPE=Release' '-DLLVM_ENABLE_ASSERTIONS=ON' ../llvm
cmake --build . --config Release --parallel 16
# <cd to the DPHPC repo here>
# Compile the baseline code
~/llvm/build/bin/clang -O3 -Wall -Wextra -ffast-math -march=native -mllvm -polly -mllvm -polly-parallel -mllvm -polly-num-threads=16 -mllvm -polly-vectorizer=stripmine -lgomp -o polly -Igemver -I. -DPOLYBENCH_TIME -DDISABLE_CHECKING -DN2=16384 polybench.c gemver/gemver.c gemver/serial_base/serial_impl.c
# Run the baseline code, which prints the runtime
./polly
```

### NPBench
The [NPBench](https://github.com/spcl/npbench) project has many of the same benchmarks used here, implemented in various Python parallelization frameworks. The following can be used to run them:

```bash
# Clone the repo
git clone https://github.com/spcl/npbench.git ~/npbench && cd ~/npbench
# Install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
sudo apt-get install -y libatlas-base-dev python-dev python-ply python-numpy python3-pythran libopenblas-dev
echo -e '[compiler]\nblas=openblas' > ~/.pythranrc # See issue #1414 at the Pythran repo
pip install numba dace pythran
# <change the desired dataset size in bench_info/gemver.json>
# Run the benchmark
python run_benchmark.py -b gemver -p L -f dace_cpu # or numpy, numba, pythran
```
