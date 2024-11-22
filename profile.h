#ifdef ENABLE_PROFILING
#include <stdio.h>
#if defined(_OPENMP)
#include <omp.h>
#define START_PROFILE(marker) const double start_##marker = omp_get_wtime()
#define END_PROFILE(marker)                      \
    const double end_##marker = omp_get_wtime(); \
    printf(#marker " %lf\n", end_##marker - start_##marker)
#elif defined(MPI_VERSION)
#include <mpi.h>
#define START_PROFILE(marker) const double start_##marker = MPI_Wtime()
#define END_PROFILE(marker)                  \
    const double end_##marker = MPI_Wtime(); \
    printf(#marker " %lf\n", end_##marker - start_##marker)
#elif defined(__CUDACC__)
#include <cuda_runtime.h>
#define START_PROFILE(marker)         \
    cudaEvent_t start_##marker;       \
    cudaEvent_t end_##marker;         \
    cudaEventCreate(&start_##marker); \
    cudaEventCreate(&end_##marker);   \
    cudaEventRecord(start_##marker, 0)
#define END_PROFILE(marker)                                                 \
    cudaEventRecord(stop_##marker, 0);                                      \
    cudaEventSynchronize(stop_##marker);                                    \
    double elapsed_##marker = 0;                                            \
    cudaEventElapsedTime(&elapsed_##marker, start_##marker, stop_##marker); \
    printf(#marker " %lf\n", elapsed_##marker)
#else
#include <time.h>
#define START_PROFILE(marker) const clock_t start_##marker = clock()
#define END_PROFILE(marker)               \
    const clock_t end_##marker = clock(); \
    printf(#marker " %lf\n", (double)(end_##marker - start_##marker) / CLOCKS_PER_SEC)
#endif
#else
#define START_PROFILE(marker)
#define END_PROFILE(marker)
#endif
