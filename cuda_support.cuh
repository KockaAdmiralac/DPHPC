#pragma once
#include <cublas_v2.h>
#include <cuda.h>

// from https://stackoverflow.com/a/14038590
#define gpuErrchk(ans) \
    { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

#define gpuCublasErrchk(ans) \
    { gpuCublasAssert((ans), __FILE__, __LINE__); }

inline void gpuCublasAssert(cublasStatus_t status, const char *file, int line, bool abort = true) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "GPUassert: %d %s %d\n", status, file, line);
        if (abort) exit(status);
    }
}