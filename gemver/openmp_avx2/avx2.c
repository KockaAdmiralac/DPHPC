#include <polybench.h>

#include "gemver.h"

#define BLOCK_SIZE 64

#include "immintrin.h"

// Merges the first and second loop.
// Based on openmp_reorder_reduce.
void kernel_gemver(int n, DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE POLYBENCH_2D(A, N2, N2, n, n),
                   DATA_TYPE POLYBENCH_1D(u1, N2, n), DATA_TYPE POLYBENCH_1D(v1, N2, n),
                   DATA_TYPE POLYBENCH_1D(u2, N2, n), DATA_TYPE POLYBENCH_1D(v2, N2, n),
                   DATA_TYPE POLYBENCH_1D(w, N2, n), DATA_TYPE POLYBENCH_1D(x, N2, n), DATA_TYPE POLYBENCH_1D(y, N2, n),
                   DATA_TYPE POLYBENCH_1D(z, N2, n)) {
#pragma omp parallel for collapse(2) reduction(+ : x[ : _PB_N])
    for (int i = 0; i < _PB_N; i += BLOCK_SIZE)
        for (int j = 0; j < _PB_N; j += BLOCK_SIZE)
            for (int bi = i; bi < i + BLOCK_SIZE && bi < _PB_N; ++bi) {
                __m256d u1_vec = _mm256_set1_pd(u1[bi]);
                __m256d u2_vec = _mm256_set1_pd(u2[bi]);
                __m256d y_vec = _mm256_set1_pd(y[bi]);
                for (int bj = j; bj < j + BLOCK_SIZE && bj < _PB_N; bj += 4) {
                    __m256d v1_vec = _mm256_loadu_pd(&v1[bj]);
                    __m256d v2_vec = _mm256_loadu_pd(&v2[bj]);
                    __m256d A_vec = _mm256_loadu_pd(&A[bi][bj]);
                    __m256d x_vec = _mm256_loadu_pd(&x[bj]);
                    __m256d A_vec2 = _mm256_fmadd_pd(u2_vec, v2_vec, _mm256_fmadd_pd(u1_vec, v1_vec, A_vec));
                    _mm256_storeu_pd(&A[bi][bj], A_vec2);
                    _mm256_storeu_pd(&x[bj], _mm256_fmadd_pd(A_vec2, y_vec, x_vec));
                }
            }

    __m256d beta_vec = _mm256_set1_pd(beta);
#pragma omp parallel for
    for (int i = 0; i < _PB_N; i += 4) {
        __m256d z_vec = _mm256_loadu_pd(&z[i]);
        __m256d x_vec = _mm256_loadu_pd(&x[i]);
        _mm256_storeu_pd(&x[i], _mm256_fmadd_pd(beta_vec, x_vec, z_vec));
    }

#pragma omp parallel for
    for (int i = 0; i < _PB_N; ++i)
        for (int j = 0; j < _PB_N; ++j) w[i] += alpha * A[i][j] * x[j];
}

void initialise_benchmark(int argc, char **argv, int n, DATA_TYPE *alpha, DATA_TYPE *beta,
                          DATA_TYPE POLYBENCH_2D(A, N2, N2, n, n), DATA_TYPE POLYBENCH_1D(u1, N2, n),
                          DATA_TYPE POLYBENCH_1D(v1, N2, n), DATA_TYPE POLYBENCH_1D(u2, N2, n),
                          DATA_TYPE POLYBENCH_1D(v2, N2, n), DATA_TYPE POLYBENCH_1D(w, N2, n),
                          DATA_TYPE POLYBENCH_1D(x, N2, n), DATA_TYPE POLYBENCH_1D(y, N2, n),
                          DATA_TYPE POLYBENCH_1D(z, N2, n)) {
    (void)argc;
    (void)argv;
    (void)n;
    (void)alpha;
    (void)beta;
    (void)A;
    (void)u1;
    (void)v1;
    (void)u2;
    (void)v2;
    (void)w;
    (void)x;
    (void)y;
    (void)z;

    int i, j;

    *alpha = 1.5;
    *beta = 1.2;

    DATA_TYPE fn = (DATA_TYPE)n;

    for (i = 0; i < n; i++) {
        u1[i] = i;
        u2[i] = ((i + 1) / fn) / 2.0;
        v1[i] = ((i + 1) / fn) / 4.0;
        v2[i] = ((i + 1) / fn) / 6.0;
        y[i] = ((i + 1) / fn) / 8.0;
        z[i] = ((i + 1) / fn) / 9.0;
        x[i] = 0.0;
        w[i] = 0.0;
        for (j = 0; j < n; j++) A[i][j] = (DATA_TYPE)(i * j % n) / n;
    }
}

void finish_benchmark(int n, DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE POLYBENCH_2D(A, N2, N2, n, n),
                      DATA_TYPE POLYBENCH_1D(u1, N2, n), DATA_TYPE POLYBENCH_1D(v1, N2, n),
                      DATA_TYPE POLYBENCH_1D(u2, N2, n), DATA_TYPE POLYBENCH_1D(v2, N2, n),
                      DATA_TYPE POLYBENCH_1D(w, N2, n), DATA_TYPE POLYBENCH_1D(x, N2, n),
                      DATA_TYPE POLYBENCH_1D(y, N2, n), DATA_TYPE POLYBENCH_1D(z, N2, n)) {
    (void)n;
    (void)alpha;
    (void)beta;
    (void)A;
    (void)u1;
    (void)v1;
    (void)u2;
    (void)v2;
    (void)w;
    (void)x;
    (void)y;
    (void)z;
}
