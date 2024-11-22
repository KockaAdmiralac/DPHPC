/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include <omp.h>

#include "adi.h"
#define BLOCK_SIZE 6

void initialise_benchmark(int argc, char** argv, int tsteps, int n, DATA_TYPE POLYBENCH_2D(u, N2, N2, n, n),
                          DATA_TYPE POLYBENCH_2D(v, N2, N2, n, n), DATA_TYPE POLYBENCH_2D(p, N2, N2, n, n),
                          DATA_TYPE POLYBENCH_2D(q, N2, N2, n, n)) {
    (void)tsteps;
    (void)argc;
    (void)argv;
    (void)n;
    (void)u;
    (void)v;
    (void)p;
    (void)q;

    int i, j;

    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++) {
            u[i][j] = (DATA_TYPE)(i + n - j) / n;
        }
}

void finish_benchmark(int tsteps, int n, DATA_TYPE POLYBENCH_2D(u, N2, N2, n, n),
                      DATA_TYPE POLYBENCH_2D(v, N2, N2, n, n), DATA_TYPE POLYBENCH_2D(p, N2, N2, n, n),
                      DATA_TYPE POLYBENCH_2D(q, N2, N2, n, n)) {
    (void)tsteps;
    (void)n;
    (void)u;
    (void)v;
    (void)p;
    (void)q;
}

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
/* Based on a Fortran code fragment from Figure 5 of
 * "Automatic Data and Computation Decomposition on Distributed Memory Parallel
 * Computers" by Peizong Lee and Zvi Meir Kedem, TOPLAS, 2002
 */
void kernel_adi(int tsteps, int n, DATA_TYPE POLYBENCH_2D(u, N2, N2, n, n), DATA_TYPE POLYBENCH_2D(v, N2, N2, n, n),
                DATA_TYPE POLYBENCH_2D(p, N2, N2, n, n), DATA_TYPE POLYBENCH_2D(q, N2, N2, n, n)) {
    int t, j;
    DATA_TYPE DX, DY, DT;
    DATA_TYPE B1, B2;
    DATA_TYPE mul1, mul2;
    DATA_TYPE a, b, c, d, e, f;

    DX = SCALAR_VAL(1.0) / (DATA_TYPE)_PB_N;
    DY = DX;
    DT = SCALAR_VAL(1.0) / (DATA_TYPE)_PB_TSTEPS;
    B1 = SCALAR_VAL(2.0);
    B2 = SCALAR_VAL(1.0);
    mul1 = B1 * DT / (DX * DX);
    mul2 = B2 * DT / (DY * DY);

    // Constants for reused values
    a = -mul1 / SCALAR_VAL(2.0);
    b = SCALAR_VAL(1.0) + mul1;
    c = a;
    d = -mul2 / SCALAR_VAL(2.0);
    e = SCALAR_VAL(1.0) + mul2;
    f = d;

    // Precompute constants for the loops
    DATA_TYPE const_1_2d = SCALAR_VAL(1.0) + SCALAR_VAL(2.0) * d;
    DATA_TYPE const_neg_a = -a;
    DATA_TYPE const_1_2a = SCALAR_VAL(1.0) + SCALAR_VAL(2.0) * a;
    DATA_TYPE const_neg_c = -c;
    DATA_TYPE denom_inv;
    //  TODO also try to store local variables p[i][j - 1] and q[i][j - 1] and see if it improves performance or reduces
    //  noise to OPENMP
    int jj;
    for (t = 1; t <= _PB_TSTEPS; t++) {
// Column Sweep
#pragma omp parallel for schedule(dynamic) private(denom_inv, j, jj)
        for (int ii = 1; ii < _PB_N - 1; ii += BLOCK_SIZE) {
            for (int jj = 1; jj < _PB_N - 1; jj += BLOCK_SIZE) {
                for (int i = ii; i < ii + BLOCK_SIZE && i < _PB_N - 1; i++) {
                    v[0][i] = SCALAR_VAL(1.0);
                    p[i][0] = SCALAR_VAL(0.0);
                    q[i][0] = v[0][i];
                    for (int j = jj; j < jj + BLOCK_SIZE && j < _PB_N - 1; j++) {
                        denom_inv = SCALAR_VAL(1.0) / (a * p[i][j - 1] + b);
                        p[i][j] = -c * denom_inv;
                        q[i][j] =
                            (-d * u[j][i - 1] + const_1_2d * u[j][i] - f * u[j][i + 1] + const_neg_a * q[i][j - 1]) *
                            denom_inv;
                    }
                }
            }
        }
#pragma omp parallel for schedule(dynamic) private(j)
        // // Backward Pass (Sequential)
        for (int i = 1; i < _PB_N - 1; i++) {
            v[_PB_N - 1][i] = SCALAR_VAL(1.0);
            for (j = _PB_N - 2; j >= 8; j -= 8) {
                v[j][i] = p[i][j] * v[j + 1][i] + q[i][j];
                v[j - 1][i] = p[i][j - 1] * v[j][i] + q[i][j - 1];
                v[j - 2][i] = p[i][j - 2] * v[j - 1][i] + q[i][j - 2];
                v[j - 3][i] = p[i][j - 3] * v[j - 2][i] + q[i][j - 3];
                v[j - 4][i] = p[i][j - 4] * v[j - 3][i] + q[i][j - 4];
                v[j - 5][i] = p[i][j - 5] * v[j - 4][i] + q[i][j - 5];
                v[j - 6][i] = p[i][j - 6] * v[j - 5][i] + q[i][j - 6];
                v[j - 7][i] = p[i][j - 7] * v[j - 6][i] + q[i][j - 7];
            }
            // // Handle remaining elements
            for (; j >= 1; j--) {
                v[j][i] = p[i][j] * v[j + 1][i] + q[i][j];
            }
        }
// Row Sweep
#pragma omp parallel for schedule(dynamic) private(j, denom_inv, jj)
        for (int ii = 1; ii < _PB_N - 1; ii += BLOCK_SIZE) {
            for (int jj = 1; jj < _PB_N - 1; jj += BLOCK_SIZE) {
                for (int i = ii; i < ii + BLOCK_SIZE && i < _PB_N - 1; i++) {
                    u[i][0] = SCALAR_VAL(1.0);
                    p[i][0] = SCALAR_VAL(0.0);
                    q[i][0] = u[i][0];

                    for (int j = jj; j < jj + BLOCK_SIZE && j < _PB_N - 1; j++) {
                        denom_inv = SCALAR_VAL(1.0) / (d * p[i][j - 1] + e);
                        p[i][j] = -f * denom_inv;
                        q[i][j] = (const_neg_a * v[i - 1][j] + const_1_2a * v[i][j] + const_neg_c * v[i + 1][j] -
                                   d * q[i][j - 1]) *
                                  denom_inv;
                    }
                }
            }
        }
#pragma omp parallel for schedule(dynamic) private(j)
        for (int i = 1; i < _PB_N - 1; i++) {
            u[i][_PB_N - 1] = SCALAR_VAL(1.0);
            j = _PB_N - 2;
            for (; j >= 8; j -= 8) {
                u[i][j] = p[i][j] * u[i][j + 1] + q[i][j];
                u[i][j - 1] = p[i][j - 1] * u[i][j] + q[i][j - 1];
                u[i][j - 2] = p[i][j - 2] * u[i][j - 1] + q[i][j - 2];
                u[i][j - 3] = p[i][j - 3] * u[i][j - 2] + q[i][j - 3];
                u[i][j - 4] = p[i][j - 4] * u[i][j - 3] + q[i][j - 4];
                u[i][j - 5] = p[i][j - 5] * u[i][j - 4] + q[i][j - 5];
                u[i][j - 6] = p[i][j - 6] * u[i][j - 5] + q[i][j - 6];
                u[i][j - 7] = p[i][j - 7] * u[i][j - 6] + q[i][j - 7];
            }
            // Handle remaining elements
            for (; j >= 1; j--) {
                u[i][j] = p[i][j] * u[i][j + 1] + q[i][j];
            }
        }
    }
}