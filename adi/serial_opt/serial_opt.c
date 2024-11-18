/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "adi.h"

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
    int t, i, j;
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

    DATA_TYPE p_im1, q_im1, denom, denom_inv;

    for (t = 1; t <= _PB_TSTEPS; t++) {
        // Column Sweep
        for (i = 1; i < _PB_N - 1; i++) {
            v[0][i] = SCALAR_VAL(1.0);
            p[0][0] = SCALAR_VAL(0.0);
            q[0][0] = v[0][i];
            for (j = 1; j < _PB_N - 1; j++) {
                // Cache previous p[0][j - 1] and q[0][j - 1]
                p_im1 = p[0][j - 1];
                q_im1 = q[0][j - 1];

                // Avoid redundant calculations by caching values
                denom = a * p_im1 + b;
                denom_inv = SCALAR_VAL(1.0) / denom;
                p[0][j] = -c * denom_inv;
                q[0][j] = (-d * u[j][i - 1] + const_1_2d * u[j][i] - f * u[j][i + 1] + const_neg_a * q_im1) * denom_inv;
            }

            v[_PB_N - 1][i] = SCALAR_VAL(1.0);
            for (j = _PB_N - 2; j >= 1; j--) {
                v[j][i] = p[0][j] * v[j + 1][i] + q[0][j];
            }
        }

        // Row Sweep
        for (i = 1; i < _PB_N - 1; i++) {
            u[i][0] = SCALAR_VAL(1.0);
            p[0][0] = SCALAR_VAL(0.0);
            q[0][0] = u[i][0];
            for (j = 1; j < _PB_N - 1; j++) {
                // Cache previous p[0][j - 1] and q[0][j - 1]
                p_im1 = p[0][j - 1];
                q_im1 = q[0][j - 1];

                // Avoid redundant calculations by caching values
                denom = d * p_im1 + e;
                denom_inv = SCALAR_VAL(1.0) / denom;
                p[0][j] = -f * denom_inv;
                q[0][j] = (const_neg_a * v[i - 1][j] + const_1_2a * v[i][j] + const_neg_c * v[i + 1][j] - d * q_im1) * denom_inv;
            }

            u[i][_PB_N - 1] = SCALAR_VAL(1.0);
            for (j = _PB_N - 2; j >= 1; j--) {
                u[i][j] = p[0][j] * u[i][j + 1] + q[0][j];
            }
        }
    }
}