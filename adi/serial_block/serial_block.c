/* Include polybench common header. */
#include <polybench.h>

#define BLOCK_SIZE 4
/* Include benchmark-specific header. */
#include "adi.h"

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
/* Based on a Fortran code fragment from Figure 5 of
 * "Automatic Data and Computation Decomposition on Distributed Memory Parallel
 * Computers" by Peizong Lee and Zvi Meir Kedem, TOPLAS, 2002
 */
void kernel_adi_orig(int tsteps, int n, DATA_TYPE POLYBENCH_2D(u, N2, N2, n, n),
                     DATA_TYPE POLYBENCH_2D(v, N2, N2, n, n), DATA_TYPE POLYBENCH_2D(p, N2, N2, n, n),
                     DATA_TYPE POLYBENCH_2D(q, N2, N2, n, n)) {
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
    DATA_TYPE prev;
    //  store local variable p[i][j - 1] and q[i][j - 1] to foo and reuse
    //
    for (t = 1; t <= _PB_TSTEPS; t++) {
        // Column Sweep
        for (int ii = 1; ii < _PB_N - 1; ii += BLOCK_SIZE) {
            for (int jj = 1; jj < _PB_N - 1; jj += BLOCK_SIZE) {
                for (int i = ii; i < ii + BLOCK_SIZE && i < _PB_N - 1; i++) {
                    v[0][i] = SCALAR_VAL(1.0);
                    p[i][0] = SCALAR_VAL(0.0);
                    q[i][0] = v[0][i];
                    prev = p[i][jj - 1];
                    for (int j = jj; j < jj + BLOCK_SIZE && j < _PB_N - 1; j++) {
                        denom_inv = SCALAR_VAL(1.0) / (a * prev + b);
                        p[i][j] = -c * denom_inv;
                        q[i][j] =
                            (-d * u[j][i - 1] + const_1_2d * u[j][i] - f * u[j][i + 1] + const_neg_a * q[i][j - 1]) *
                            denom_inv;
                        prev = p[i][j];
                    }
                }
            }
        }
        // // Backward Pass (Sequential)
        for (int i = 1; i < _PB_N - 1; i++) {
            v[_PB_N - 1][i] = SCALAR_VAL(1.0);
            for (j = _PB_N - 2; j >= 1; j--) {
                v[j][i] = p[i][j] * v[j + 1][i] + q[i][j];
            }
        }
        // Row Sweep
        for (int ii = 1; ii < _PB_N - 1; ii += BLOCK_SIZE) {
            for (int jj = 1; jj < _PB_N - 1; jj += BLOCK_SIZE) {
                for (int i = ii; i < ii + BLOCK_SIZE && i < _PB_N - 1; i++) {
                    u[i][0] = SCALAR_VAL(1.0);
                    p[i][0] = SCALAR_VAL(0.0);
                    q[i][0] = u[i][0];
                    prev = p[i][jj - 1];
                    for (int j = jj; j < jj + BLOCK_SIZE && j < _PB_N - 1; j++) {
                        denom_inv = SCALAR_VAL(1.0) / (d * prev + e);
                        p[i][j] = -f * denom_inv;
                        q[i][j] = (const_neg_a * v[i - 1][j] + const_1_2a * v[i][j] + const_neg_c * v[i + 1][j] -
                                   d * q[i][j - 1]) *
                                  denom_inv;
                        prev = p[i][j];
                    }
                }
            }
        }

        for (int i = 1; i < _PB_N - 1; i++) {
            u[i][_PB_N - 1] = SCALAR_VAL(1.0);
            for (j = _PB_N - 2; j >= 1; j--) {
                u[i][j] = p[i][j] * u[i][j + 1] + q[i][j];
            }
        }
    }
}

void kernel_adi(void *gen_data_ptr) {
    default_adi_data_t *data_ptr = (default_adi_data_t *)gen_data_ptr;
    kernel_adi_orig(data_ptr->tsteps, data_ptr->n, data_ptr->u, data_ptr->v, data_ptr->p, data_ptr->q);
}