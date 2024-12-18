/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "adi.h"

typedef struct {
    DATA_TYPE POLYBENCH_1D(u, N2 *N2, n *n);
    DATA_TYPE POLYBENCH_1D(v, N2 *N2, n *n);
    DATA_TYPE POLYBENCH_1D(p, N2, n);
    DATA_TYPE POLYBENCH_1D(q, N2, n);
    int tsteps;
    int n;
    int argc;
    char **argv;
} serial_adi_data_t;

void initialise_benchmark(int argc, char **argv, int tsteps, int n, void **gen_data_ptr) {
    *gen_data_ptr = polybench_alloc_data(1, sizeof(serial_adi_data_t));
    serial_adi_data_t *data_ptr = (serial_adi_data_t *)(*gen_data_ptr);
    data_ptr->tsteps = tsteps;
    data_ptr->n = n;
    data_ptr->argc = argc;
    data_ptr->argv = argv;

    int i, j;

    for (i = 0; i < n; i++) {
        data_ptr->p[i] = 0.0;
        data_ptr->q[i] = 0.0;
        for (j = 0; j < n; j++) {
            data_ptr->u[(i)*n + (j)] = (DATA_TYPE)(i + n - j) / n;
            data_ptr->v[(i)*n + (j)] = 0.0;
        }
    }
}
/* Main computational kernel. The whole function will be timed,
   including the call and return. */
/* Based on a Fortran code fragment from Figure 5 of
 * "Automatic Data and Computation Decomposition on Distributed Memory Parallel
 * Computers" by Peizong Lee and Zvi Meir Kedem, TOPLAS, 2002
 */
void kernel_adi_orig(int tsteps, int n, DATA_TYPE POLYBENCH_1D(u, N2 *N2, n *n),
                     DATA_TYPE POLYBENCH_1D(v, N2 *N2, n *n), DATA_TYPE POLYBENCH_1D(p, N2, n),
                     DATA_TYPE POLYBENCH_1D(q, N2, n)) {
    int t, i, j;
    DATA_TYPE DX, DY, DT;
    DATA_TYPE B1, B2;
    DATA_TYPE mul1, mul2;
    DATA_TYPE a, b, c, d, e, f;

    DX = SCALAR_VAL(1.0) / (DATA_TYPE)_PB_N;
    DY = SCALAR_VAL(1.0) / (DATA_TYPE)_PB_N;
    DT = SCALAR_VAL(1.0) / (DATA_TYPE)_PB_TSTEPS;
    B1 = SCALAR_VAL(2.0);
    B2 = SCALAR_VAL(1.0);
    mul1 = B1 * DT / (DX * DX);
    mul2 = B2 * DT / (DY * DY);

    a = -mul1 / SCALAR_VAL(2.0);
    b = SCALAR_VAL(1.0) + mul1;
    c = a;
    d = -mul2 / SCALAR_VAL(2.0);
    e = SCALAR_VAL(1.0) + mul2;
    f = d;

    for (t = 1; t <= _PB_TSTEPS; t++) {
        // Column Sweep
        for (i = 1; i < _PB_N - 1; i++) {
            v[(0) * n + (i)] = SCALAR_VAL(1.0);
            p[0] = SCALAR_VAL(0.0);
            q[0] = v[(0) * n + (i)];
            for (j = 1; j < _PB_N - 1; j++) {
                p[j] = -c / (a * p[j - 1] + b);
                q[j] = (-d * u[(j)*n + (i - 1)] + (SCALAR_VAL(1.0) + SCALAR_VAL(2.0) * d) * u[(j)*n + (i)] -
                        f * u[(j)*n + (i + 1)] - a * q[j - 1]) /
                       (a * p[j - 1] + b);
            }

            v[(_PB_N - 1) * n + (i)] = SCALAR_VAL(1.0);
            for (j = _PB_N - 2; j >= 1; j--) {
                v[(j)*n + (i)] = p[j] * v[(j + 1) * n + (i)] + q[j];
            }
        }
        // Row Sweep
        for (i = 1; i < _PB_N - 1; i++) {
            u[(i)*n + (0)] = SCALAR_VAL(1.0);
            p[0] = SCALAR_VAL(0.0);
            q[0] = u[(i)*n + (0)];
            for (j = 1; j < _PB_N - 1; j++) {
                p[j] = -f / (d * p[j - 1] + e);
                q[j] = (-a * v[(i - 1) * n + (j)] + (SCALAR_VAL(1.0) + SCALAR_VAL(2.0) * a) * v[(i)*n + (j)] -
                        c * v[(i + 1) * n + (j)] - d * q[j - 1]) /
                       (d * p[j - 1] + e);
            }
            u[(i)*n + (_PB_N - 1)] = SCALAR_VAL(1.0);
            for (j = _PB_N - 2; j >= 1; j--) {
                u[(i)*n + (j)] = p[j] * u[(i)*n + (j + 1)] + q[j];
            }
        }
    }
}

void kernel_adi(void *gen_data_ptr) {
    serial_adi_data_t *data_ptr = (serial_adi_data_t *)gen_data_ptr;
    kernel_adi_orig(data_ptr->tsteps, data_ptr->n, data_ptr->u, data_ptr->v, data_ptr->p, data_ptr->q);
}