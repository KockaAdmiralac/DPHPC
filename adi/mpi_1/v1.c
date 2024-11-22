/* Include polybench common header. */
#include "polybench.h"

#define Q q[0]
#define P p[0]

/* Include benchmark-specific header. */
#include <mpi.h>

#include "adi.h"

#define MAX_PROCESSES 16

static int rank, world_size;
static int start_i, end_i;
static int bounds_per_process[MAX_PROCESSES][2];
static int elems_per_process[MAX_PROCESSES];
static int displs[MAX_PROCESSES];
static void *all_u, *all_v;

void initialise_benchmark(int argc, char** argv, int tsteps, int n, DATA_TYPE POLYBENCH_2D(u, N2, N2, n, n),
                          DATA_TYPE POLYBENCH_2D(v, N2, N2, n, n), DATA_TYPE POLYBENCH_2D(p, N2, N2, n, n),
                          DATA_TYPE POLYBENCH_2D(q, N2, N2, n, n)) {
    (void)tsteps;
    (void)v;
    (void)p;
    (void)q;

    POLYBENCH_2D_ARRAY_DECL(tmp_u, DATA_TYPE, N2, N2, n, n);
    POLYBENCH_2D_ARRAY_DECL(tmp_v, DATA_TYPE, N2, N2, n, n);
    all_u = tmp_u;
    all_v = tmp_v;

    int i, j, cnt = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    // All processes should know each other's bounds
    for (i = 0; i < world_size - 1; i++) {
        bounds_per_process[i][0] = i * _PB_N / world_size;
        bounds_per_process[i][1] = (i + 1) * _PB_N / world_size;
        elems_per_process[i] = (bounds_per_process[i][1] - bounds_per_process[i][0]) * _PB_N;
        displs[i] = cnt;
        cnt += elems_per_process[i];
    }
    bounds_per_process[world_size - 1][0] = (world_size - 1) * _PB_N / world_size;
    bounds_per_process[world_size - 1][1] = _PB_N - 1;
    bounds_per_process[0][0] = 1;
    elems_per_process[world_size - 1] =
        (bounds_per_process[world_size - 1][1] - bounds_per_process[world_size - 1][0] + 1) * _PB_N;
    displs[world_size - 1] = cnt;

    start_i = bounds_per_process[rank][0];
    end_i = bounds_per_process[rank][1];

    for (i = 0; i < n; i++) {
        v[i][0] = SCALAR_VAL(1.0);
        v[i][_PB_N - 1] = SCALAR_VAL(1.0);
        for (j = 0; j < n; j++) {
            u[i][j] = (DATA_TYPE)(i + n - j) / n;
        }
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
    POLYBENCH_FREE_ARRAY(all_u);
    POLYBENCH_FREE_ARRAY(all_v);
    MPI_Finalize();
    if (rank != 0) {
        POLYBENCH_FREE_ARRAY(u);
        POLYBENCH_FREE_ARRAY(v);
        POLYBENCH_FREE_ARRAY(p);
        POLYBENCH_FREE_ARRAY(q);
        exit(0);
    }
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
    P[0] = SCALAR_VAL(0.0);
    Q[0] = SCALAR_VAL(1.0);

    // Sequential execution of this loop. Parallelization not possible due to arrays u and v where they store results
    // from previous iterations.
    for (t = 1; t <= _PB_TSTEPS; t++) {
        // Column Sweep --> Parallelizable! Each row is independent of one another.
        for (i = start_i; i < end_i; i++) {
            for (j = 1; j < _PB_N - 1; j++) {
                P[j] = -c / (a * P[j - 1] + b);
                Q[j] = (-d * u[j][i - 1] + (SCALAR_VAL(1.0) + SCALAR_VAL(2.0) * d) * u[j][i] - f * u[j][i + 1] -
                        a * Q[j - 1]) /
                       (a * P[j - 1] + b);
            }
            for (j = _PB_N - 2; j >= 1; j--) {
                v[i][j] = P[j] * v[i][j + 1] + Q[j];
            }
        }
        // Sync array v among all processes here!
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, v, elems_per_process, displs, MPI_DOUBLE, MPI_COMM_WORLD);

        // Row Sweep
        for (i = start_i; i < end_i; i++) {
            u[i][0] = SCALAR_VAL(1.0);
            for (j = 1; j < _PB_N - 1; j++) {
                P[j] = -f / (d * P[j - 1] + e);
                Q[j] = (-a * v[j][i - 1] + (SCALAR_VAL(1.0) + SCALAR_VAL(2.0) * a) * v[j][i] - c * v[j][i + 1] -
                        d * Q[j - 1]) /
                       (d * P[j - 1] + e);
            }
            u[i][_PB_N - 1] = SCALAR_VAL(1.0);
            for (j = _PB_N - 2; j >= 1; j--) {
                u[i][j] = P[j] * u[i][j + 1] + Q[j];
            }
        }
        // Sync here the array u only once again!
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, u, elems_per_process, displs, MPI_DOUBLE, MPI_COMM_WORLD);
    }
}
