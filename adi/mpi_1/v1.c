/* Include polybench common header. */
#include "polybench.h"

/* Include benchmark-specific header. */
#include <mpi.h>
#include <stdio.h>

#include "adi.h"

#define MAX_PROCESSES 16

typedef struct {
    DATA_TYPE POLYBENCH_2D(u, N2, N2, n, n);
    DATA_TYPE POLYBENCH_2D(v, N2, N2, n, n);
    DATA_TYPE POLYBENCH_1D(p, N2, n);
    DATA_TYPE POLYBENCH_1D(q, N2, n);
    int tsteps;
    int n;
    int argc;
    char** argv;
    int rank, world_size;
    int start_i, end_i;
    int bounds_per_process[MAX_PROCESSES][2];
    int elems_per_process[MAX_PROCESSES];
    int displs[MAX_PROCESSES];
} mpi_adi_data_t;

void initialise_benchmark(int argc, char** argv, int tsteps, int n, void** gen_data_ptr) {
    *gen_data_ptr = polybench_alloc_data(1, sizeof(mpi_adi_data_t));
    mpi_adi_data_t* data_ptr = (mpi_adi_data_t*)(*gen_data_ptr);

    data_ptr->tsteps = tsteps;
    data_ptr->n = n;
    data_ptr->argc = argc;
    data_ptr->argv = argv;

    int i, j;

    for (i = 0; i < n; i++) {
        data_ptr->p[i] = 0.0;
        data_ptr->q[i] = 0.0;
        for (j = 0; j < n; j++) {
            data_ptr->u[i][j] = (DATA_TYPE)(i + n - j) / n;
            data_ptr->v[i][j] = 0.0;
        }
    }
    int cnt = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &data_ptr->rank);
    MPI_Comm_size(MPI_COMM_WORLD, &data_ptr->world_size);
    // All processes should know each other's bounds
    for (i = 0; i < data_ptr->world_size - 1; i++) {
        data_ptr->bounds_per_process[i][0] = i * _PB_N / data_ptr->world_size;
        data_ptr->bounds_per_process[i][1] = (i + 1) * _PB_N / data_ptr->world_size;
        data_ptr->elems_per_process[i] =
            (data_ptr->bounds_per_process[i][1] - data_ptr->bounds_per_process[i][0]) * _PB_N;
        data_ptr->displs[i] = cnt;
        cnt += data_ptr->elems_per_process[i];
    }
    data_ptr->bounds_per_process[data_ptr->world_size - 1][0] =
        (data_ptr->world_size - 1) * _PB_N / data_ptr->world_size;
    data_ptr->bounds_per_process[data_ptr->world_size - 1][1] = _PB_N - 1;
    data_ptr->bounds_per_process[0][0] = 1;
    data_ptr->elems_per_process[data_ptr->world_size - 1] =
        (data_ptr->bounds_per_process[data_ptr->world_size - 1][1] -
         data_ptr->bounds_per_process[data_ptr->world_size - 1][0] + 1) *
        _PB_N;
    data_ptr->displs[data_ptr->world_size - 1] = cnt;

    data_ptr->start_i = data_ptr->bounds_per_process[data_ptr->rank][0];
    data_ptr->end_i = data_ptr->bounds_per_process[data_ptr->rank][1];

    for (i = 0; i < n; i++) {
        data_ptr->v[i][0] = SCALAR_VAL(1.0);
        data_ptr->v[i][_PB_N - 1] = SCALAR_VAL(1.0);
    }
}

void finish_benchmark(void* gen_data_ptr) {
    (void)gen_data_ptr;
    MPI_Finalize();
}

bool should_print_counter(void* gen_data_ptr) {
    mpi_adi_data_t* data_ptr = (mpi_adi_data_t*)gen_data_ptr;
    return data_ptr->rank == 0;
}

void print_data(int argc, char** argv, int n, void* gen_data_ptr) {
    fflush(stdout);
    mpi_adi_data_t* data_ptr = (mpi_adi_data_t*)gen_data_ptr;
    if (data_ptr->rank == 0) {
        default_adi_data_t* adi_data = polybench_alloc_data(1, sizeof(default_adi_data_t));
        int i, j;
        for (i = 0; i < n; i++)
            for (j = 0; j < n; j++) adi_data->u[i][j] = data_ptr->u[i][j];
        for (i = 0; i < n; i++)
            for (j = 0; j < n; j++) adi_data->v[i][j] = data_ptr->v[i][j];
        default_print_data(argc, argv, n, adi_data);
    }
}

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
/* Based on a Fortran code fragment from Figure 5 of
 * "Automatic Data and Computation Decomposition on Distributed Memory Parallel
 * Computers" by Peizong Lee and Zvi Meir Kedem, TOPLAS, 2002
 */
void kernel_adi(void* gen_data_ptr) {
    mpi_adi_data_t* data_ptr = (mpi_adi_data_t*)gen_data_ptr;
    int n = data_ptr->n;
    int tsteps = data_ptr->tsteps;
    int start_i = data_ptr->start_i;
    int end_i = data_ptr->end_i;

    int t, i, j;
    DATA_TYPE DX, DY, DT;
    DATA_TYPE B1, B2;
    DATA_TYPE mul1, mul2;
    DATA_TYPE a, b, c, d, e, f;
    DATA_TYPE* p = data_ptr->p;
    DATA_TYPE* q = data_ptr->q;

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
    p[0] = SCALAR_VAL(0.0);
    q[0] = SCALAR_VAL(1.0);

    // Sequential execution of this loop. Parallelization not possible due to arrays u and v where they store results
    // from previous iterations.
    for (t = 1; t <= _PB_TSTEPS; t++) {
        // Column Sweep --> Parallelizable! Each row is independent of one another.

        for (i = start_i; i < end_i; i++) {
            for (j = 1; j < _PB_N - 1; j++) {
                p[j] = -c / (a * p[j - 1] + b);
                q[j] = (-d * data_ptr->u[j][i - 1] + (SCALAR_VAL(1.0) + SCALAR_VAL(2.0) * d) * data_ptr->u[j][i] -
                        f * data_ptr->u[j][i + 1] - a * q[j - 1]) /
                       (a * p[j - 1] + b);
            }
            for (j = _PB_N - 2; j >= 1; j--) {
                data_ptr->v[i][j] = p[j] * data_ptr->v[i][j + 1] + q[j];
            }
        }
        // Sync array v among all processes here!
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, data_ptr->v, data_ptr->elems_per_process, data_ptr->displs,
                       MPI_DOUBLE, MPI_COMM_WORLD);
        // Row Sweep
        for (i = data_ptr->start_i; i < data_ptr->end_i; i++) {
            data_ptr->u[i][0] = SCALAR_VAL(1.0);
            for (j = 1; j < _PB_N - 1; j++) {
                p[j] = -f / (d * p[j - 1] + e);
                q[j] = (-a * data_ptr->v[j][i - 1] + (SCALAR_VAL(1.0) + SCALAR_VAL(2.0) * a) * data_ptr->v[j][i] -
                        c * data_ptr->v[j][i + 1] - d * q[j - 1]) /
                       (d * p[j - 1] + e);
            }
            data_ptr->u[i][_PB_N - 1] = SCALAR_VAL(1.0);
            for (j = _PB_N - 2; j >= 1; j--) {
                data_ptr->u[i][j] = p[j] * data_ptr->u[i][j + 1] + q[j];
            }
        }
        // Sync here the array u only once again!
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, data_ptr->u, data_ptr->elems_per_process, data_ptr->displs,
                       MPI_DOUBLE, MPI_COMM_WORLD);
    }
}
