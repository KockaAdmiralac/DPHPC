/* Include polybench common header. */
#include "polybench.h"

/* Include benchmark-specific header. */
#include <mpi.h>
#include <string.h>

#include "adi.h"

#define MAX_PROCESSES 16

typedef struct {
    DATA_TYPE* u;
    DATA_TYPE* v;
    DATA_TYPE POLYBENCH_1D(p, N2, n);
    DATA_TYPE POLYBENCH_1D(q, N2, n);
    MPI_Win win_v;
    MPI_Win win_u;
    int tsteps;
    int n;
    int argc;
    char** argv;
    int rank, world_size;
    int start_i, end_i;
    int bounds_per_process[MAX_PROCESSES][2];
    MPI_Comm shared_comm;
} mpi_adi_data_t;

void initialise_benchmark(int argc, char** argv, int tsteps, int n, void** gen_data_ptr) {
    MPI_Init(&argc, &argv);

    *gen_data_ptr = polybench_alloc_data(1, sizeof(mpi_adi_data_t));
    mpi_adi_data_t* data_ptr = (mpi_adi_data_t*)(*gen_data_ptr);
    data_ptr->u = polybench_alloc_data(n * n, sizeof(DATA_TYPE));
    data_ptr->v = polybench_alloc_data(n * n, sizeof(DATA_TYPE));

    // Split communicator to include only shared memory processes
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &data_ptr->shared_comm);

    MPI_Comm_rank(data_ptr->shared_comm, &data_ptr->rank);
    MPI_Comm_size(data_ptr->shared_comm, &data_ptr->world_size);

    int i = 0, j;
    data_ptr->tsteps = tsteps;
    data_ptr->n = n;
    data_ptr->argc = argc;
    data_ptr->argv = argv;
    // All processes should know each other's bounds
    for (i = 0; i < data_ptr->world_size - 1; i++) {
        data_ptr->bounds_per_process[i][0] = i * _PB_N / data_ptr->world_size;
        data_ptr->bounds_per_process[i][1] = (i + 1) * _PB_N / data_ptr->world_size;
    }
    data_ptr->bounds_per_process[data_ptr->world_size - 1][0] =
        (data_ptr->world_size - 1) * _PB_N / data_ptr->world_size;
    data_ptr->bounds_per_process[data_ptr->world_size - 1][1] = _PB_N - 1;

    data_ptr->bounds_per_process[0][0] = 1;

    data_ptr->start_i = data_ptr->bounds_per_process[data_ptr->rank][0];
    data_ptr->end_i = data_ptr->bounds_per_process[data_ptr->rank][1];

    // Step 2: Allocate a shared memory window for the entire v array
    MPI_Aint v_size_bytes = (MPI_Aint)data_ptr->n * (MPI_Aint)data_ptr->n * (MPI_Aint)sizeof(DATA_TYPE);

    if (data_ptr->rank == 0) {
        // Rank 0 of shared_comm allocates the full memory
        MPI_Win_allocate_shared(v_size_bytes, sizeof(DATA_TYPE), MPI_INFO_NULL, data_ptr->shared_comm, &data_ptr->v,
                                &data_ptr->win_v);
    } else {
        // Other ranks allocate 0 and query
        MPI_Aint sz;
        int disp_unit;
        MPI_Win_allocate_shared(0, sizeof(DATA_TYPE), MPI_INFO_NULL, data_ptr->shared_comm, &data_ptr->v,
                                &data_ptr->win_v);
        MPI_Win_shared_query(data_ptr->win_v, 0, &sz, &disp_unit, &data_ptr->v);
    }
    // apply for u as well
    if (data_ptr->rank == 0) {
        // Rank 0 of shared_comm allocates the full memory
        MPI_Win_allocate_shared(v_size_bytes, sizeof(DATA_TYPE), MPI_INFO_NULL, data_ptr->shared_comm, &data_ptr->u,
                                &data_ptr->win_u);
    } else {
        // Other ranks allocate 0 and query
        MPI_Aint sz;
        int disp_unit;
        MPI_Win_allocate_shared(0, sizeof(DATA_TYPE), MPI_INFO_NULL, data_ptr->shared_comm, &data_ptr->u,
                                &data_ptr->win_u);
        MPI_Win_shared_query(data_ptr->win_u, 0, &sz, &disp_unit, &data_ptr->u);
    }
    for (i = 0; i < n; i++) {
        data_ptr->p[i] = 0.0;
        data_ptr->q[i] = 0.0;
        for (j = 0; j < n; j++) {
            data_ptr->u[i * n + j] = (DATA_TYPE)(i + n - j) / n;
            data_ptr->v[i * n + j] = 0.0;
        }
        data_ptr->v[i * n + 0] = SCALAR_VAL(1.0);
        data_ptr->v[i * n + _PB_N - 1] = SCALAR_VAL(1.0);
    }
    data_ptr->v[data_ptr->rank] = -5;
    MPI_Win_sync(data_ptr->win_v);
    MPI_Barrier(data_ptr->shared_comm);
}

void finish_benchmark(void* gen_data_ptr) { (void)gen_data_ptr; }

void free_data(void* gen_data_ptr) {
    mpi_adi_data_t* data_ptr = (mpi_adi_data_t*)gen_data_ptr;
    MPI_Win_free(&data_ptr->win_u);
    MPI_Win_free(&data_ptr->win_v);
    free(gen_data_ptr);
    MPI_Finalize();
}

bool should_print_counter(void* gen_data_ptr) {
    mpi_adi_data_t* data_ptr = (mpi_adi_data_t*)gen_data_ptr;
    return data_ptr->rank == 0;
}

void print_data(int argc, char** argv, int n, void* gen_data_ptr) {
    mpi_adi_data_t* data_ptr = (mpi_adi_data_t*)gen_data_ptr;
    if (data_ptr->rank == 0) {
        default_adi_data_t* adi_data = polybench_alloc_data(1, sizeof(default_adi_data_t));
        memcpy(adi_data->u, data_ptr->u, n * n * sizeof(DATA_TYPE));
        memcpy(adi_data->v, data_ptr->v, n * n * sizeof(DATA_TYPE));
        default_print_data(argc, argv, n, adi_data);
        polybench_free_data(adi_data);
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
                q[j] = (-d * data_ptr->u[j * n + i - 1] +
                        (SCALAR_VAL(1.0) + SCALAR_VAL(2.0) * d) * data_ptr->u[j * n + i] -
                        f * data_ptr->u[j * n + i + 1] - a * q[j - 1]) /
                       (a * p[j - 1] + b);
            }
            for (j = _PB_N - 2; j >= 1; j--) {
                data_ptr->v[i * n + j] = p[j] * data_ptr->v[i * n + j + 1] + q[j];
            }
        }
        MPI_Win_sync(data_ptr->win_v);
        MPI_Barrier(data_ptr->shared_comm);
        // Row Sweep
        for (i = data_ptr->start_i; i < data_ptr->end_i; i++) {
            data_ptr->u[i * n + 0] = SCALAR_VAL(1.0);
            for (j = 1; j < _PB_N - 1; j++) {
                p[j] = -f / (d * p[j - 1] + e);
                q[j] = (-a * data_ptr->v[j * n + i - 1] +
                        (SCALAR_VAL(1.0) + SCALAR_VAL(2.0) * a) * data_ptr->v[j * n + i] -
                        c * data_ptr->v[j * n + i + 1] - d * q[j - 1]) /
                       (d * p[j - 1] + e);
            }
            data_ptr->u[i * n + _PB_N - 1] = SCALAR_VAL(1.0);
            for (j = _PB_N - 2; j >= 1; j--) {
                data_ptr->u[i * n + j] = p[j] * data_ptr->u[i * n + j + 1] + q[j];
            }
        }
        MPI_Win_sync(data_ptr->win_u);
        MPI_Barrier(data_ptr->shared_comm);
    }
}
