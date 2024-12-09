#include <gemver.h>
#include <mpi.h>
#include <polybench.h>

#include "profile.h"

static int world_rank;
static int process_size;
static MPI_Win win;

#define MAX_PROCESSES 16
static int block_start_indx[MAX_PROCESSES];
static int num_elements[MAX_PROCESSES];

static double* process_y;
static double* process_u1;
static double* process_u2;
static double* process_w;
static double* process_A;

void kernel_gemver(int n, DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE POLYBENCH_2D(A, N2, N2, n, n),
                   DATA_TYPE POLYBENCH_1D(u1, N2, n), DATA_TYPE POLYBENCH_1D(v1, N2, n),
                   DATA_TYPE POLYBENCH_1D(u2, N2, n), DATA_TYPE POLYBENCH_1D(v2, N2, n),
                   DATA_TYPE POLYBENCH_1D(w, N2, n), DATA_TYPE POLYBENCH_1D(x, N2, n), DATA_TYPE POLYBENCH_1D(y, N2, n),
                   DATA_TYPE POLYBENCH_1D(z, N2, n)) {
    (void)u1;
    (void)u2;
    (void)y;
    (void)A;

    for (int i = 0; i < process_size; ++i)
        for (int j = 0; j < _PB_N; ++j) {
            process_A[i * _PB_N + j] += process_u1[i] * v1[j] + process_u2[i] * v2[j];
            z[j] += beta * process_A[i * _PB_N + j] * process_y[i];
        }

    MPI_Put(process_A, process_size * _PB_N, MPI_DOUBLE, 0, block_start_indx[world_rank] * _PB_N, process_size * _PB_N,
            MPI_DOUBLE, win);
    MPI_Allreduce(z, x, _PB_N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    for (int i = 0; i < process_size; ++i)
        for (int j = 0; j < _PB_N; ++j) process_w[i] += alpha * process_A[i * _PB_N + j] * x[j];

    MPI_Gatherv(process_w, process_size, MPI_DOUBLE, w, num_elements, block_start_indx, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Win_fence(0, win);
}

void initialise_benchmark(int argc, char** argv, int n, DATA_TYPE* alpha, DATA_TYPE* beta,
                          DATA_TYPE POLYBENCH_2D(A, N2, N2, n, n), DATA_TYPE POLYBENCH_1D(u1, N2, n),
                          DATA_TYPE POLYBENCH_1D(v1, N2, n), DATA_TYPE POLYBENCH_1D(u2, N2, n),
                          DATA_TYPE POLYBENCH_1D(v2, N2, n), DATA_TYPE POLYBENCH_1D(w, N2, n),
                          DATA_TYPE POLYBENCH_1D(x, N2, n), DATA_TYPE POLYBENCH_1D(y, N2, n),
                          DATA_TYPE POLYBENCH_1D(z, N2, n)) {
    (void)u1;
    (void)u2;
    (void)w;
    (void)y;
    MPI_Init(&argc, &argv);
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // v1: N-sized, read-only
    // v2: N-sized, read-only
    // z:  N-sized, read-write
    // x:  N-sized, read-write
    // u1: process-sized, read-only
    // u2: process-sized, read-only
    // y:  process-sized, read-only
    // w:  process-sized, read-write, gather in master
    // A:  process-sized x N-sized, read-write

    *alpha = 1.5;
    *beta = 1.2;

    DATA_TYPE fn = (DATA_TYPE)n;

    for (int i = 0; i < n; i++) {
        v1[i] = ((i + 1) / fn) / 4.0;
        v2[i] = ((i + 1) / fn) / 6.0;
        if (world_rank == 0)
            z[i] = ((i + 1) / fn) / 9.0;
        else
            z[i] = 0.0;
        x[i] = 0.0;
        if (world_rank == 0)
            for (int j = 0; j < n; j++) A[i][j] = (DATA_TYPE)(i * j % n) / n;
    }

    int block = n / world_size;
    int remaining = n % world_size;
    for (int i = 0; i < world_size; i++) {
        if (i < world_size - 1)
            num_elements[i] = block;
        else
            num_elements[i] = block + remaining;
        block_start_indx[i] = i * block;
    }

    process_size = num_elements[world_rank];

    process_y = polybench_alloc_data(process_size, sizeof(DATA_TYPE));
    process_u1 = polybench_alloc_data(process_size, sizeof(DATA_TYPE));
    process_u2 = polybench_alloc_data(process_size, sizeof(DATA_TYPE));
    process_w = polybench_alloc_data(process_size, sizeof(DATA_TYPE));
    process_A = polybench_alloc_data(n * process_size, sizeof(DATA_TYPE));

    for (int i = 0; i < process_size; i++) {
        int i_value = i + block_start_indx[world_rank];
        process_u1[i] = i_value;
        process_u2[i] = ((i_value + 1) / fn) / 2.0;
        process_y[i] = ((i_value + 1) / fn) / 8.0;
        process_w[i] = 0.0;
        for (int j = 0; j < n; j++) {
            process_A[i * n + j] = (DATA_TYPE)(i_value * j % n) / n;
        }
    }

    MPI_Aint winsize = (world_rank == 0) ? (n * n * sizeof(DATA_TYPE)) : 0;
    MPI_Win_create(A, winsize, sizeof(DATA_TYPE), MPI_INFO_NULL, MPI_COMM_WORLD, &win);
    MPI_Win_fence(0, win);
    MPI_Barrier(MPI_COMM_WORLD);
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

    MPI_Win_free(&win);
    MPI_Finalize();

    if (world_rank != 0) exit(0);
}
