/* Include polybench common header. */
#include "polybench.h"

/* Include benchmark-specific header. */
#include <assert.h>
#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "adi.h"

#define MAX_PROCESSES 16

typedef struct {
    DATA_TYPE POLYBENCH_2D(u, N2, N2, n, n);
    DATA_TYPE POLYBENCH_2D(v, N2, N2, n, n);
    DATA_TYPE POLYBENCH_1D(p, N2, n);
    DATA_TYPE POLYBENCH_1D(q, N2, n);
    DATA_TYPE** send_buf;
    DATA_TYPE** recv_buf;
    int tsteps;
    int n;
    int argc;
    char** argv;
    int rank, world_size;
    int start_i, end_i;
    int bounds_per_process[MAX_PROCESSES][2];
    int send_count_to_process[MAX_PROCESSES];
    int recv_count_from_process[MAX_PROCESSES];
    int elems_per_process[MAX_PROCESSES];
    int displs[MAX_PROCESSES];
} mpi_adi_data_t;

void initialise_benchmark(int argc, char** argv, int tsteps, int n, void** gen_data_ptr) {
    *gen_data_ptr = polybench_alloc_data(1, sizeof(mpi_adi_data_t));
    mpi_adi_data_t* data_ptr = (mpi_adi_data_t*)(*gen_data_ptr);

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &data_ptr->rank);
    MPI_Comm_size(MPI_COMM_WORLD, &data_ptr->world_size);

    int i = 0, j;
    data_ptr->tsteps = tsteps;
    data_ptr->n = n;
    data_ptr->argc = argc;
    data_ptr->argv = argv;

    for (i = 0; i < n; i++) {
        data_ptr->p[i] = 0.0;
        data_ptr->q[i] = 0.0;
        for (j = 0; j < n; j++) {
            data_ptr->u[i][j] = (DATA_TYPE)(i + n - j) / n;
            data_ptr->v[i][j] = 0.0;
        }
    }
    int cnt = 0;

    data_ptr->send_buf = polybench_alloc_data(data_ptr->world_size, sizeof(DATA_TYPE*));
    data_ptr->recv_buf = polybench_alloc_data(data_ptr->world_size, sizeof(DATA_TYPE*));
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
    data_ptr->elems_per_process[data_ptr->world_size - 1] =
        (data_ptr->bounds_per_process[data_ptr->world_size - 1][1] -
         data_ptr->bounds_per_process[data_ptr->world_size - 1][0] + 1) *
        _PB_N;
    data_ptr->displs[data_ptr->world_size - 1] = cnt;

    data_ptr->bounds_per_process[0][0] = 1;

    data_ptr->start_i = data_ptr->bounds_per_process[data_ptr->rank][0];
    data_ptr->end_i = data_ptr->bounds_per_process[data_ptr->rank][1];

    // process_i to process_j sends (end_i - start_i) * (end_j - start_j + 1) elements
    // My process sends to p_j (data_ptr->end_i - data_ptr->start_i) * (data_ptr->bounds_per_process[j][1] -
    // data_ptr->bounds_per_process[j][0] + 1) elements My process receives from p_j (data_ptr->bounds_per_process[j][1]
    // - data_ptr->bounds_per_process[j][0]) * (data_ptr->end_i - data_ptr->start_i + 1) elements
    for (i = 0; i < data_ptr->world_size; i++) {
        if (i == data_ptr->rank) continue;
        int scount = data_ptr->bounds_per_process[i][1] - data_ptr->bounds_per_process[i][0];
        int rcount = data_ptr->end_i - data_ptr->start_i;
        data_ptr->send_count_to_process[i] = (rcount) * (scount + 2);
        data_ptr->recv_count_from_process[i] = (scount) * (rcount + 2);

        data_ptr->send_buf[i] = polybench_alloc_data(data_ptr->send_count_to_process[i], sizeof(DATA_TYPE));
        data_ptr->recv_buf[i] = polybench_alloc_data(data_ptr->recv_count_from_process[i], sizeof(DATA_TYPE));
    }

    for (i = 0; i < n; i++) {
        data_ptr->v[i][0] = SCALAR_VAL(1.0);
        data_ptr->v[i][_PB_N - 1] = SCALAR_VAL(1.0);
    }
}

void finish_benchmark(void* gen_data_ptr) {
    mpi_adi_data_t* data_ptr = (mpi_adi_data_t*)gen_data_ptr;

    // Sync here the array u only once again!
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, data_ptr->u, data_ptr->elems_per_process, data_ptr->displs,
                   MPI_DOUBLE, MPI_COMM_WORLD);

    for (int i = 0; i < data_ptr->world_size; i++) {
        if (i == data_ptr->rank) continue;
        polybench_free_data(data_ptr->send_buf[i]);
        polybench_free_data(data_ptr->recv_buf[i]);
    }

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
    int world_size = data_ptr->world_size;
    int rank = data_ptr->rank;
    int tsteps = data_ptr->tsteps;
    int start_i = data_ptr->start_i;
    int end_i = data_ptr->end_i;
    int needed_start, needed_end;

    int t, i, j, pi, pcount;
    MPI_Request* recv_requests_v = (MPI_Request*)polybench_alloc_data(world_size - 1, sizeof(MPI_Request));
    MPI_Request* send_requests_v = (MPI_Request*)polybench_alloc_data(world_size - 1, sizeof(MPI_Request));
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
            for (pi = world_size - 1; pi >= 0; pi--) {
                if (pi == rank) continue;
                needed_end = data_ptr->bounds_per_process[pi][1] + 1;
                needed_start = data_ptr->bounds_per_process[pi][0] - 1;
                pcount = needed_end - needed_start;
                memcpy(&data_ptr->send_buf[pi][(i - start_i) * pcount], &data_ptr->v[i][needed_start],
                       pcount * sizeof(DATA_TYPE));

                if (!((i - start_i) * pcount + pcount <= data_ptr->send_count_to_process[pi])) {
                    printf("Rank %d: i = %d start_i = %d pcount = %d send_count = %d\n", rank, i, start_i, pcount,
                           data_ptr->send_count_to_process[pi]);
                    exit(1);
                }
            }
        }
        // Sync array v among all processes here!
        pcount = 0;
        for (pi = 0; pi < world_size; pi++) {
            if (pi == rank) continue;
            MPI_Isend(data_ptr->send_buf[pi], data_ptr->send_count_to_process[pi], MPI_DOUBLE, pi, 0, MPI_COMM_WORLD,
                      &send_requests_v[pcount]);
            MPI_Irecv(data_ptr->recv_buf[pi], data_ptr->recv_count_from_process[pi], MPI_DOUBLE, pi, 0, MPI_COMM_WORLD,
                      &recv_requests_v[pcount++]);
        }
        MPI_Waitall(world_size - 1, send_requests_v, MPI_STATUSES_IGNORE);
        MPI_Waitall(world_size - 1, recv_requests_v, MPI_STATUSES_IGNORE);

        pcount = end_i - start_i + 2;
        for (pi = 0; pi < world_size; pi++) {
            if (pi == rank) continue;
            //                bool flag = true;
            //                while (flag)
            //                    sleep(2);
            needed_start = data_ptr->bounds_per_process[pi][0];
            needed_end = data_ptr->bounds_per_process[pi][1];
            for (i = needed_start; i < needed_end; i++) {
                memcpy(&data_ptr->v[i][start_i - 1], &(data_ptr->recv_buf[pi][(i - needed_start) * pcount]),
                       pcount * sizeof(DATA_TYPE));
                assert(start_i - 1 >= 0);
                assert((i - needed_start) * pcount + pcount <= data_ptr->recv_count_from_process[pi]);
            }
        }
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
            for (pi = world_size - 1; pi >= 0; pi--) {
                if (pi == rank) continue;
                needed_end = data_ptr->bounds_per_process[pi][1] + 1;
                needed_start = data_ptr->bounds_per_process[pi][0] - 1;
                pcount = needed_end - needed_start;
                memcpy(&data_ptr->send_buf[pi][(i - start_i) * pcount], &data_ptr->u[i][needed_start],
                       pcount * sizeof(DATA_TYPE));
                assert((i - start_i) * pcount + pcount <= data_ptr->send_count_to_process[pi]);
            }
        }
        // Sync array u among all processes here!
        pcount = 0;
        for (pi = 0; pi < world_size; pi++) {
            if (pi == rank) continue;
            MPI_Isend(data_ptr->send_buf[pi], data_ptr->send_count_to_process[pi], MPI_DOUBLE, pi, 0, MPI_COMM_WORLD,
                      &send_requests_v[pcount]);
            MPI_Irecv(data_ptr->recv_buf[pi], data_ptr->recv_count_from_process[pi], MPI_DOUBLE, pi, 0, MPI_COMM_WORLD,
                      &recv_requests_v[pcount++]);
        }
        MPI_Waitall(world_size - 1, send_requests_v, MPI_STATUSES_IGNORE);
        MPI_Waitall(world_size - 1, recv_requests_v, MPI_STATUSES_IGNORE);

        pcount = end_i - start_i + 2;
        for (pi = 0; pi < world_size; pi++) {
            if (pi == rank) continue;
            needed_start = data_ptr->bounds_per_process[pi][0];
            needed_end = data_ptr->bounds_per_process[pi][1];
            for (i = needed_start; i < needed_end; i++) {
                memcpy(&data_ptr->u[i][start_i - 1], &(data_ptr->recv_buf[pi][(i - needed_start) * pcount]),
                       pcount * sizeof(DATA_TYPE));
                assert(start_i - 1 >= 0);
                assert((i - needed_start) * pcount + pcount <= data_ptr->recv_count_from_process[pi]);
            }
        }
    }
    polybench_free_data(recv_requests_v);
    polybench_free_data(send_requests_v);
}
