/* Include polybench common header. */
#include <polybench.h>

/* Include mpi header. */
#include <mpi.h>

/* Include benchmark-specific header. */
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "gemver.h"

int world_size;
int world_rank;

/*
VERSION 2
central process distributes the work in blocks equally on all others and collects results
non blocking communication
non blocking allows computations and communication to overlap, which generally leads to improved performance
*/

void initialise_benchmark(int argc, char **argv, int n, DATA_TYPE *alpha, DATA_TYPE *beta,
                          DATA_TYPE POLYBENCH_2D(A, N2, N2, n, n), DATA_TYPE POLYBENCH_1D(u1, N2, n),
                          DATA_TYPE POLYBENCH_1D(v1, N2, n), DATA_TYPE POLYBENCH_1D(u2, N2, n),
                          DATA_TYPE POLYBENCH_1D(v2, N2, n), DATA_TYPE POLYBENCH_1D(w, N2, n),
                          DATA_TYPE POLYBENCH_1D(x, N2, n), DATA_TYPE POLYBENCH_1D(y, N2, n),
                          DATA_TYPE POLYBENCH_1D(z, N2, n)) {
    // Initialize the MPI environment
    MPI_Init(&argc, &argv);

    // Get the number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

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

    // Finalize the MPI environment.
    MPI_Finalize();

    if (world_rank != 0) {
        /* Be clean. */
        POLYBENCH_FREE_ARRAY(A);
        POLYBENCH_FREE_ARRAY(u1);
        POLYBENCH_FREE_ARRAY(v1);
        POLYBENCH_FREE_ARRAY(u2);
        POLYBENCH_FREE_ARRAY(v2);
        POLYBENCH_FREE_ARRAY(w);
        POLYBENCH_FREE_ARRAY(x);
        POLYBENCH_FREE_ARRAY(y);
        POLYBENCH_FREE_ARRAY(z);
        exit(0);
    }
}

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
void kernel_gemver(int n, DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE POLYBENCH_2D(A, N2, N2, n, n),
                   DATA_TYPE POLYBENCH_1D(u1, N2, n), DATA_TYPE POLYBENCH_1D(v1, N2, n),
                   DATA_TYPE POLYBENCH_1D(u2, N2, n), DATA_TYPE POLYBENCH_1D(v2, N2, n),
                   DATA_TYPE POLYBENCH_1D(w, N2, n), DATA_TYPE POLYBENCH_1D(x, N2, n), DATA_TYPE POLYBENCH_1D(y, N2, n),
                   DATA_TYPE POLYBENCH_1D(z, N2, n)) {
    // split into blocks to distribute over the processes
    int block = n / world_size;
    int remaining = n % world_size;

    int *block_start_indx = (int *)malloc(world_size * sizeof(int));
    int *num_elements = (int *)malloc(world_size * sizeof(int));
    for (int i = 0; i < world_size; i++) {
        if (i < world_size - 1) {
            num_elements[i] = block;
        } else {
            num_elements[i] =
                block + remaining;  // last process also gets the remaining elements that dont fit in the vlock split
        }
        block_start_indx[i] = i * block;
    }

    int process_size = num_elements[world_rank];

    POLYBENCH_1D_ARRAY_DECL(process_u1, DATA_TYPE, process_size, process_size);
    POLYBENCH_1D_ARRAY_DECL(process_u2, DATA_TYPE, process_size, process_size);

    MPI_Request send_u1, send_u2;

    MPI_Iscatterv(u1, num_elements, block_start_indx, MPI_DOUBLE, process_u1, process_size, MPI_DOUBLE, 0,
                  MPI_COMM_WORLD, &send_u1);
    MPI_Iscatterv(u2, num_elements, block_start_indx, MPI_DOUBLE, process_u2, process_size, MPI_DOUBLE, 0,
                  MPI_COMM_WORLD, &send_u2);

    // do the block split for A as well
    int *block_start_indx_matrix = (int *)malloc(world_size * sizeof(int));
    int *num_elements_matrix = (int *)malloc(world_size * sizeof(int));
    for (int i = 0; i < world_size; i++) {
        if (i < world_size - 1) {
            num_elements_matrix[i] = block * n;
        } else {
            num_elements_matrix[i] =
                (block + remaining) *
                n;  // last process also gets the remaining elements that dont fit in the vlock split
        }
        block_start_indx_matrix[i] = i * block * n;
    }

    POLYBENCH_2D_ARRAY_DECL(process_A, DATA_TYPE, process_size, N2, process_size, n);
    MPI_Request send_A;
    MPI_Request rec_A;

    MPI_Iscatterv(A, num_elements_matrix, block_start_indx_matrix, MPI_DOUBLE, process_A, process_size * n, MPI_DOUBLE,
                  0, MPI_COMM_WORLD, &send_A);

    MPI_Bcast(v1, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(v2, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Wait(&send_u1, MPI_STATUS_IGNORE);
    MPI_Wait(&send_u2, MPI_STATUS_IGNORE);
    MPI_Wait(&send_A, MPI_STATUS_IGNORE);

    // compute A
    for (int i = 0; i < process_size; i++) {
        for (int j = 0; j < n; j++) {
            (*process_A)[i][j] += (*process_u1)[i] * v1[j] + (*process_u2)[i] * v2[j];
        }
    }
    // send results to first process
    MPI_Igatherv(process_A, process_size * n, MPI_DOUBLE, A, num_elements_matrix, block_start_indx_matrix, MPI_DOUBLE,
                 0, MPI_COMM_WORLD, &rec_A);

    /*
    calculate  x = x + b * A^T * y + z
    */

    MPI_Datatype columns, new_columns;  // used to send tge blocks of the 2d array

    MPI_Type_vector(n, 1, n, MPI_DOUBLE, &columns);

    MPI_Type_commit(&columns);

    MPI_Type_create_resized(columns, 0, 1 * sizeof(DATA_TYPE), &new_columns);
    MPI_Type_commit(&new_columns);

    int *block_start_indx_matrix_T = (int *)malloc(world_size * sizeof(int));
    int *num_elements_matrix_T = (int *)malloc(world_size * sizeof(int));
    int block_start = 0;
    for (int i = 0; i < world_size; i++) {
        if (i < world_size - 1) {
            num_elements_matrix_T[i] = block;
        } else {
            num_elements_matrix_T[i] = (block + remaining);
        }
        block_start_indx_matrix_T[i] = block_start;
        block_start += num_elements_matrix_T[i];
    }

    MPI_Request send_A_T, send_z, send_x;
    MPI_Wait(&rec_A, MPI_STATUS_IGNORE);
    // transpose A by sending columnsas rows
    POLYBENCH_2D_ARRAY_DECL(process_A_T, DATA_TYPE, process_size, N2, process_size, n);
    MPI_Iscatterv(A, num_elements_matrix_T, block_start_indx_matrix_T, new_columns, process_A_T, process_size * n,
                  MPI_DOUBLE, 0, MPI_COMM_WORLD, &send_A_T);

    POLYBENCH_1D_ARRAY_DECL(process_z, DATA_TYPE, process_size, process_size);
    POLYBENCH_1D_ARRAY_DECL(process_x, DATA_TYPE, process_size, process_size);
    MPI_Iscatterv(z, num_elements, block_start_indx, MPI_DOUBLE, process_z, process_size, MPI_DOUBLE, 0, MPI_COMM_WORLD,
                  &send_z);
    MPI_Iscatterv(x, num_elements, block_start_indx, MPI_DOUBLE, process_x, process_size, MPI_DOUBLE, 0, MPI_COMM_WORLD,
                  &send_x);

    // send y,b everywhere
    MPI_Bcast(y, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&beta, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Wait(&send_A_T, MPI_STATUS_IGNORE);
    MPI_Wait(&send_z, MPI_STATUS_IGNORE);
    MPI_Wait(&send_x, MPI_STATUS_IGNORE);

    // x = b * A^T * y + z
    for (int i = 0; i < process_size; i++) {
        for (int j = 0; j < n; j++) {
            (*process_x)[i] = (*process_x)[i] + beta * (*process_A_T)[i][j] * y[j];
        }
        (*process_x)[i] += (*process_z)[i];
    }
    MPI_Request rec_x;
    // send results to first process
    MPI_Igatherv(process_x, process_size, MPI_DOUBLE, x, num_elements, block_start_indx, MPI_DOUBLE, 0, MPI_COMM_WORLD,
                 &rec_x);

    /*
    calculate  w = a * A * x
    */
    MPI_Request send_A_res, send_x_res, send_w;

    POLYBENCH_2D_ARRAY_DECL(process_A_res, DATA_TYPE, process_size, N2, process_size, n);
    MPI_Iscatterv(A, num_elements_matrix, block_start_indx_matrix, MPI_DOUBLE, process_A_res, process_size * n,
                  MPI_DOUBLE, 0, MPI_COMM_WORLD, &send_A_res);

    // send x everywhere
    MPI_Wait(&rec_x, MPI_STATUS_IGNORE);
    MPI_Ibcast(x, n, MPI_DOUBLE, 0, MPI_COMM_WORLD, &send_x_res);
    MPI_Bcast(&alpha, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    POLYBENCH_1D_ARRAY_DECL(process_w, DATA_TYPE, process_size, process_size);
    MPI_Iscatterv(w, num_elements, block_start_indx, MPI_DOUBLE, process_w, process_size, MPI_DOUBLE, 0, MPI_COMM_WORLD,
                  &send_w);

    MPI_Wait(&send_A_res, MPI_STATUS_IGNORE);
    MPI_Wait(&send_x_res, MPI_STATUS_IGNORE);
    MPI_Wait(&send_w, MPI_STATUS_IGNORE);

    // w = a * A * x
    for (int i = 0; i < process_size; i++) {
        for (int j = 0; j < n; j++) {
            (*process_w)[i] = (*process_w)[i] + alpha * (*process_A_res)[i][j] * x[j];
        }
    }

    MPI_Request rec_w_res;
    // send results to first process
    MPI_Igatherv(process_w, process_size, MPI_DOUBLE, w, num_elements, block_start_indx, MPI_DOUBLE, 0, MPI_COMM_WORLD,
                 &rec_w_res);
    MPI_Wait(&rec_w_res, MPI_STATUS_IGNORE);

    // cleanup
    POLYBENCH_FREE_ARRAY(num_elements);
    POLYBENCH_FREE_ARRAY(num_elements_matrix);
    POLYBENCH_FREE_ARRAY(block_start_indx);
    POLYBENCH_FREE_ARRAY(block_start_indx_matrix);
    POLYBENCH_FREE_ARRAY(process_u1);
    POLYBENCH_FREE_ARRAY(process_u2);
    POLYBENCH_FREE_ARRAY(process_A);
    POLYBENCH_FREE_ARRAY(process_A_T);
    POLYBENCH_FREE_ARRAY(process_z);
    POLYBENCH_FREE_ARRAY(process_x);
    POLYBENCH_FREE_ARRAY(process_w);
    POLYBENCH_FREE_ARRAY(process_A_res);
}
