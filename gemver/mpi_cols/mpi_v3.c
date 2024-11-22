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

int block;
int remaining;

int process_size;

#define MAX_PROCESSES 16

static int block_start_indx[MAX_PROCESSES];
static int num_elements[MAX_PROCESSES];
static int block_start_indx_matrix[MAX_PROCESSES];
static int num_elements_matrix[MAX_PROCESSES];

MPI_Datatype columns, new_columns;  // used to send tge blocks of the 2d array
MPI_Datatype columns_to_get, new_columns_to_get;

double *process_z, *process_v1, *process_v2, *process_x, *process_w;
double *process_A;

/*
VERSION 3
central process distributes the work in blocks equally on all others and collects results
this time we split into columns not rows
so we have to do less scatters
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
        y[i] = ((i + 1) / fn) / 8.0;
        x[i] = 0.0;
        w[i] = 0.0;
        for (j = 0; j < n; j++) A[i][j] = (DATA_TYPE)(i * j % n) / n;
    }

    // split into blocks to distribute over the processes
    block = n / world_size;
    remaining = n % world_size;

    for (int i = 0; i < world_size; i++) {
        if (i < world_size - 1) {
            num_elements[i] = block;
        } else {
            num_elements[i] =
                block + remaining;  // last process also gets the remaining elements that dont fit in the vlock split
        }
        block_start_indx[i] = i * block;
    }

    process_size = num_elements[world_rank];

    int block_start = 0;
    for (int i = 0; i < world_size; i++) {
        if (i < world_size - 1) {
            num_elements_matrix[i] = block;
        } else {
            num_elements_matrix[i] = (block + remaining);
        }
        block_start_indx_matrix[i] = block_start;
        block_start += num_elements_matrix[i];
    }

    MPI_Type_vector(n, 1, process_size, MPI_DOUBLE, &columns);
    MPI_Type_commit(&columns);
    MPI_Type_create_resized(columns, 0, 1 * sizeof(DATA_TYPE), &new_columns);
    MPI_Type_commit(&new_columns);

    MPI_Type_vector(n, 1, n, MPI_DOUBLE, &columns_to_get);

    MPI_Type_commit(&columns_to_get);
    MPI_Type_create_resized(columns_to_get, 0, sizeof(DATA_TYPE), &new_columns_to_get);
    MPI_Type_commit(&new_columns_to_get);

    process_z = POLYBENCH_ALLOC_1D_ARRAY(process_size, DATA_TYPE);
    process_v1 = POLYBENCH_ALLOC_1D_ARRAY(process_size, DATA_TYPE);
    process_v2 = POLYBENCH_ALLOC_1D_ARRAY(process_size, DATA_TYPE);
    process_x = POLYBENCH_ALLOC_1D_ARRAY(process_size, DATA_TYPE);
    process_w = POLYBENCH_ALLOC_1D_ARRAY(N2, DATA_TYPE);
    process_A = POLYBENCH_ALLOC_1D_ARRAY(N2 * process_size, DATA_TYPE);
    for (int i = 0; i < process_size; i++) {
        int i_value = i + block_start_indx[world_rank];
        process_z[i] = ((i_value + 1) / fn) / 9.0;
        process_v1[i] = ((i_value + 1) / fn) / 4.0;
        process_v2[i] = ((i_value + 1) / fn) / 6.0;
        process_x[i] = 0.0;
    }

    for (int i = 0; i < n; i++) {
        process_w[i] = 0.0;
        for (int j = 0; j < process_size; j++) {
            process_A[i * process_size + j] = (DATA_TYPE)(i * (j + block_start_indx[world_rank]) % n) / n;
        }
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
        // POLYBENCH_FREE_ARRAY(A);
        // POLYBENCH_FREE_ARRAY(u1);
        // POLYBENCH_FREE_ARRAY(v1);
        // POLYBENCH_FREE_ARRAY(u2);
        // POLYBENCH_FREE_ARRAY(v2);
        // POLYBENCH_FREE_ARRAY(w);
        // POLYBENCH_FREE_ARRAY(x);
        // POLYBENCH_FREE_ARRAY(y);
        // POLYBENCH_FREE_ARRAY(z);

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
    /*
    calculate  A = A + u1*v1 + u2*v2
    */
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < process_size; j++) {
            process_A[i * process_size + j] += u1[i] * process_v1[j] + u2[i] * process_v2[j];
        }
    }

    MPI_Gatherv(process_A, process_size, new_columns, A, num_elements_matrix, block_start_indx_matrix,
                new_columns_to_get, 0, MPI_COMM_WORLD);

    /*
    calculate  x = x + b * A^T * y + z
    */

    for (int i = 0; i < n; i++)  // change indixes to transpose A
    {
        for (int j = 0; j < process_size; j++) {
            process_x[j] += beta * process_A[i * process_size + j] * y[i];
        }
    }

    for (int j = 0; j < process_size; j++)  // careful need to add z outside of i loop!
    {
        process_x[j] += process_z[j];
    }

    // send results to first process
    MPI_Gatherv(process_x, process_size, MPI_DOUBLE, x, num_elements, block_start_indx, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /*
    calculate  w += a * A * x
    */

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < process_size; j++) {
            process_w[i] += alpha * process_A[i * process_size + j] * process_x[j];
        }
    }
    // combine all process w to the main w in process 0
    MPI_Reduce(process_w, w, n, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    POLYBENCH_FREE_ARRAY(process_v2);
    POLYBENCH_FREE_ARRAY(process_v1);
    POLYBENCH_FREE_ARRAY(process_w);
    POLYBENCH_FREE_ARRAY(process_x);
    POLYBENCH_FREE_ARRAY(process_z);
    POLYBENCH_FREE_ARRAY(process_A);
}
