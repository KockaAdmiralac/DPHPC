#include "gemver.h"


/* Array initialization. */
static void init_array(int n, DATA_TYPE *alpha, DATA_TYPE *beta, DATA_TYPE POLYBENCH_2D(A, N, N, n, n),
                       DATA_TYPE POLYBENCH_1D(u1, N, n), DATA_TYPE POLYBENCH_1D(v1, N, n),
                       DATA_TYPE POLYBENCH_1D(u2, N, n), DATA_TYPE POLYBENCH_1D(v2, N, n),
                       DATA_TYPE POLYBENCH_1D(w, N, n), DATA_TYPE POLYBENCH_1D(x, N, n),
                       DATA_TYPE POLYBENCH_1D(y, N, n), DATA_TYPE POLYBENCH_1D(z, N, n)) {
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

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(int n, DATA_TYPE POLYBENCH_1D(w, N, n)) {
    int i;

    POLYBENCH_DUMP_START;
    POLYBENCH_DUMP_BEGIN("w");
    for (i = 0; i < n; i++) {
        if (i % 20 == 0) fprintf(POLYBENCH_DUMP_TARGET, "\n");
        fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, w[i]);
    }
    POLYBENCH_DUMP_END("w");
    POLYBENCH_DUMP_FINISH;
}

//central process distributes the work in blocks equally on all others and collects results
void gemver(int argc, char** argv) {
    // Initialize the MPI environment
    MPI_Init(&argc, &argv);

    /* Retrieve problem size. */
    int n = N;

    /* Variable declaration/allocation. */
    DATA_TYPE alpha;
    DATA_TYPE beta;
    
    POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);
    POLYBENCH_1D_ARRAY_DECL(u1, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(v1, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(u2, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(v2, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(w, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(x, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(y, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(z, DATA_TYPE, N, n);

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    //assign mpi memory for variables

    //for all processes
    MPI_Alloc_mem(n * sizeof(DATA_TYPE), MPI_INFO_NULL, &v1);
    MPI_Alloc_mem(n * sizeof(DATA_TYPE), MPI_INFO_NULL, &v2);
    MPI_Alloc_mem(n * sizeof(DATA_TYPE), MPI_INFO_NULL, &y);
    MPI_Alloc_mem(n * sizeof(DATA_TYPE), MPI_INFO_NULL, &x);

    //for first process only
    if (world_rank == 0)
    {
        MPI_Alloc_mem(n * n * sizeof(DATA_TYPE), MPI_INFO_NULL, &A);
        MPI_Alloc_mem(n * sizeof(DATA_TYPE), MPI_INFO_NULL, &u1);
        MPI_Alloc_mem(n * sizeof(DATA_TYPE), MPI_INFO_NULL, &u2);
        MPI_Alloc_mem(n * sizeof(DATA_TYPE), MPI_INFO_NULL, &w);
        MPI_Alloc_mem(n * sizeof(DATA_TYPE), MPI_INFO_NULL, &z);

        // Initialize array(s) only for process 0
        init_array(n, &alpha, &beta, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(u1), POLYBENCH_ARRAY(v1), POLYBENCH_ARRAY(u2),
               POLYBENCH_ARRAY(v2), POLYBENCH_ARRAY(w), POLYBENCH_ARRAY(x), POLYBENCH_ARRAY(y), POLYBENCH_ARRAY(z));
    }

    /*
    calculate  A = A + u1*v1 + u2*v2
    */

   if (world_rank == 0){
    polybench_start_instruments;
   }

    //split into blocks to distribute over the processes
    int block = n / world_size;
    int remaining = n % world_size;

    int *block_start_indx = (int *)malloc(world_size * sizeof(int));  
    int *num_elements = (int *)malloc(world_size * sizeof(int)); 
    for (int i = 0; i < world_size; i++)
    {
        if (i < world_size - 1)
        {
            num_elements[i] = block;
        }
        else
        {
            num_elements[i] = block + remaining;//last process also gets the remaining elements that dont fit in the vlock split
        }
        block_start_indx[i] = i * block;
    }

    int process_size = num_elements[world_rank];

    POLYBENCH_1D_ARRAY_DECL(process_u1, DATA_TYPE, process_size, process_size);
    POLYBENCH_1D_ARRAY_DECL(process_u2, DATA_TYPE, process_size, process_size);

    MPI_Scatterv(u1, num_elements, block_start_indx, MPI_DOUBLE, process_u1, process_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(u2, num_elements, block_start_indx, MPI_DOUBLE, process_u2, process_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    //do the block split for A as well
    int *block_start_indx_matrix = (int *)malloc(world_size * sizeof(int));
    int *num_elements_matrix = (int *)malloc(world_size * sizeof(int));
    for (int i = 0; i < world_size; i++)
    {
        if (i < world_size - 1)
        {
            num_elements_matrix[i] = block * n;
        }
        else
        {
            num_elements_matrix[i] = (block + remaining) * n;//last process also gets the remaining elements that dont fit in the vlock split
        }
        block_start_indx_matrix[i] = i * block * n;
    }

    POLYBENCH_2D_ARRAY_DECL(process_A, DATA_TYPE, process_size, N, process_size, n);
    //double *process_A = (double *)malloc(process_size * n * sizeof(double));
    MPI_Scatterv(A, num_elements_matrix, block_start_indx_matrix, MPI_DOUBLE, process_A, process_size * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(v1, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(v2, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    //compute A
    for (int i = 0; i < process_size; i++)
    {
        for (int j = 0; j < n; j++)
        {
            (*process_A)[i][j] += (*process_u1)[i] * (*v1)[j] + (*process_u2)[i] * (*v2)[j];
        }
    }
    //send results to first process
    MPI_Gatherv(process_A, process_size * n, MPI_DOUBLE, A, num_elements_matrix, block_start_indx_matrix, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /*
    calculate  x = x + b * A^T * y + z
    */

    MPI_Datatype columns, new_columns;//used to send tge blocks of the 2d array

    MPI_Type_vector(n, 1, n, MPI_DOUBLE, &columns);

    MPI_Type_commit(&columns);

    MPI_Type_create_resized(columns, 0, 1 * sizeof(DATA_TYPE), &new_columns); 
    MPI_Type_commit(&new_columns);

    int *displsMatrixTranspose = (int *)malloc(world_size * sizeof(int));  // stores start index of each block
    int *scountsMatrixTranspose = (int *)malloc(world_size * sizeof(int)); // stores number of elements in each block
    int block_start = 0;
    for (int i = 0; i < world_size; i++)
    {
        if (i < world_size - 1)
        {
            scountsMatrixTranspose[i] = block;
        }
        else
        {
            scountsMatrixTranspose[i] = (block + remaining);
        }
        displsMatrixTranspose[i] = block_start;
        block_start += scountsMatrixTranspose[i];
    }

    //transpose A by sending columnsas rows
    POLYBENCH_2D_ARRAY_DECL(process_A_T, DATA_TYPE, process_size, N, process_size, n);
    MPI_Scatterv(A, scountsMatrixTranspose, displsMatrixTranspose, new_columns, process_A_T, process_size * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    POLYBENCH_1D_ARRAY_DECL(process_z, DATA_TYPE, process_size, process_size);
    POLYBENCH_1D_ARRAY_DECL(process_x, DATA_TYPE, process_size, process_size);
    MPI_Scatterv(z, num_elements, block_start_indx, MPI_DOUBLE, process_z, process_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(x, num_elements, block_start_indx, MPI_DOUBLE, process_x, process_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    //send y,b everywhere
    MPI_Bcast(y, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&beta, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    //x = b * A^T * y + z
    for (int i = 0; i < process_size; i++)
    {
        for (int j = 0; j < n; j++)
        {
            (*process_x)[i] = (*process_x)[i] + beta * (*process_A_T)[i][j] * (*y)[j];
        }
        (*process_x)[i] += (*process_z)[i];
    }

    //send results to first process
    MPI_Gatherv(process_x, process_size, MPI_DOUBLE, x, num_elements, block_start_indx, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /*
    calculate  w = a * A * x
    */

    POLYBENCH_2D_ARRAY_DECL(process_A_res, DATA_TYPE, process_size, N, process_size, n);
    MPI_Scatterv(A, num_elements_matrix, block_start_indx_matrix, MPI_DOUBLE, process_A_res, process_size * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    //send x everywhere
    MPI_Bcast(x, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&alpha, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    POLYBENCH_1D_ARRAY_DECL(process_w, DATA_TYPE, process_size, process_size);
    MPI_Scatterv(w, num_elements, block_start_indx, MPI_DOUBLE, process_w, process_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    //w = a * A * x
    for (int i = 0; i < process_size; i++)
    {
        for (int j = 0; j < n; j++)
        {
            (*process_w)[i] = (*process_w)[i] + alpha * (*process_A_res)[i][j] * (*x)[j];
        }
    }
        

    //send results to first process
    MPI_Gatherv(process_w, process_size, MPI_DOUBLE, w, num_elements, block_start_indx, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (world_rank == 0){
        /* Stop and print timer. */
        polybench_stop_instruments;
        polybench_print_instruments;

        /* Prevent dead-code elimination. All live-out data must be printed
        by the function call in argument. */
        polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(w)));
    }

    //cleanup
    MPI_Free_mem(num_elements);
    POLYBENCH_FREE_ARRAY(num_elements);
    MPI_Free_mem(num_elements_matrix);
    POLYBENCH_FREE_ARRAY(num_elements_matrix);
    MPI_Free_mem(block_start_indx);
    POLYBENCH_FREE_ARRAY(block_start_indx);
    MPI_Free_mem(block_start_indx_matrix);
    POLYBENCH_FREE_ARRAY(block_start_indx_matrix);
    MPI_Free_mem(process_u1);
    POLYBENCH_FREE_ARRAY(process_u1);
    MPI_Free_mem(process_u2);
    POLYBENCH_FREE_ARRAY(process_u2);
    MPI_Free_mem(process_A);
    POLYBENCH_FREE_ARRAY(process_A);
    MPI_Free_mem(process_A_T);
    POLYBENCH_FREE_ARRAY(process_A_T);
    MPI_Free_mem(process_z);
    POLYBENCH_FREE_ARRAY(process_z);
    MPI_Free_mem(process_x);
    POLYBENCH_FREE_ARRAY(process_x);
    MPI_Free_mem(process_w);
    POLYBENCH_FREE_ARRAY(process_w);
    MPI_Free_mem(process_A_res);
    POLYBENCH_FREE_ARRAY(process_A_res);
    MPI_Free_mem(v1); 
    POLYBENCH_FREE_ARRAY(v1);
    MPI_Free_mem(v2);
    POLYBENCH_FREE_ARRAY(v2);
    MPI_Free_mem(x);
    POLYBENCH_FREE_ARRAY(x);
    MPI_Free_mem(y);
    POLYBENCH_FREE_ARRAY(y);

    if (world_rank==0)
    {
        MPI_Free_mem(A);
        MPI_Free_mem(u1);
        MPI_Free_mem(u2);
        MPI_Free_mem(z);
        MPI_Free_mem(w);

        POLYBENCH_FREE_ARRAY(A);
        POLYBENCH_FREE_ARRAY(u1);
        POLYBENCH_FREE_ARRAY(u2);
        POLYBENCH_FREE_ARRAY(w);
        POLYBENCH_FREE_ARRAY(z);
    }

    // Finalize the MPI environment.
    MPI_Finalize();

}

//central process distributes the work in blocks equally on all others and collects results
//non blocking communication
//non blocking allows computations and communication to overlap, which generally leads to improved performance
void gemver_non_blocking(int argc, char** argv) {
    // Initialize the MPI environment
    MPI_Init(&argc, &argv);

    /* Retrieve problem size. */
    int n = N;

    /* Variable declaration/allocation. */
    DATA_TYPE alpha;
    DATA_TYPE beta;
    
    POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);
    POLYBENCH_1D_ARRAY_DECL(u1, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(v1, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(u2, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(v2, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(w, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(x, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(y, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(z, DATA_TYPE, N, n);

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    //assign mpi memory for variables

    //for all processes
    MPI_Alloc_mem(n * sizeof(DATA_TYPE), MPI_INFO_NULL, &v1);
    MPI_Alloc_mem(n * sizeof(DATA_TYPE), MPI_INFO_NULL, &v2);
    MPI_Alloc_mem(n * sizeof(DATA_TYPE), MPI_INFO_NULL, &y);
    MPI_Alloc_mem(n * sizeof(DATA_TYPE), MPI_INFO_NULL, &x);

    //for first process only
    if (world_rank == 0)
    {
        MPI_Alloc_mem(n * n * sizeof(DATA_TYPE), MPI_INFO_NULL, &A);
        MPI_Alloc_mem(n * sizeof(DATA_TYPE), MPI_INFO_NULL, &u1);
        MPI_Alloc_mem(n * sizeof(DATA_TYPE), MPI_INFO_NULL, &u2);
        MPI_Alloc_mem(n * sizeof(DATA_TYPE), MPI_INFO_NULL, &w);
        MPI_Alloc_mem(n * sizeof(DATA_TYPE), MPI_INFO_NULL, &z);

        // Initialize array(s) only for process 0
        init_array(n, &alpha, &beta, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(u1), POLYBENCH_ARRAY(v1), POLYBENCH_ARRAY(u2),
               POLYBENCH_ARRAY(v2), POLYBENCH_ARRAY(w), POLYBENCH_ARRAY(x), POLYBENCH_ARRAY(y), POLYBENCH_ARRAY(z));
    }

    /*
    calculate  A = A + u1*v1 + u2*v2
    */

   if (world_rank == 0){
    polybench_start_instruments;
   }

    //split into blocks to distribute over the processes
    int block = n / world_size;
    int remaining = n % world_size;

    int *block_start_indx = (int *)malloc(world_size * sizeof(int));  
    int *num_elements = (int *)malloc(world_size * sizeof(int)); 
    for (int i = 0; i < world_size; i++)
    {
        if (i < world_size - 1)
        {
            num_elements[i] = block;
        }
        else
        {
            num_elements[i] = block + remaining;//last process also gets the remaining elements that dont fit in the vlock split
        }
        block_start_indx[i] = i * block;
    }

    int process_size = num_elements[world_rank];

    POLYBENCH_1D_ARRAY_DECL(process_u1, DATA_TYPE, process_size, process_size);
    POLYBENCH_1D_ARRAY_DECL(process_u2, DATA_TYPE, process_size, process_size);

    MPI_Request send_u1, send_u2;

    MPI_Iscatterv(u1, num_elements, block_start_indx, MPI_DOUBLE, process_u1, process_size, MPI_DOUBLE, 0, MPI_COMM_WORLD, &send_u1);
    MPI_Iscatterv(u2, num_elements, block_start_indx, MPI_DOUBLE, process_u2, process_size, MPI_DOUBLE, 0, MPI_COMM_WORLD, &send_u2);

    //do the block split for A as well
    int *block_start_indx_matrix = (int *)malloc(world_size * sizeof(int));
    int *num_elements_matrix = (int *)malloc(world_size * sizeof(int));
    for (int i = 0; i < world_size; i++)
    {
        if (i < world_size - 1)
        {
            num_elements_matrix[i] = block * n;
        }
        else
        {
            num_elements_matrix[i] = (block + remaining) * n;//last process also gets the remaining elements that dont fit in the vlock split
        }
        block_start_indx_matrix[i] = i * block * n;
    }

    POLYBENCH_2D_ARRAY_DECL(process_A, DATA_TYPE, process_size, N, process_size, n);
    MPI_Request send_A;
    MPI_Request rec_A;

    MPI_Iscatterv(A, num_elements_matrix, block_start_indx_matrix, MPI_DOUBLE, process_A, process_size * n, MPI_DOUBLE, 0, MPI_COMM_WORLD, &send_A);

    MPI_Bcast(v1, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(v2, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Wait(&send_u1, MPI_STATUS_IGNORE);
    MPI_Wait(&send_u2, MPI_STATUS_IGNORE);
    MPI_Wait(&send_A, MPI_STATUS_IGNORE);

    //compute A
    for (int i = 0; i < process_size; i++)
    {
        for (int j = 0; j < n; j++)
        {
            (*process_A)[i][j] += (*process_u1)[i] * (*v1)[j] + (*process_u2)[i] * (*v2)[j];
        }
    }
    //send results to first process
    MPI_Igatherv(process_A, process_size * n, MPI_DOUBLE, A, num_elements_matrix, block_start_indx_matrix, MPI_DOUBLE, 0, MPI_COMM_WORLD, &rec_A);

    /*
    calculate  x = x + b * A^T * y + z
    */

    MPI_Datatype columns, new_columns;//used to send tge blocks of the 2d array

    MPI_Type_vector(n, 1, n, MPI_DOUBLE, &columns);

    MPI_Type_commit(&columns);

    MPI_Type_create_resized(columns, 0, 1 * sizeof(DATA_TYPE), &new_columns); 
    MPI_Type_commit(&new_columns);

    int *displsMatrixTranspose = (int *)malloc(world_size * sizeof(int));  // stores start index of each block
    int *scountsMatrixTranspose = (int *)malloc(world_size * sizeof(int)); // stores number of elements in each block
    int block_start = 0;
    for (int i = 0; i < world_size; i++)
    {
        if (i < world_size - 1)
        {
            scountsMatrixTranspose[i] = block;
        }
        else
        {
            scountsMatrixTranspose[i] = (block + remaining);
        }
        displsMatrixTranspose[i] = block_start;
        block_start += scountsMatrixTranspose[i];
    }

    MPI_Request send_A_T, send_z, send_x;
    MPI_Wait(&rec_A, MPI_STATUS_IGNORE);
    //transpose A by sending columnsas rows
    POLYBENCH_2D_ARRAY_DECL(process_A_T, DATA_TYPE, process_size, N, process_size, n);
    MPI_Iscatterv(A, scountsMatrixTranspose, displsMatrixTranspose, new_columns, process_A_T, process_size * n, MPI_DOUBLE, 0, MPI_COMM_WORLD, &send_A_T);

    POLYBENCH_1D_ARRAY_DECL(process_z, DATA_TYPE, process_size, process_size);
    POLYBENCH_1D_ARRAY_DECL(process_x, DATA_TYPE, process_size, process_size);
    MPI_Iscatterv(z, num_elements, block_start_indx, MPI_DOUBLE, process_z, process_size, MPI_DOUBLE, 0, MPI_COMM_WORLD, &send_z);
    MPI_Iscatterv(x, num_elements, block_start_indx, MPI_DOUBLE, process_x, process_size, MPI_DOUBLE, 0, MPI_COMM_WORLD, &send_x);

    //send y,b everywhere
    MPI_Bcast(y, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&beta, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Wait(&send_A_T, MPI_STATUS_IGNORE);
    MPI_Wait(&send_z, MPI_STATUS_IGNORE);
    MPI_Wait(&send_x, MPI_STATUS_IGNORE);

    //x = b * A^T * y + z
    for (int i = 0; i < process_size; i++)
    {
        for (int j = 0; j < n; j++)
        {
            (*process_x)[i] = (*process_x)[i] + beta * (*process_A_T)[i][j] * (*y)[j];
        }
        (*process_x)[i] += (*process_z)[i];
    }
    MPI_Request rec_x;
    //send results to first process
    MPI_Igatherv(process_x, process_size, MPI_DOUBLE, x, num_elements, block_start_indx, MPI_DOUBLE, 0, MPI_COMM_WORLD, &rec_x);

    /*
    calculate  w = a * A * x
    */
    MPI_Request send_A_res, send_x_res, send_w;

    POLYBENCH_2D_ARRAY_DECL(process_A_res, DATA_TYPE, process_size, N, process_size, n);
    MPI_Iscatterv(A, num_elements_matrix, block_start_indx_matrix, MPI_DOUBLE, process_A_res, process_size * n, MPI_DOUBLE, 0, MPI_COMM_WORLD, &send_A_res);

    //send x everywhere
    MPI_Wait(&rec_x, MPI_STATUS_IGNORE);
    MPI_Ibcast(x, n, MPI_DOUBLE, 0, MPI_COMM_WORLD, &send_x_res);
    MPI_Bcast(&alpha, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    POLYBENCH_1D_ARRAY_DECL(process_w, DATA_TYPE, process_size, process_size);
    MPI_Iscatterv(w, num_elements, block_start_indx, MPI_DOUBLE, process_w, process_size, MPI_DOUBLE, 0, MPI_COMM_WORLD, &send_w);

    MPI_Wait(&send_A_res, MPI_STATUS_IGNORE);
    MPI_Wait(&send_x_res, MPI_STATUS_IGNORE);
    MPI_Wait(&send_w, MPI_STATUS_IGNORE);

    //w = a * A * x
    for (int i = 0; i < process_size; i++)
    {
        for (int j = 0; j < n; j++)
        {
            (*process_w)[i] = (*process_w)[i] + alpha * (*process_A_res)[i][j] * (*x)[j];
        }
    }
        
    MPI_Request rec_w_res;
    //send results to first process
    MPI_Igatherv(process_w, process_size, MPI_DOUBLE, w, num_elements, block_start_indx, MPI_DOUBLE, 0, MPI_COMM_WORLD, &rec_w_res);
    MPI_Wait(&rec_w_res, MPI_STATUS_IGNORE);

    if (world_rank == 0){
        /* Stop and print timer. */
        polybench_stop_instruments;
        polybench_print_instruments;

        /* Prevent dead-code elimination. All live-out data must be printed
        by the function call in argument. */
        polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(w)));
    }

    //cleanup
    MPI_Free_mem(num_elements);
    POLYBENCH_FREE_ARRAY(num_elements);
    MPI_Free_mem(num_elements_matrix);
    POLYBENCH_FREE_ARRAY(num_elements_matrix);
    MPI_Free_mem(block_start_indx);
    POLYBENCH_FREE_ARRAY(block_start_indx);
    MPI_Free_mem(block_start_indx_matrix);
    POLYBENCH_FREE_ARRAY(block_start_indx_matrix);
    MPI_Free_mem(process_u1);
    POLYBENCH_FREE_ARRAY(process_u1);
    MPI_Free_mem(process_u2);
    POLYBENCH_FREE_ARRAY(process_u2);
    MPI_Free_mem(process_A);
    POLYBENCH_FREE_ARRAY(process_A);
    MPI_Free_mem(process_A_T);
    POLYBENCH_FREE_ARRAY(process_A_T);
    MPI_Free_mem(process_z);
    POLYBENCH_FREE_ARRAY(process_z);
    MPI_Free_mem(process_x);
    POLYBENCH_FREE_ARRAY(process_x);
    MPI_Free_mem(process_w);
    POLYBENCH_FREE_ARRAY(process_w);
    MPI_Free_mem(process_A_res);
    POLYBENCH_FREE_ARRAY(process_A_res);
    MPI_Free_mem(v1); 
    POLYBENCH_FREE_ARRAY(v1);
    MPI_Free_mem(v2);
    POLYBENCH_FREE_ARRAY(v2);
    MPI_Free_mem(x);
    POLYBENCH_FREE_ARRAY(x);
    MPI_Free_mem(y);
    POLYBENCH_FREE_ARRAY(y);

    if (world_rank==0)
    {
        MPI_Free_mem(A);
        MPI_Free_mem(u1);
        MPI_Free_mem(u2);
        MPI_Free_mem(z);
        MPI_Free_mem(w);

        POLYBENCH_FREE_ARRAY(A);
        POLYBENCH_FREE_ARRAY(u1);
        POLYBENCH_FREE_ARRAY(u2);
        POLYBENCH_FREE_ARRAY(w);
        POLYBENCH_FREE_ARRAY(z);
    }

    // Finalize the MPI environment.
    MPI_Finalize();

}


//central process distributes the work in blocks equally on all others and collects results
//this time we split into columns not rows
//so we have to do less scatters

void gemver_column_split(int argc, char** argv) {
    // Initialize the MPI environment
    MPI_Init(&argc, &argv);

    /* Retrieve problem size. */
    int n = N;

    /* Variable declaration/allocation. */
    DATA_TYPE alpha;
    DATA_TYPE beta;
    
    POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N, N, n, n);
    POLYBENCH_1D_ARRAY_DECL(u1, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(v1, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(u2, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(v2, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(w, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(x, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(y, DATA_TYPE, N, n);
    POLYBENCH_1D_ARRAY_DECL(z, DATA_TYPE, N, n);

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    //assign mpi memory for variables

    //for all processes
    MPI_Alloc_mem(n * sizeof(DATA_TYPE), MPI_INFO_NULL, &v1);
    MPI_Alloc_mem(n * sizeof(DATA_TYPE), MPI_INFO_NULL, &v2);
    MPI_Alloc_mem(n * sizeof(DATA_TYPE), MPI_INFO_NULL, &y);
    MPI_Alloc_mem(n * sizeof(DATA_TYPE), MPI_INFO_NULL, &x);

    //for first process only
    if (world_rank == 0)
    {
        MPI_Alloc_mem(n * n * sizeof(DATA_TYPE), MPI_INFO_NULL, &A);
        MPI_Alloc_mem(n * sizeof(DATA_TYPE), MPI_INFO_NULL, &u1);
        MPI_Alloc_mem(n * sizeof(DATA_TYPE), MPI_INFO_NULL, &u2);
        MPI_Alloc_mem(n * sizeof(DATA_TYPE), MPI_INFO_NULL, &w);
        MPI_Alloc_mem(n * sizeof(DATA_TYPE), MPI_INFO_NULL, &z);

        // Initialize array(s) only for process 0
        init_array(n, &alpha, &beta, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(u1), POLYBENCH_ARRAY(v1), POLYBENCH_ARRAY(u2),
               POLYBENCH_ARRAY(v2), POLYBENCH_ARRAY(w), POLYBENCH_ARRAY(x), POLYBENCH_ARRAY(y), POLYBENCH_ARRAY(z));
    }

    /*
    calculate  A = A + u1*v1 + u2*v2
    */

   if (world_rank == 0){
    polybench_start_instruments;
   }

    //split into blocks to distribute over the processes
    int block = n / world_size;
    int remaining = n % world_size;

    int *block_start_indx = (int *)malloc(world_size * sizeof(int));  
    int *num_elements = (int *)malloc(world_size * sizeof(int)); 
    for (int i = 0; i < world_size; i++)
    {
        if (i < world_size - 1)
        {
            num_elements[i] = block;
        }
        else
        {
            num_elements[i] = block + remaining;//last process also gets the remaining elements that dont fit in the vlock split
        }
        block_start_indx[i] = i * block;
    }

    int process_size = num_elements[world_rank];

    POLYBENCH_1D_ARRAY_DECL(process_u1, DATA_TYPE, process_size, process_size);
    POLYBENCH_1D_ARRAY_DECL(process_u2, DATA_TYPE, process_size, process_size);

    MPI_Scatterv(u1, num_elements, block_start_indx, MPI_DOUBLE, process_u1, process_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(u2, num_elements, block_start_indx, MPI_DOUBLE, process_u2, process_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    //do the block split for A as well
    int *block_start_indx_matrix = (int *)malloc(world_size * sizeof(int));
    int *num_elements_matrix = (int *)malloc(world_size * sizeof(int));
    for (int i = 0; i < world_size; i++)
    {
        if (i < world_size - 1)
        {
            num_elements_matrix[i] = block * n;
        }
        else
        {
            num_elements_matrix[i] = (block + remaining) * n;//last process also gets the remaining elements that dont fit in the vlock split
        }
        block_start_indx_matrix[i] = i * block * n;
    }

    POLYBENCH_2D_ARRAY_DECL(process_A, DATA_TYPE, process_size, N, process_size, n);
    //double *process_A = (double *)malloc(process_size * n * sizeof(double));
    MPI_Scatterv(A, num_elements_matrix, block_start_indx_matrix, MPI_DOUBLE, process_A, process_size * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(v1, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(v2, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    //compute A
    for (int i = 0; i < process_size; i++)
    {
        for (int j = 0; j < n; j++)
        {
            (*process_A)[i][j] += (*process_u1)[i] * (*v1)[j] + (*process_u2)[i] * (*v2)[j];
        }
    }
    //send results to first process
    MPI_Gatherv(process_A, process_size * n, MPI_DOUBLE, A, num_elements_matrix, block_start_indx_matrix, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /*
    calculate  x = x + b * A^T * y + z
    */

    MPI_Datatype columns, new_columns;//used to send tge blocks of the 2d array

    MPI_Type_vector(n, 1, n, MPI_DOUBLE, &columns);

    MPI_Type_commit(&columns);

    MPI_Type_create_resized(columns, 0, 1 * sizeof(DATA_TYPE), &new_columns); 
    MPI_Type_commit(&new_columns);

    int *displsMatrixTranspose = (int *)malloc(world_size * sizeof(int));  // stores start index of each block
    int *scountsMatrixTranspose = (int *)malloc(world_size * sizeof(int)); // stores number of elements in each block
    int block_start = 0;
    for (int i = 0; i < world_size; i++)
    {
        if (i < world_size - 1)
        {
            scountsMatrixTranspose[i] = block;
        }
        else
        {
            scountsMatrixTranspose[i] = (block + remaining);
        }
        displsMatrixTranspose[i] = block_start;
        block_start += scountsMatrixTranspose[i];
    }

    //transpose A by sending columnsas rows
    POLYBENCH_2D_ARRAY_DECL(process_A_T, DATA_TYPE, process_size, N, process_size, n);
    MPI_Scatterv(A, scountsMatrixTranspose, displsMatrixTranspose, new_columns, process_A_T, process_size * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    POLYBENCH_1D_ARRAY_DECL(process_z, DATA_TYPE, process_size, process_size);
    POLYBENCH_1D_ARRAY_DECL(process_x, DATA_TYPE, process_size, process_size);
    MPI_Scatterv(z, num_elements, block_start_indx, MPI_DOUBLE, process_z, process_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(x, num_elements, block_start_indx, MPI_DOUBLE, process_x, process_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    //send y,b everywhere
    MPI_Bcast(y, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&beta, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    //x = b * A^T * y + z
    for (int i = 0; i < process_size; i++)
    {
        for (int j = 0; j < n; j++)
        {
            (*process_x)[i] = (*process_x)[i] + beta * (*process_A_T)[i][j] * (*y)[j];
        }
        (*process_x)[i] += (*process_z)[i];
    }

    //send results to first process
    MPI_Gatherv(process_x, process_size, MPI_DOUBLE, x, num_elements, block_start_indx, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /*
    calculate  w = a * A * x
    */

    POLYBENCH_2D_ARRAY_DECL(process_A_res, DATA_TYPE, process_size, N, process_size, n);
    MPI_Scatterv(A, num_elements_matrix, block_start_indx_matrix, MPI_DOUBLE, process_A_res, process_size * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    //send x everywhere
    MPI_Bcast(x, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&alpha, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    POLYBENCH_1D_ARRAY_DECL(process_w, DATA_TYPE, process_size, process_size);
    MPI_Scatterv(w, num_elements, block_start_indx, MPI_DOUBLE, process_w, process_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    //w = a * A * x
    for (int i = 0; i < process_size; i++)
    {
        for (int j = 0; j < n; j++)
        {
            (*process_w)[i] = (*process_w)[i] + alpha * (*process_A_res)[i][j] * (*x)[j];
        }
    }
        

    //send results to first process
    MPI_Gatherv(process_w, process_size, MPI_DOUBLE, w, num_elements, block_start_indx, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (world_rank == 0){
        /* Stop and print timer. */
        polybench_stop_instruments;
        polybench_print_instruments;

        /* Prevent dead-code elimination. All live-out data must be printed
        by the function call in argument. */
        polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(w)));
    }

    //cleanup
    MPI_Free_mem(num_elements);
    POLYBENCH_FREE_ARRAY(num_elements);
    MPI_Free_mem(num_elements_matrix);
    POLYBENCH_FREE_ARRAY(num_elements_matrix);
    MPI_Free_mem(block_start_indx);
    POLYBENCH_FREE_ARRAY(block_start_indx);
    MPI_Free_mem(block_start_indx_matrix);
    POLYBENCH_FREE_ARRAY(block_start_indx_matrix);
    MPI_Free_mem(process_u1);
    POLYBENCH_FREE_ARRAY(process_u1);
    MPI_Free_mem(process_u2);
    POLYBENCH_FREE_ARRAY(process_u2);
    MPI_Free_mem(process_A);
    POLYBENCH_FREE_ARRAY(process_A);
    MPI_Free_mem(process_A_T);
    POLYBENCH_FREE_ARRAY(process_A_T);
    MPI_Free_mem(process_z);
    POLYBENCH_FREE_ARRAY(process_z);
    MPI_Free_mem(process_x);
    POLYBENCH_FREE_ARRAY(process_x);
    MPI_Free_mem(process_w);
    POLYBENCH_FREE_ARRAY(process_w);
    MPI_Free_mem(process_A_res);
    POLYBENCH_FREE_ARRAY(process_A_res);
    MPI_Free_mem(v1); 
    POLYBENCH_FREE_ARRAY(v1);
    MPI_Free_mem(v2);
    POLYBENCH_FREE_ARRAY(v2);
    MPI_Free_mem(x);
    POLYBENCH_FREE_ARRAY(x);
    MPI_Free_mem(y);
    POLYBENCH_FREE_ARRAY(y);

    if (world_rank==0)
    {
        MPI_Free_mem(A);
        MPI_Free_mem(u1);
        MPI_Free_mem(u2);
        MPI_Free_mem(z);
        MPI_Free_mem(w);

        POLYBENCH_FREE_ARRAY(A);
        POLYBENCH_FREE_ARRAY(u1);
        POLYBENCH_FREE_ARRAY(u2);
        POLYBENCH_FREE_ARRAY(w);
        POLYBENCH_FREE_ARRAY(z);
    }

    // Finalize the MPI environment.
    MPI_Finalize();

}


int main(int argc, char** argv){

    gemver(argc,argv);
    gemver_non_blocking(argc,argv);

    return 0;
}