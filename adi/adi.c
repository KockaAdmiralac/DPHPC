/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* adi.c: this file is part of PolyBench/C */

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "adi.h"

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(int n, DATA_TYPE POLYBENCH_2D(u, N2, N2, n, n), const char* arr_name)

{
#ifdef DUMP_DATA_HUMAN_READABLE
    int i, j;

    POLYBENCH_DUMP_START;
    POLYBENCH_DUMP_BEGIN(arr_name);
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++) {
            if ((i * n + j) == 0) fprintf(POLYBENCH_DUMP_TARGET, "\n");
            fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, u[i][j]);
        }
    POLYBENCH_DUMP_END(arr_name);
    POLYBENCH_DUMP_FINISH;
#else
    print_binary_array(n * n, (double*)u, arr_name[0]);
#endif
}

void default_initialise_benchmark(int argc, char** argv, int tsteps, int n, default_adi_data_t* data_ptr) {
    data_ptr->tsteps = tsteps;
    data_ptr->n = n;
    data_ptr->argc = argc;
    data_ptr->argv = argv;

    int i, j;

    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++) {
            data_ptr->u[i][j] = (DATA_TYPE)(i + n - j) / n;
            data_ptr->v[i][j] = 0.0;
            data_ptr->p[i][j] = 0.0;
            data_ptr->q[i][j] = 0.0;
        }
}

void default_initialise_benchmark_and_alloc(int argc, char** argv, int tsteps, int n, default_adi_data_t** data_ptr) {
    *data_ptr = polybench_alloc_data(1, sizeof(default_adi_data_t));
    default_initialise_benchmark(argc, argv, tsteps, n, *data_ptr);
}

void default_print_data(int argc, char** argv, int n, default_adi_data_t* data_ptr) {
#ifndef DISABLE_CHECKING
    (void)argc;
    (void)argv;
    polybench_prevent_dce(print_array(n, data_ptr->u, "u"));
#endif
}

#ifdef __GNUC__

__attribute__((weak)) void initialise_benchmark(int argc, char** argv, int tsteps, int n, void** gen_data_ptr) {
    default_initialise_benchmark_and_alloc(argc, argv, tsteps, n, (default_adi_data_t**)gen_data_ptr);
}

__attribute__((weak)) void finish_benchmark(void* data_ptr) { (void)data_ptr; }

__attribute__((weak)) void print_data(int argc, char** argv, int n, void* gen_data_ptr) {
    default_print_data(argc, argv, n, (default_adi_data_t*)gen_data_ptr);
}

__attribute__((weak)) bool should_print_counter(void* gen_data_ptr) {
    (void)gen_data_ptr;
    return true;
}

__attribute__((weak)) void free_data(void* gen_data_ptr) { free(gen_data_ptr); }
#endif

int main(int argc, char** argv) {
    /* Retrieve problem size. */
    int n = N2;
    int tsteps = TSTEPS;

    /* Variable declaration/allocation. */
    void* data;

    initialise_benchmark(argc, argv, tsteps, n, &data);

    /* Start timer. */
    polybench_start_instruments;

    /* Run kernel. */
    kernel_adi(data);

    /* Stop and print timer. */
    polybench_stop_instruments;

    finish_benchmark(data);

    if (should_print_counter(data)) {
        polybench_print_instruments;
    }
    /* Prevent dead-code elimination. All live-out data must be printed
       by the function call in argument. */
#ifndef DISABLE_CHECKING
    print_data(argc, argv, n, data);
#endif

    /* Be clean. */
    free_data(data);

    return 0;
}
