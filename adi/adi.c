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

int main(int argc, char** argv) {
    /* Retrieve problem size. */
    int n = N2;
    int tsteps = TSTEPS;

    /* Variable declaration/allocation. */
    POLYBENCH_2D_ARRAY_DECL(u, DATA_TYPE, N2, N2, n, n);
    POLYBENCH_2D_ARRAY_DECL(v, DATA_TYPE, N2, N2, n, n);
    POLYBENCH_2D_ARRAY_DECL(p, DATA_TYPE, N2, N2, n, n);
    POLYBENCH_2D_ARRAY_DECL(q, DATA_TYPE, N2, N2, n, n);

    initialise_benchmark(argc, argv, tsteps, n, POLYBENCH_ARRAY(u), POLYBENCH_ARRAY(v), POLYBENCH_ARRAY(p),
                         POLYBENCH_ARRAY(q));

    /* Start timer. */
    polybench_start_instruments;

    /* Run kernel. */
    kernel_adi(tsteps, n, POLYBENCH_ARRAY(u), POLYBENCH_ARRAY(v), POLYBENCH_ARRAY(p), POLYBENCH_ARRAY(q));

    /* Stop and print timer. */
    polybench_stop_instruments;

    finish_benchmark(tsteps, n, POLYBENCH_ARRAY(u), POLYBENCH_ARRAY(v), POLYBENCH_ARRAY(p), POLYBENCH_ARRAY(q));

    polybench_print_instruments;
    /* Prevent dead-code elimination. All live-out data must be printed
       by the function call in argument. */
#ifndef DISABLE_CHECKING
    polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(u), "u"));
#endif

    /* Be clean. */
    POLYBENCH_FREE_ARRAY(u);
    POLYBENCH_FREE_ARRAY(v);
    POLYBENCH_FREE_ARRAY(p);
    POLYBENCH_FREE_ARRAY(q);

    return 0;
}
