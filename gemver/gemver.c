/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/* gemver.c: this file is part of PolyBench/C */

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "gemver.h"

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(int n, DATA_TYPE POLYBENCH_1D(w, N2, n), const char *array_name) {
#ifdef DUMP_DATA_HUMAN_READABLE
    int i;

    POLYBENCH_DUMP_START;
    POLYBENCH_DUMP_BEGIN(array_name);
    for (i = 0; i < n; i++) {
        if (i == 0) fprintf(POLYBENCH_DUMP_TARGET, "\n");
        fprintf(POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, w[i]);
    }
    POLYBENCH_DUMP_END(array_name);
    POLYBENCH_DUMP_FINISH;
#else
    print_binary_array(n, (double *)w, array_name[0]);
#endif
}

int main(int argc, char **argv) {
    (void)argc;
    (void)argv;
    /* Retrieve problem size. */
    int n = N2;

    /* Variable declaration/allocation. */
    DATA_TYPE alpha;
    DATA_TYPE beta;
    POLYBENCH_2D_ARRAY_DECL(A, DATA_TYPE, N2, N2, n, n);
    POLYBENCH_1D_ARRAY_DECL(u1, DATA_TYPE, N2, n);
    POLYBENCH_1D_ARRAY_DECL(v1, DATA_TYPE, N2, n);
    POLYBENCH_1D_ARRAY_DECL(u2, DATA_TYPE, N2, n);
    POLYBENCH_1D_ARRAY_DECL(v2, DATA_TYPE, N2, n);
    POLYBENCH_1D_ARRAY_DECL(w, DATA_TYPE, N2, n);
    POLYBENCH_1D_ARRAY_DECL(x, DATA_TYPE, N2, n);
    POLYBENCH_1D_ARRAY_DECL(y, DATA_TYPE, N2, n);
    POLYBENCH_1D_ARRAY_DECL(z, DATA_TYPE, N2, n);

    initialise_benchmark(argc, argv, n, &alpha, &beta, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(u1), POLYBENCH_ARRAY(v1),
                         POLYBENCH_ARRAY(u2), POLYBENCH_ARRAY(v2), POLYBENCH_ARRAY(w), POLYBENCH_ARRAY(x),
                         POLYBENCH_ARRAY(y), POLYBENCH_ARRAY(z));

    /* Start timer. */
    polybench_start_instruments;

    /* Run kernel. */
    kernel_gemver(n, alpha, beta, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(u1), POLYBENCH_ARRAY(v1), POLYBENCH_ARRAY(u2),
                  POLYBENCH_ARRAY(v2), POLYBENCH_ARRAY(w), POLYBENCH_ARRAY(x), POLYBENCH_ARRAY(y), POLYBENCH_ARRAY(z));

    /* Stop and print timer. */
    polybench_stop_instruments;
    polybench_print_instruments;

    finish_benchmark(n, alpha, beta, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(u1), POLYBENCH_ARRAY(v1), POLYBENCH_ARRAY(u2),
                     POLYBENCH_ARRAY(v2), POLYBENCH_ARRAY(w), POLYBENCH_ARRAY(x), POLYBENCH_ARRAY(y),
                     POLYBENCH_ARRAY(z));

    /* Prevent dead-code elimination. All live-out data must be printed
       by the function call in argument. */
#ifndef DISABLE_CHECKING
    polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(w), "w"));
    polybench_prevent_dce(print_array(n * n, (double *)A, "A"));
    polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(x), "x"));
#endif

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

    return 0;
}
