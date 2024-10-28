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

/* Array initialization. */
static void init_array(int n, DATA_TYPE *alpha, DATA_TYPE *beta, DATA_TYPE POLYBENCH_2D(A, N2, N2, n, n),
                       DATA_TYPE POLYBENCH_1D(u1, N2, n), DATA_TYPE POLYBENCH_1D(v1, N2, n),
                       DATA_TYPE POLYBENCH_1D(u2, N2, n), DATA_TYPE POLYBENCH_1D(v2, N2, n),
                       DATA_TYPE POLYBENCH_1D(w, N2, n), DATA_TYPE POLYBENCH_1D(x, N2, n),
                       DATA_TYPE POLYBENCH_1D(y, N2, n), DATA_TYPE POLYBENCH_1D(z, N2, n)) {
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

    /* Initialize array(s). */
    init_array(n, &alpha, &beta, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(u1), POLYBENCH_ARRAY(v1), POLYBENCH_ARRAY(u2),
               POLYBENCH_ARRAY(v2), POLYBENCH_ARRAY(w), POLYBENCH_ARRAY(x), POLYBENCH_ARRAY(y), POLYBENCH_ARRAY(z));

    /* Start timer. */
    polybench_start_instruments;

    /* Run kernel. */
    kernel_gemver(n, alpha, beta, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(u1), POLYBENCH_ARRAY(v1), POLYBENCH_ARRAY(u2),
                  POLYBENCH_ARRAY(v2), POLYBENCH_ARRAY(w), POLYBENCH_ARRAY(x), POLYBENCH_ARRAY(y), POLYBENCH_ARRAY(z));

    /* Stop and print timer. */
    polybench_stop_instruments;
    polybench_print_instruments;

    /* Prevent dead-code elimination. All live-out data must be printed
       by the function call in argument. */
    polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(w), "w"));
    polybench_prevent_dce(print_array(n * n, (double *)A, "A"));
    polybench_prevent_dce(print_array(n, POLYBENCH_ARRAY(x), "x"));

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
