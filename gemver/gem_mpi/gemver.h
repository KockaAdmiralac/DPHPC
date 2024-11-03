/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
/*
* header file taken and edited for DPHPC 2024 project
* added function declarations for mpi
* added all includes needed to have only one include in main c file
*/
#ifndef _GEMVER_MPI_H
#define _GEMVER_MPI_H

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

/* Include polybench common header. */
#include <polybench.h>
/* Include mpi header. */
#include <mpi.h>

/* Default to LARGE_DATASET. */
#if !defined(MINI_DATASET) && !defined(SMALL_DATASET) && !defined(MEDIUM_DATASET) && !defined(LARGE_DATASET) && \
    !defined(EXTRALARGE_DATASET)
#define LARGE_DATASET
#endif

#if !defined(N)
/* Define sample dataset sizes. */
#ifdef MINI_DATASET
#define N 40
#endif

#ifdef SMALL_DATASET
#define N 120
#endif

#ifdef MEDIUM_DATASET
#define N 400
#endif

#ifdef LARGE_DATASET
#define N 2000
#endif

#ifdef EXTRALARGE_DATASET
#define N 4000
#endif

#endif /* !(N) */

#define _PB_N POLYBENCH_LOOP_BOUND(N, n)

/* Default data type */
#if !defined(DATA_TYPE_IS_INT) && !defined(DATA_TYPE_IS_FLOAT) && !defined(DATA_TYPE_IS_DOUBLE)
#define DATA_TYPE_IS_DOUBLE
#endif

#ifdef DATA_TYPE_IS_INT
#define DATA_TYPE int
#define DATA_PRINTF_MODIFIER "%d "
#endif

#ifdef DATA_TYPE_IS_FLOAT
#define DATA_TYPE float
#define DATA_PRINTF_MODIFIER "%0.2f "
#define SCALAR_VAL(x) x##f
#define SQRT_FUN(x) sqrtf(x)
#define EXP_FUN(x) expf(x)
#define POW_FUN(x, y) powf(x, y)
#endif

#ifdef DATA_TYPE_IS_DOUBLE
#define DATA_TYPE double
#define DATA_PRINTF_MODIFIER "%0.2lf "
#define SCALAR_VAL(x) x
#define SQRT_FUN(x) sqrt(x)
#define EXP_FUN(x) exp(x)
#define POW_FUN(x, y) pow(x, y)
#endif

void kernel_gemver(int n, DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE POLYBENCH_2D(A, N, N, n, n),
                   DATA_TYPE POLYBENCH_1D(u1, N, n), DATA_TYPE POLYBENCH_1D(v1, N, n), DATA_TYPE POLYBENCH_1D(u2, N, n),
                   DATA_TYPE POLYBENCH_1D(v2, N, n), DATA_TYPE POLYBENCH_1D(w, N, n), DATA_TYPE POLYBENCH_1D(x, N, n),
                   DATA_TYPE POLYBENCH_1D(y, N, n), DATA_TYPE POLYBENCH_1D(z, N, n));

#endif /* !_GEMVER_MPI_H */
