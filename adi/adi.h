/**
 * This version is stamped on May 10, 2016
 *
 * Contact:
 *   Louis-Noel Pouchet <pouchet.ohio-state.edu>
 *   Tomofumi Yuki <tomofumi.yuki.fr>
 *
 * Web address: http://polybench.sourceforge.net
 */
#ifndef _ADI_H
#define _ADI_H
#include <stdbool.h>

/* Default to LARGE_DATASET. */
#if !defined(MINI_DATASET) && !defined(SMALL_DATASET) && !defined(MEDIUM_DATASET) && !defined(LARGE_DATASET) && \
    !defined(EXTRALARGE_DATASET)
#define LARGE_DATASET
#endif

#if !defined(TSTEPS) && !defined(N2)
/* Define sample dataset sizes. */
#ifdef MINI_DATASET
#define TSTEPS 20
#define N2 20
#endif

#ifdef SMALL_DATASET
#define TSTEPS 40
#define N2 60
#endif

#ifdef MEDIUM_DATASET
#define TSTEPS 100
#define N2 200
#endif

#ifdef LARGE_DATASET
#define TSTEPS 500
#define N2 1000
#endif

#ifdef EXTRALARGE_DATASET
#define TSTEPS 1000
#define N2 2000
#endif

#endif /* !(TSTEPS N2) */

#define _PB_TSTEPS POLYBENCH_LOOP_BOUND(TSTEPS, tsteps)
#define _PB_N POLYBENCH_LOOP_BOUND(N2, n)

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
#define DATA_PRINTF_MODIFIER "%0.8le "
#define SCALAR_VAL(x) x
#define SQRT_FUN(x) sqrt(x)
#define EXP_FUN(x) exp(x)
#define POW_FUN(x, y) pow(x, y)
#endif

typedef struct {
    DATA_TYPE POLYBENCH_2D(u, N2, N2, n, n);
    DATA_TYPE POLYBENCH_2D(v, N2, N2, n, n);
    DATA_TYPE POLYBENCH_2D(p, N2, N2, n, n);
    DATA_TYPE POLYBENCH_2D(q, N2, N2, n, n);
    int tsteps;
    int n;
    int argc;
    char **argv;
} default_adi_data_t;

#ifdef __cplusplus
extern "C"
#endif
    void
    default_initialise_benchmark(int argc, char **argv, int tsteps, int n, default_adi_data_t *data_ptr);
#ifdef __cplusplus
extern "C"
#endif
    void
    initialise_benchmark(int argc, char **argv, int tsteps, int n, void **data_ptr);

#ifdef __cplusplus
extern "C"
#endif
    void
    finish_benchmark(void *data_ptr);

#ifdef __cplusplus
extern "C"
#endif
    void
    default_print_data(int argc, char **argv, int n, default_adi_data_t *data_ptr);
#ifdef __cplusplus
extern "C"
#endif
    bool
    should_print_counter(void *gen_data_ptr);
#ifdef __cplusplus
extern "C"
#endif
    void
    print_data(int argc, char **argv, int n, void *data_ptr);

#ifdef __cplusplus
extern "C"
#endif
    void
    free_data(void *data_ptr);

#ifdef __cplusplus
extern "C"
#endif
    void
    kernel_adi(void *data_ptr);

#endif /* !_ADI_H */
