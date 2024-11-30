/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include <cuda_support.cuh>

#include "adi.h"

typedef struct {
    default_adi_data_t adi_data;
    DATA_TYPE *u_dev;
    DATA_TYPE *v_dev;
    DATA_TYPE *p_dev;
    DATA_TYPE *q_dev;
} cuda_adi_data_t;

void initialise_benchmark(int argc, char **argv, int tsteps, int n, void **gen_data_ptr) {
    *gen_data_ptr = polybench_alloc_data(1, sizeof(cuda_adi_data_t));
    cuda_adi_data_t *data_ptr = (cuda_adi_data_t *)(*gen_data_ptr);
    default_initialise_benchmark(argc, argv, tsteps, n, &data_ptr->adi_data);

    gpuErrchk(cudaMalloc(&data_ptr->u_dev, sizeof(DATA_TYPE) * n * n));
    gpuErrchk(cudaMalloc(&data_ptr->v_dev, sizeof(DATA_TYPE) * n * n));
    gpuErrchk(cudaMalloc(&data_ptr->p_dev, sizeof(DATA_TYPE) * n * n));
    gpuErrchk(cudaMalloc(&data_ptr->q_dev, sizeof(DATA_TYPE) * n * n));

    gpuErrchk(cudaMemcpy(data_ptr->u_dev, data_ptr->adi_data.u, sizeof(DATA_TYPE) * n * n, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(data_ptr->v_dev, data_ptr->adi_data.v, sizeof(DATA_TYPE) * n * n, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(data_ptr->p_dev, data_ptr->adi_data.p, sizeof(DATA_TYPE) * n * n, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(data_ptr->q_dev, data_ptr->adi_data.q, sizeof(DATA_TYPE) * n * n, cudaMemcpyHostToDevice));
    gpuErrchk(cudaDeviceSynchronize());
}

void finish_benchmark(void *gen_data_ptr) {
    cuda_adi_data_t *data_ptr = (cuda_adi_data_t *)gen_data_ptr;
    int n = data_ptr->adi_data.n;
    gpuErrchk(cudaMemcpy(data_ptr->adi_data.u, data_ptr->u_dev, sizeof(DATA_TYPE) * n * n, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(data_ptr->adi_data.v, data_ptr->v_dev, sizeof(DATA_TYPE) * n * n, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(data_ptr->adi_data.p, data_ptr->p_dev, sizeof(DATA_TYPE) * n * n, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(data_ptr->adi_data.q, data_ptr->q_dev, sizeof(DATA_TYPE) * n * n, cudaMemcpyDeviceToHost));
}

void print_data(int argc, char **argv, int n, void *gen_data_ptr) {
    default_print_data(argc, argv, n, &((cuda_adi_data_t *)gen_data_ptr)->adi_data);
}

void free_data(void *gen_data_ptr) {
    cuda_adi_data_t *data_ptr = (cuda_adi_data_t *)gen_data_ptr;
    gpuErrchk(cudaFree(data_ptr->u_dev));
    gpuErrchk(cudaFree(data_ptr->v_dev));
    gpuErrchk(cudaFree(data_ptr->p_dev));
    gpuErrchk(cudaFree(data_ptr->q_dev));
    free(gen_data_ptr);
}

__global__ void col_sweep(int tsteps, int n, DATA_TYPE *u, DATA_TYPE *v, DATA_TYPE *p, DATA_TYPE *q, DATA_TYPE a,
                          DATA_TYPE b, DATA_TYPE c, DATA_TYPE d, DATA_TYPE f) {
    for (int i = 1 + blockDim.x * blockIdx.x + threadIdx.x; i < _PB_N - 1; i += gridDim.x * blockDim.x) {
        v[0 * n + i] = SCALAR_VAL(1.0);
        p[i * n + 0] = SCALAR_VAL(0.0);
        q[i * n + 0] = v[0 * n + i];
        for (int j = 1; j < _PB_N - 1; j++) {
            p[i * n + j] = -c / (a * p[i * n + (j - 1)] + b);
            q[i * n + j] = (-d * u[j * n + (i - 1)] + (SCALAR_VAL(1.0) + SCALAR_VAL(2.0) * d) * u[j * n + i] -
                            f * u[j * n + (i + 1)] - a * q[i * n + (j - 1)]) /
                           (a * p[i * n + (j - 1)] + b);
        }

        v[(_PB_N - 1) * n + i] = SCALAR_VAL(1.0);
        for (int j = _PB_N - 2; j >= 1; j--) {
            v[j * n + i] = p[i * n + j] * v[(j + 1) * n + i] + q[i * n + j];
        }
    }
}

__global__ void row_sweep(int tsteps, int n, DATA_TYPE *u, DATA_TYPE *v, DATA_TYPE *p, DATA_TYPE *q, DATA_TYPE a,
                          DATA_TYPE c, DATA_TYPE d, DATA_TYPE e, DATA_TYPE f) {
    for (int i = 1 + blockDim.x * blockIdx.x + threadIdx.x; i < _PB_N - 1; i += gridDim.x * blockDim.x) {
        u[i * n + 0] = SCALAR_VAL(1.0);
    }
    for (int i = 1 + blockDim.x * blockIdx.x + threadIdx.x; i < _PB_N - 1; i += gridDim.x * blockDim.x) {
        p[i * n + 0] = SCALAR_VAL(0.0);
        q[i * n + 0] = u[i * n + 0];
    }
    for (int i = 1 + blockDim.x * blockIdx.x + threadIdx.x; i < _PB_N - 1; i += gridDim.x * blockDim.x) {
        for (int j = 1; j < _PB_N - 1; j++) {
            p[i * n + j] = -f / (d * p[i * n + (j - 1)] + e);
            q[i * n + j] = (-a * v[(i - 1) * n + j] + (SCALAR_VAL(1.0) + SCALAR_VAL(2.0) * a) * v[i * n + j] -
                            c * v[(i + 1) * n + j] - d * q[i * n + (j - 1)]) /
                           (d * p[i * n + (j - 1)] + e);
        }
        u[i * n + (_PB_N - 1)] = SCALAR_VAL(1.0);
        for (int j = _PB_N - 2; j >= 1; j--) {
            u[i * n + j] = p[i * n + j] * u[i * n + (j + 1)] + q[i * n + j];
        }
    }
}

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
/* Based on a Fortran code fragment from Figure 5 of
 * "Automatic Data and Computation Decomposition on Distributed Memory Parallel
 * Computers" by Peizong Lee and Zvi Meir Kedem, TOPLAS, 2002
 */
void kernel_adi_inner(int tsteps, int n, DATA_TYPE *u, DATA_TYPE *v, DATA_TYPE *p, DATA_TYPE *q) {
    DATA_TYPE DX, DY, DT;
    DATA_TYPE B1, B2;
    DATA_TYPE mul1, mul2;
    DATA_TYPE a, b, c, d, e, f;

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

    for (int t = 1; t <= _PB_TSTEPS; t++) {
        // Column Sweep
        col_sweep<<<32, 64>>>(tsteps, n, u, v, p, q, a, b, c, d, f);
        // Row Sweep
        row_sweep<<<32, 64>>>(tsteps, n, u, v, p, q, a, c, d, e, f);
    }
}

void kernel_adi(void *gen_data_ptr) {
    cuda_adi_data_t *data_ptr = (cuda_adi_data_t *)gen_data_ptr;
    kernel_adi_inner(data_ptr->adi_data.tsteps, data_ptr->adi_data.n, data_ptr->u_dev, data_ptr->v_dev, data_ptr->p_dev,
                     data_ptr->q_dev);
    gpuErrchk(cudaDeviceSynchronize());
}
