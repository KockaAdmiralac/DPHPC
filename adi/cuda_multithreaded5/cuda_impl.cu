/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include <cuda_support.cuh>

#include "adi.h"

typedef struct {
    DATA_TYPE *u_dev;
    DATA_TYPE *v_dev;
    DATA_TYPE *p_dev;
    DATA_TYPE *q_dev;
} kernel_init_t;

kernel_init_t device_addrs;

void initialise_benchmark(int argc, char **argv, int tsteps, int n, DATA_TYPE POLYBENCH_2D(u, N2, N2, n, n),
                          DATA_TYPE POLYBENCH_2D(v, N2, N2, n, n), DATA_TYPE POLYBENCH_2D(p, N2, N2, n, n),
                          DATA_TYPE POLYBENCH_2D(q, N2, N2, n, n)) {
    (void)tsteps;
    (void)n;
    (void)argc;
    (void)argv;

    int i, j;

    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++) {
            u[i][j] = (DATA_TYPE)(i + n - j) / n;
        }

    gpuErrchk(cudaMalloc(&device_addrs.u_dev, sizeof(DATA_TYPE) * n * n));
    gpuErrchk(cudaMalloc(&device_addrs.v_dev, sizeof(DATA_TYPE) * n * n));
    gpuErrchk(cudaMalloc(&device_addrs.p_dev, sizeof(DATA_TYPE) * n * n));
    gpuErrchk(cudaMalloc(&device_addrs.q_dev, sizeof(DATA_TYPE) * n * n));

    gpuErrchk(cudaMemcpy(device_addrs.u_dev, u, sizeof(DATA_TYPE) * n * n, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(device_addrs.v_dev, v, sizeof(DATA_TYPE) * n * n, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(device_addrs.p_dev, p, sizeof(DATA_TYPE) * n * n, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(device_addrs.q_dev, q, sizeof(DATA_TYPE) * n * n, cudaMemcpyHostToDevice));
}

void finish_benchmark(int tsteps, int n, DATA_TYPE POLYBENCH_2D(u, N2, N2, n, n),
                      DATA_TYPE POLYBENCH_2D(v, N2, N2, n, n), DATA_TYPE POLYBENCH_2D(p, N2, N2, n, n),
                      DATA_TYPE POLYBENCH_2D(q, N2, N2, n, n)) {
    (void)tsteps;
    (void)n;
    gpuErrchk(cudaMemcpy(u, device_addrs.u_dev, sizeof(DATA_TYPE) * n * n, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(v, device_addrs.v_dev, sizeof(DATA_TYPE) * n * n, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(p, device_addrs.p_dev, sizeof(DATA_TYPE) * n * n, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(q, device_addrs.q_dev, sizeof(DATA_TYPE) * n * n, cudaMemcpyDeviceToHost));

    gpuErrchk(cudaFree(device_addrs.u_dev));
    gpuErrchk(cudaFree(device_addrs.v_dev));
    gpuErrchk(cudaFree(device_addrs.p_dev));
    gpuErrchk(cudaFree(device_addrs.q_dev));
}

__global__ void col_sweep(int tsteps, int n, DATA_TYPE *u, DATA_TYPE *v, DATA_TYPE *p, DATA_TYPE *q, DATA_TYPE a,
                          DATA_TYPE b, DATA_TYPE c, DATA_TYPE d, DATA_TYPE f) {
    for (int i = 1 + blockDim.x * blockIdx.x + threadIdx.x; i < _PB_N - 1; i += gridDim.x * blockDim.x) {
        DATA_TYPE prev_v = SCALAR_VAL(1.0);
        v[0 * n + i] = prev_v;
        DATA_TYPE prev_p = SCALAR_VAL(0.0);
        p[i * n + 0] = prev_p;
        DATA_TYPE prev_q = prev_v;
        q[i * n + 0] = prev_q;
        for (int j = 1; j < _PB_N - 1; j++) {
            double new_prev_p = -c / (a * prev_p + b);
            double new_prev_q = (-d * u[j * n + (i - 1)] + (SCALAR_VAL(1.0) + SCALAR_VAL(2.0) * d) * u[j * n + i] -
                                 f * u[j * n + (i + 1)] - a * prev_q) /
                                (a * prev_p + b);
            p[i * n + j] = new_prev_p;
            q[i * n + j] = new_prev_q;
            v[j * n + i] = new_prev_q;  // writing to v here means penalty for cache, but accessing v (no bank
                                        // conflicts) is much better than q.
            prev_p = new_prev_p;
            prev_q = new_prev_q;
        }

        prev_v = SCALAR_VAL(1.0);
        v[(_PB_N - 1) * n + i] = prev_v;
        for (int j = _PB_N - 2; j >= 1; j--) {
            prev_v = p[i * n + j] * prev_v + v[j * n + i];
            v[j * n + i] = prev_v;
        }
    }
}

__global__ void row_sweep(int tsteps, int n, DATA_TYPE *u, DATA_TYPE *v, DATA_TYPE *p, DATA_TYPE *q, DATA_TYPE a,
                          DATA_TYPE c, DATA_TYPE d, DATA_TYPE e, DATA_TYPE f) {
    for (int i = 1 + blockDim.x * blockIdx.x + threadIdx.x; i < _PB_N - 1; i += gridDim.x * blockDim.x) {
        u[i * n + 0] = SCALAR_VAL(1.0);
        DATA_TYPE prev_p = SCALAR_VAL(0.0);
        p[i * n + 0] = prev_p;
        DATA_TYPE prev_q = u[i * n + 0];
        q[i * n + 0] = prev_q;
        for (int j = 1; j < _PB_N - 1; j++) {
            DATA_TYPE new_prev_p = -f / (d * prev_p + e);
            DATA_TYPE new_prev_q = (-a * v[(i - 1) * n + j] + (SCALAR_VAL(1.0) + SCALAR_VAL(2.0) * a) * v[i * n + j] -
                                    c * v[(i + 1) * n + j] - d * prev_q) /
                                   (d * prev_p + e);
            p[i * n + j] = new_prev_p;
            q[i * n + j] = new_prev_q;
            prev_p = new_prev_p;
            prev_q = new_prev_q;
        }
        DATA_TYPE prev_u = SCALAR_VAL(1.0);
        u[i * n + (_PB_N - 1)] = prev_u;
        for (int j = _PB_N - 2; j >= 1; j--) {
            prev_u = p[i * n + j] * prev_u + q[i * n + j];
            u[i * n + j] = prev_u;
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

void kernel_adi(int tsteps, int n, DATA_TYPE POLYBENCH_2D(u, N2, N2, n, n), DATA_TYPE POLYBENCH_2D(v, N2, N2, n, n),
                DATA_TYPE POLYBENCH_2D(p, N2, N2, n, n), DATA_TYPE POLYBENCH_2D(q, N2, N2, n, n)) {
    kernel_adi_inner(tsteps, n, device_addrs.u_dev, device_addrs.v_dev, device_addrs.p_dev, device_addrs.q_dev);
    gpuErrchk(cudaDeviceSynchronize());
}
