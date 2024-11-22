#include <gemver.h>
#include <polybench.h>
#include <stdio.h>
#include <string.h>

#include <cuda_support.cuh>

typedef struct {
    DATA_TYPE *A_dev = NULL;
    DATA_TYPE *u1_dev = NULL;
    DATA_TYPE *v1_dev = NULL;
    DATA_TYPE *u2_dev = NULL;
    DATA_TYPE *v2_dev = NULL;
    DATA_TYPE *w_dev = NULL;
    DATA_TYPE *x_dev = NULL;
    DATA_TYPE *y_dev = NULL;
    DATA_TYPE *z_dev = NULL;
} kernel_data_t;

kernel_data_t kdata;

void initialise_benchmark(int argc, char **argv, int n, DATA_TYPE *alpha, DATA_TYPE *beta,
                          DATA_TYPE POLYBENCH_2D(A, N2, N2, n, n), DATA_TYPE POLYBENCH_1D(u1, N2, n),
                          DATA_TYPE POLYBENCH_1D(v1, N2, n), DATA_TYPE POLYBENCH_1D(u2, N2, n),
                          DATA_TYPE POLYBENCH_1D(v2, N2, n), DATA_TYPE POLYBENCH_1D(w, N2, n),
                          DATA_TYPE POLYBENCH_1D(x, N2, n), DATA_TYPE POLYBENCH_1D(y, N2, n),
                          DATA_TYPE POLYBENCH_1D(z, N2, n)) {
    (void)argc;
    (void)argv;

    *alpha = 1.5;
    *beta = 1.2;

    DATA_TYPE fn = (DATA_TYPE)n;

    for (int i = 0; i < n; i++) {
        u1[i] = i;
        u2[i] = ((i + 1) / fn) / 2.0;
        v1[i] = ((i + 1) / fn) / 4.0;
        v2[i] = ((i + 1) / fn) / 6.0;
        y[i] = ((i + 1) / fn) / 8.0;
        z[i] = ((i + 1) / fn) / 9.0;
        x[i] = 0.0;
        w[i] = 0.0;
        for (int j = 0; j < n; j++) A[i][j] = (DATA_TYPE)(i * j % n) / n;
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
}

__global__ void kernel_big(int n, DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE *A, DATA_TYPE *u1, DATA_TYPE *v1,
                           DATA_TYPE *u2, DATA_TYPE *v2, DATA_TYPE *w, DATA_TYPE *x, DATA_TYPE *y, DATA_TYPE *z) {
    int i, j;

    for (i = 0; i < _PB_N; i++)
        for (j = 0; j < _PB_N; j++) A[i * n + j] = A[i * n + j] + u1[i] * v1[j] + u2[i] * v2[j];

    for (i = 0; i < _PB_N; i++) {
        for (j = 0; j < _PB_N; j++) x[i] = x[i] + beta * A[j * n + i] * y[j];
        x[i] = x[i] + z[i];
    }

    for (i = 0; i < _PB_N; i++)
        for (j = 0; j < _PB_N; j++) w[i] = w[i] + alpha * A[i * n + j] * x[j];
}

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
void kernel_gemver(int n, DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE POLYBENCH_2D(A, N2, N2, n, n),
                   DATA_TYPE POLYBENCH_1D(u1, N2, n), DATA_TYPE POLYBENCH_1D(v1, N2, n),
                   DATA_TYPE POLYBENCH_1D(u2, N2, n), DATA_TYPE POLYBENCH_1D(v2, N2, n),
                   DATA_TYPE POLYBENCH_1D(w, N2, n), DATA_TYPE POLYBENCH_1D(x, N2, n), DATA_TYPE POLYBENCH_1D(y, N2, n),
                   DATA_TYPE POLYBENCH_1D(z, N2, n)) {
    gpuErrchk(cudaMalloc(&kdata.A_dev, sizeof(DATA_TYPE) * n * n));
    gpuErrchk(cudaMalloc(&kdata.u1_dev, sizeof(DATA_TYPE) * n));
    gpuErrchk(cudaMalloc(&kdata.v1_dev, sizeof(DATA_TYPE) * n));
    gpuErrchk(cudaMalloc(&kdata.u2_dev, sizeof(DATA_TYPE) * n));
    gpuErrchk(cudaMalloc(&kdata.v2_dev, sizeof(DATA_TYPE) * n));
    gpuErrchk(cudaMalloc(&kdata.w_dev, sizeof(DATA_TYPE) * n));
    gpuErrchk(cudaMalloc(&kdata.x_dev, sizeof(DATA_TYPE) * n));
    gpuErrchk(cudaMalloc(&kdata.y_dev, sizeof(DATA_TYPE) * n));
    gpuErrchk(cudaMalloc(&kdata.z_dev, sizeof(DATA_TYPE) * n));

    gpuErrchk(cudaMemcpy(kdata.A_dev, A, sizeof(DATA_TYPE) * n * n, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(kdata.u1_dev, u1, sizeof(DATA_TYPE) * n, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(kdata.v1_dev, v1, sizeof(DATA_TYPE) * n, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(kdata.u2_dev, u2, sizeof(DATA_TYPE) * n, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(kdata.v2_dev, v2, sizeof(DATA_TYPE) * n, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(kdata.w_dev, w, sizeof(DATA_TYPE) * n, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(kdata.x_dev, x, sizeof(DATA_TYPE) * n, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(kdata.y_dev, y, sizeof(DATA_TYPE) * n, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(kdata.z_dev, z, sizeof(DATA_TYPE) * n, cudaMemcpyHostToDevice));

    kernel_big<<<1, 1>>>(n, alpha, beta, kdata.A_dev, kdata.u1_dev, kdata.v1_dev, kdata.u2_dev, kdata.v2_dev,
                         kdata.w_dev, kdata.x_dev, kdata.y_dev, kdata.z_dev);

    cudaDeviceSynchronize();

    cudaMemcpy(w, kdata.w_dev, sizeof(DATA_TYPE) * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(A, kdata.A_dev, sizeof(DATA_TYPE) * n * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(x, kdata.x_dev, sizeof(DATA_TYPE) * n, cudaMemcpyDeviceToHost);

    cudaFree(kdata.A_dev);
    cudaFree(kdata.u1_dev);
    cudaFree(kdata.v1_dev);
    cudaFree(kdata.u2_dev);
    cudaFree(kdata.v2_dev);
    cudaFree(kdata.w_dev);
    cudaFree(kdata.x_dev);
    cudaFree(kdata.y_dev);
    cudaFree(kdata.z_dev);
}
