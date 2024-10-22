#include <gemver.h>
#include <polybench.h>
#include <stdio.h>
#include <string.h>

#include <cuda_support.cuh>

void initialise_benchmark(int n, DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE POLYBENCH_2D(A, N2, N2, n, n),
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
    DATA_TYPE *A_dev = NULL;
    DATA_TYPE *u1_dev = NULL;
    DATA_TYPE *v1_dev = NULL;
    DATA_TYPE *u2_dev = NULL;
    DATA_TYPE *v2_dev = NULL;
    DATA_TYPE *w_dev = NULL;
    DATA_TYPE *x_dev = NULL;
    DATA_TYPE *y_dev = NULL;
    DATA_TYPE *z_dev = NULL;

    gpuErrchk(cudaMalloc(&A_dev, sizeof(DATA_TYPE) * n * n));
    gpuErrchk(cudaMalloc(&u1_dev, sizeof(DATA_TYPE) * n));
    gpuErrchk(cudaMalloc(&v1_dev, sizeof(DATA_TYPE) * n));
    gpuErrchk(cudaMalloc(&u2_dev, sizeof(DATA_TYPE) * n));
    gpuErrchk(cudaMalloc(&v2_dev, sizeof(DATA_TYPE) * n));
    gpuErrchk(cudaMalloc(&w_dev, sizeof(DATA_TYPE) * n));
    gpuErrchk(cudaMalloc(&x_dev, sizeof(DATA_TYPE) * n));
    gpuErrchk(cudaMalloc(&y_dev, sizeof(DATA_TYPE) * n));
    gpuErrchk(cudaMalloc(&z_dev, sizeof(DATA_TYPE) * n));

    gpuErrchk(cudaMemcpy(A_dev, A, sizeof(DATA_TYPE) * n * n, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(u1_dev, u1, sizeof(DATA_TYPE) * n, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(v1_dev, v1, sizeof(DATA_TYPE) * n, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(u2_dev, u2, sizeof(DATA_TYPE) * n, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(v2_dev, v2, sizeof(DATA_TYPE) * n, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(w_dev, w, sizeof(DATA_TYPE) * n, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(x_dev, x, sizeof(DATA_TYPE) * n, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(y_dev, y, sizeof(DATA_TYPE) * n, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(z_dev, z, sizeof(DATA_TYPE) * n, cudaMemcpyHostToDevice));

    kernel_big<<<1, 1>>>(n, alpha, beta, A_dev, u1_dev, v1_dev, u2_dev, v2_dev, w_dev, x_dev, y_dev, z_dev);

    cudaDeviceSynchronize();

    cudaMemcpy(w, w_dev, sizeof(DATA_TYPE) * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(A, A_dev, sizeof(DATA_TYPE) * n * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(x, x_dev, sizeof(DATA_TYPE) * n, cudaMemcpyDeviceToHost);

    cudaFree(A_dev);
    cudaFree(u1_dev);
    cudaFree(v1_dev);
    cudaFree(u2_dev);
    cudaFree(v2_dev);
    cudaFree(w_dev);
    cudaFree(x_dev);
    cudaFree(y_dev);
    cudaFree(z_dev);
}