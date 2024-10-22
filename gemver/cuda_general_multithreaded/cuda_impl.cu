#include <gemver.h>
#include <polybench.h>
#include <stdio.h>

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

__global__ void kernel_1(int n, DATA_TYPE *A, DATA_TYPE *u1, DATA_TYPE *v1, DATA_TYPE *u2, DATA_TYPE *v2) {
    for (int i = blockIdx.y * blockDim.y + threadIdx.y; i < n; i += blockDim.y * gridDim.y) {
        for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n; j += blockDim.x * gridDim.x) {
            A[i * n + j] = A[i * n + j] + u1[i] * v1[j] + u2[i] * v2[j];
        }
    }
}

__global__ void kernel_2(int n, const DATA_TYPE beta, const DATA_TYPE *A, DATA_TYPE *x, const DATA_TYPE *y,
                         const DATA_TYPE *z) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < _PB_N; i += gridDim.x * blockDim.x) {
        for (int j = 0; j < _PB_N; j++) {
            // could tile accesses to A so they're not done as strided.
            x[i] = x[i] + beta * A[j * n + i] * y[j];
        }
        x[i] = x[i] + z[i];
    }
}

__global__ void kernel_3(const int n, const DATA_TYPE alpha, const DATA_TYPE *A, DATA_TYPE *w, const DATA_TYPE *x) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < _PB_N; i += gridDim.x * blockDim.x) {
        for (int j = 0; j < _PB_N; j++) {
            w[i] = w[i] + alpha * A[i * n + j] * x[j];
        }
    }
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

    dim3 tpb1(8, 8);
    dim3 bpg1(1, 2);
    dim3 tpb2(64);
    dim3 bpg2(4);
    dim3 tpb3(64);
    dim3 bpg3(4);
    kernel_1<<<bpg1, tpb1>>>(n, A_dev, u1_dev, v1_dev, u2_dev, v2_dev);
    kernel_2<<<bpg2, tpb2>>>(n, beta, A_dev, x_dev, y_dev, z_dev);
    kernel_3<<<bpg3, tpb3>>>(n, alpha, A_dev, w_dev, x_dev);

    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaMemcpy(w, w_dev, sizeof(DATA_TYPE) * n, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(A, A_dev, sizeof(DATA_TYPE) * n * n, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(x, x_dev, sizeof(DATA_TYPE) * n, cudaMemcpyDeviceToHost));

    gpuErrchk(cudaFree(A_dev));
    gpuErrchk(cudaFree(u1_dev));
    gpuErrchk(cudaFree(v1_dev));
    gpuErrchk(cudaFree(u2_dev));
    gpuErrchk(cudaFree(v2_dev));
    gpuErrchk(cudaFree(w_dev));
    gpuErrchk(cudaFree(x_dev));
    gpuErrchk(cudaFree(y_dev));
    gpuErrchk(cudaFree(z_dev));
}