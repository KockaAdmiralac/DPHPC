#include <assert.h>
#include <gemver.h>
#include <polybench.h>
#include <stdio.h>

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
    *alpha = 1.5;
    *beta = 1.2;
    (void)argc;
    (void)argv;

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

    gpuErrchk(cudaMalloc(&kdata.A_dev, sizeof(DATA_TYPE) * n * n));
    gpuErrchk(cudaMalloc(&kdata.u1_dev, sizeof(DATA_TYPE) * n));
    gpuErrchk(cudaMalloc(&kdata.v1_dev, sizeof(DATA_TYPE) * n));
    gpuErrchk(cudaMalloc(&kdata.u2_dev, sizeof(DATA_TYPE) * n));
    gpuErrchk(cudaMalloc(&kdata.v2_dev, sizeof(DATA_TYPE) * n));
    gpuErrchk(cudaMalloc(&kdata.w_dev, sizeof(DATA_TYPE) * n));
    gpuErrchk(cudaMalloc(&kdata.x_dev, sizeof(DATA_TYPE) * n));
    gpuErrchk(cudaMalloc(&kdata.y_dev, sizeof(DATA_TYPE) * n));
    gpuErrchk(cudaMalloc(&kdata.z_dev, sizeof(DATA_TYPE) * n));

    cudaMemcpy(kdata.A_dev, A, sizeof(DATA_TYPE) * n * n, cudaMemcpyHostToDevice);
    cudaMemcpy(kdata.u1_dev, u1, sizeof(DATA_TYPE) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(kdata.v1_dev, v1, sizeof(DATA_TYPE) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(kdata.u2_dev, u2, sizeof(DATA_TYPE) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(kdata.v2_dev, v2, sizeof(DATA_TYPE) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(kdata.w_dev, w, sizeof(DATA_TYPE) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(kdata.x_dev, x, sizeof(DATA_TYPE) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(kdata.y_dev, y, sizeof(DATA_TYPE) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(kdata.z_dev, z, sizeof(DATA_TYPE) * n, cudaMemcpyHostToDevice);

    gpuErrchk(cudaDeviceSynchronize());
}

void finish_benchmark(int n, DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE POLYBENCH_2D(A, N2, N2, n, n),
                      DATA_TYPE POLYBENCH_1D(u1, N2, n), DATA_TYPE POLYBENCH_1D(v1, N2, n),
                      DATA_TYPE POLYBENCH_1D(u2, N2, n), DATA_TYPE POLYBENCH_1D(v2, N2, n),
                      DATA_TYPE POLYBENCH_1D(w, N2, n), DATA_TYPE POLYBENCH_1D(x, N2, n),
                      DATA_TYPE POLYBENCH_1D(y, N2, n), DATA_TYPE POLYBENCH_1D(z, N2, n)) {
    (void)alpha;
    (void)beta;
    (void)u1;
    (void)v1;
    (void)u2;
    (void)v2;
    (void)w;
    (void)y;
    (void)z;

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

__global__ void kernel_3(const int n, const DATA_TYPE alpha, const DATA_TYPE *A, DATA_TYPE *w, const DATA_TYPE *x) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < _PB_N; i += gridDim.x * blockDim.x) {
        DATA_TYPE loc = w[i];
        for (int j = 0; j < _PB_N; j++) {
            int jmod = (j + threadIdx.x % 32) % _PB_N;
            loc += alpha * A[i * n + jmod] * x[jmod];
        }
        w[i] = loc;
    }
}

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
void kernel_gemver(int n, DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE POLYBENCH_2D(A, N2, N2, n, n),
                   DATA_TYPE POLYBENCH_1D(u1, N2, n), DATA_TYPE POLYBENCH_1D(v1, N2, n),
                   DATA_TYPE POLYBENCH_1D(u2, N2, n), DATA_TYPE POLYBENCH_1D(v2, N2, n),
                   DATA_TYPE POLYBENCH_1D(w, N2, n), DATA_TYPE POLYBENCH_1D(x, N2, n), DATA_TYPE POLYBENCH_1D(y, N2, n),
                   DATA_TYPE POLYBENCH_1D(z, N2, n)) {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    int sm_count = deviceProp.multiProcessorCount;
    int max_threads_per_sm = deviceProp.maxThreadsPerMultiProcessor;
    int max_threads_for_gpu = sm_count * max_threads_per_sm;

    dim3 tpb1(64);
    int blocks_k1 = max_threads_for_gpu / tpb1.x;
    dim3 bpg1(blocks_k1);
    // printf("Using %d blocks_k1, max threads for GPU is %d, max threads per SM is %d\n", blocks_k1,
    // max_threads_for_gpu, max_threads_per_sm);

    dim3 tpb2(64);
    dim3 bpg2(max_threads_for_gpu / tpb2.x);
    dim3 tpb3(64);
    dim3 bpg3(max_threads_for_gpu / tpb3.x);
    kernel_3<<<bpg3, tpb3>>>(n, alpha, kdata.A_dev, kdata.w_dev, kdata.x_dev);

    cudaDeviceSynchronize();
}
