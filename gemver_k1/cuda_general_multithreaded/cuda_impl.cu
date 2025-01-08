#include <gemver.h>
#include <polybench.h>
#include <stdio.h>

#include <cuda_support.cuh>

typedef struct {
    DATA_TYPE *A_dev;
    DATA_TYPE *u1_dev;
    DATA_TYPE *v1_dev;
    DATA_TYPE *u2_dev;
    DATA_TYPE *v2_dev;
    DATA_TYPE *w_dev;
    DATA_TYPE *x_dev;
    DATA_TYPE *y_dev;
    DATA_TYPE *z_dev;
    dim3 tpb_1;
    dim3 bpg_1;
    dim3 tpb_2;
    dim3 bpg_2;
    dim3 tpb_3;
    dim3 bpg_3;
} kernel_init_t;

kernel_init_t device_addrs;
void initialise_benchmark(int argc, char **argv, int n, DATA_TYPE *alpha, DATA_TYPE *beta,
                          DATA_TYPE POLYBENCH_2D(A, N2, N2, n, n), DATA_TYPE POLYBENCH_1D(u1, N2, n),
                          DATA_TYPE POLYBENCH_1D(v1, N2, n), DATA_TYPE POLYBENCH_1D(u2, N2, n),
                          DATA_TYPE POLYBENCH_1D(v2, N2, n), DATA_TYPE POLYBENCH_1D(w, N2, n),
                          DATA_TYPE POLYBENCH_1D(x, N2, n), DATA_TYPE POLYBENCH_1D(y, N2, n),
                          DATA_TYPE POLYBENCH_1D(z, N2, n)) {
    (void)alpha;
    (void)beta;
    (void)argc;
    (void)argv;
    gpuErrchk(cudaMalloc(&device_addrs.A_dev, sizeof(DATA_TYPE) * n * n));
    gpuErrchk(cudaMalloc(&device_addrs.u1_dev, sizeof(DATA_TYPE) * n));
    gpuErrchk(cudaMalloc(&device_addrs.v1_dev, sizeof(DATA_TYPE) * n));
    gpuErrchk(cudaMalloc(&device_addrs.u2_dev, sizeof(DATA_TYPE) * n));
    gpuErrchk(cudaMalloc(&device_addrs.v2_dev, sizeof(DATA_TYPE) * n));
    gpuErrchk(cudaMalloc(&device_addrs.w_dev, sizeof(DATA_TYPE) * n));
    gpuErrchk(cudaMalloc(&device_addrs.x_dev, sizeof(DATA_TYPE) * n));
    gpuErrchk(cudaMalloc(&device_addrs.y_dev, sizeof(DATA_TYPE) * n));
    gpuErrchk(cudaMalloc(&device_addrs.z_dev, sizeof(DATA_TYPE) * n));

    cudaMemcpy(device_addrs.A_dev, A, sizeof(DATA_TYPE) * n * n, cudaMemcpyHostToDevice);
    cudaMemcpy(device_addrs.u1_dev, u1, sizeof(DATA_TYPE) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(device_addrs.v1_dev, v1, sizeof(DATA_TYPE) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(device_addrs.u2_dev, u2, sizeof(DATA_TYPE) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(device_addrs.v2_dev, v2, sizeof(DATA_TYPE) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(device_addrs.w_dev, w, sizeof(DATA_TYPE) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(device_addrs.x_dev, x, sizeof(DATA_TYPE) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(device_addrs.y_dev, y, sizeof(DATA_TYPE) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(device_addrs.z_dev, z, sizeof(DATA_TYPE) * n, cudaMemcpyHostToDevice);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    int sm_count = deviceProp.multiProcessorCount;
    int max_threads_per_sm = deviceProp.maxThreadsPerMultiProcessor;
    int max_threads_for_gpu = sm_count * max_threads_per_sm;

#ifndef TPB
    unsigned int tpb = 64;
#else
    unsigned int tpb = TPB;
#endif

    device_addrs.tpb_1.x = tpb;
    device_addrs.bpg_1.x = max_threads_for_gpu / device_addrs.tpb_1.x;

    device_addrs.tpb_2.x = tpb;
    device_addrs.bpg_2.x = max_threads_for_gpu / device_addrs.tpb_2.x;

    device_addrs.tpb_3.x = tpb;
    device_addrs.bpg_3.x = max_threads_for_gpu / device_addrs.tpb_3.x;

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

    gpuErrchk(cudaMalloc(&device_addrs.A_dev, sizeof(DATA_TYPE) * n * n));
    gpuErrchk(cudaMalloc(&device_addrs.u1_dev, sizeof(DATA_TYPE) * n));
    gpuErrchk(cudaMalloc(&device_addrs.v1_dev, sizeof(DATA_TYPE) * n));
    gpuErrchk(cudaMalloc(&device_addrs.u2_dev, sizeof(DATA_TYPE) * n));
    gpuErrchk(cudaMalloc(&device_addrs.v2_dev, sizeof(DATA_TYPE) * n));
    gpuErrchk(cudaMalloc(&device_addrs.w_dev, sizeof(DATA_TYPE) * n));
    gpuErrchk(cudaMalloc(&device_addrs.x_dev, sizeof(DATA_TYPE) * n));
    gpuErrchk(cudaMalloc(&device_addrs.y_dev, sizeof(DATA_TYPE) * n));
    gpuErrchk(cudaMalloc(&device_addrs.z_dev, sizeof(DATA_TYPE) * n));

    gpuErrchk(cudaMemcpy(device_addrs.A_dev, A, sizeof(DATA_TYPE) * n * n, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(device_addrs.u1_dev, u1, sizeof(DATA_TYPE) * n, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(device_addrs.v1_dev, v1, sizeof(DATA_TYPE) * n, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(device_addrs.u2_dev, u2, sizeof(DATA_TYPE) * n, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(device_addrs.v2_dev, v2, sizeof(DATA_TYPE) * n, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(device_addrs.w_dev, w, sizeof(DATA_TYPE) * n, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(device_addrs.x_dev, x, sizeof(DATA_TYPE) * n, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(device_addrs.y_dev, y, sizeof(DATA_TYPE) * n, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(device_addrs.z_dev, z, sizeof(DATA_TYPE) * n, cudaMemcpyHostToDevice));

    gpuErrchk(cudaDeviceSynchronize());
}

void finish_benchmark(int n, DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE POLYBENCH_2D(A, N2, N2, n, n),
                      DATA_TYPE POLYBENCH_1D(u1, N2, n), DATA_TYPE POLYBENCH_1D(v1, N2, n),
                      DATA_TYPE POLYBENCH_1D(u2, N2, n), DATA_TYPE POLYBENCH_1D(v2, N2, n),
                      DATA_TYPE POLYBENCH_1D(w, N2, n), DATA_TYPE POLYBENCH_1D(x, N2, n),
                      DATA_TYPE POLYBENCH_1D(y, N2, n), DATA_TYPE POLYBENCH_1D(z, N2, n)) {
    (void)n;
    (void)alpha;
    (void)beta;
    (void)u1;
    (void)v1;
    (void)u2;
    (void)v2;
    (void)y;
    (void)z;

    cudaMemcpy(w, device_addrs.w_dev, sizeof(DATA_TYPE) * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(A, device_addrs.A_dev, sizeof(DATA_TYPE) * n * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(x, device_addrs.x_dev, sizeof(DATA_TYPE) * n, cudaMemcpyDeviceToHost);

    cudaFree(device_addrs.A_dev);
    cudaFree(device_addrs.u1_dev);
    cudaFree(device_addrs.v1_dev);
    cudaFree(device_addrs.u2_dev);
    cudaFree(device_addrs.v2_dev);
    cudaFree(device_addrs.w_dev);
    cudaFree(device_addrs.x_dev);
    cudaFree(device_addrs.y_dev);
    cudaFree(device_addrs.z_dev);
}

__global__ void kernel_1(int n, DATA_TYPE *A, DATA_TYPE *u1, DATA_TYPE *v1, DATA_TYPE *u2, DATA_TYPE *v2) {
    for (int i = blockIdx.y * blockDim.y + threadIdx.y; i < n; i += blockDim.y * gridDim.y) {
        for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n; j += blockDim.x * gridDim.x) {
            A[i * n + j] = A[i * n + j] + u1[i] * v1[j] + u2[i] * v2[j];
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
    kernel_1<<<device_addrs.bpg_1, device_addrs.tpb_1>>>(n, device_addrs.A_dev, device_addrs.u1_dev,
                                                         device_addrs.v1_dev, device_addrs.u2_dev, device_addrs.v2_dev);

    gpuErrchk(cudaDeviceSynchronize());
}
