#include <assert.h>
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
    dim3 tpb_Ax;
    dim3 bpg_Ax;
    dim3 tpb_3;
    dim3 bpg_3;
} kernel_init_t;

kernel_init_t device_addrs;

void initialise_benchmark(int n, DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE POLYBENCH_2D(A, N2, N2, n, n),
                          DATA_TYPE POLYBENCH_1D(u1, N2, n), DATA_TYPE POLYBENCH_1D(v1, N2, n),
                          DATA_TYPE POLYBENCH_1D(u2, N2, n), DATA_TYPE POLYBENCH_1D(v2, N2, n),
                          DATA_TYPE POLYBENCH_1D(w, N2, n), DATA_TYPE POLYBENCH_1D(x, N2, n),
                          DATA_TYPE POLYBENCH_1D(y, N2, n), DATA_TYPE POLYBENCH_1D(z, N2, n)) {
    (void)alpha;
    (void)beta;
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

    device_addrs.tpb_Ax.x = tpb;
    device_addrs.bpg_Ax.x = 64 * max_threads_for_gpu / device_addrs.tpb_Ax.x;

    device_addrs.tpb_3.x = tpb;
    device_addrs.bpg_3.x = max_threads_for_gpu / device_addrs.tpb_3.x;
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

__global__ void kernel_Ax_combi(int n, const DATA_TYPE beta, DATA_TYPE *A, DATA_TYPE *u1, DATA_TYPE *v1, DATA_TYPE *u2,
                                DATA_TYPE *v2, DATA_TYPE *x, const DATA_TYPE *y, const DATA_TYPE *z

) {
    extern __shared__ DATA_TYPE all_shared[];
    DATA_TYPE *u1_s = all_shared;
    DATA_TYPE *u2_s = &all_shared[n];
    DATA_TYPE *y_s = &all_shared[2 * n];
    // note: shared memory is per-block, NOT grid-wide
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        u1_s[i] = u1[i];
        u2_s[i] = u2[i];
        y_s[i] = y[i];
    }
    __syncthreads();
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        DATA_TYPE x_loc = x[i];
        DATA_TYPE v1_loc = v1[i];
        DATA_TYPE v2_loc = v2[i];
        for (int j = 0; j < n; j++) {
            double A_loc = A[j * n + i];
            A_loc += u1_s[j] * v1_loc + u2_s[j] * v2_loc;
            x_loc = x_loc + beta * A_loc * y_s[j];
            A[j * n + i] = A_loc;
        }
        x[i] = x_loc + z[i];
    }
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
    kernel_Ax_combi<<<device_addrs.bpg_Ax, device_addrs.tpb_Ax, 3 * n * sizeof(DATA_TYPE)>>>(
        n, beta, device_addrs.A_dev, device_addrs.u1_dev, device_addrs.v1_dev, device_addrs.u2_dev, device_addrs.v2_dev,
        device_addrs.x_dev, device_addrs.y_dev, device_addrs.z_dev);
    kernel_3<<<device_addrs.bpg_3, device_addrs.tpb_3>>>(n, alpha, device_addrs.A_dev, device_addrs.w_dev,
                                                         device_addrs.x_dev);

    gpuErrchk(cudaDeviceSynchronize());
}