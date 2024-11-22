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

    device_addrs.tpb_1.x = 1024;
    device_addrs.bpg_1.x = max_threads_for_gpu / device_addrs.tpb_1.x;

    device_addrs.tpb_2.x = tpb;
    device_addrs.bpg_2.x = max_threads_for_gpu / device_addrs.tpb_2.x;

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

__global__ void kernel_1(int n, DATA_TYPE *A, DATA_TYPE *u1, DATA_TYPE *v1, DATA_TYPE *u2, DATA_TYPE *v2) {
    const int smem_capacity = 48 * 1024;
    const int total_smem_slots = smem_capacity / sizeof(DATA_TYPE);
    const int smem_slots_per_arr = total_smem_slots / 2;
    __shared__ DATA_TYPE smem_cache_u1[smem_slots_per_arr];
    __shared__ DATA_TYPE smem_cache_u2[smem_slots_per_arr];
    for (int j_block = 0; j_block < n; j_block += smem_slots_per_arr) {
        for (unsigned int my_slot = threadIdx.x; my_slot < smem_slots_per_arr && j_block + my_slot < n;
             my_slot += blockDim.x) {
            smem_cache_u1[my_slot] = u1[j_block + my_slot];
            smem_cache_u2[my_slot] = u2[j_block + my_slot];
        }
        __syncthreads();

        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
            // int j_slot_with_thread_offset = threadIdx.x%4;
            for (int j_slot = 0; j_block + j_slot < n && j_slot < smem_slots_per_arr;
                 j_slot++) {  // no longer bothering with 2d thread ids because makes shared memory more complicated,
                              // and
                              // threads/memory ratio means n^2 threads is unrealistic anyway.
                int j = j_block + j_slot;

                // int j_slot_with_thread_offset = (j_slot + threadIdx.x) % smem_slots_per_arr;
                // int j_slot_with_thread_offset = threadIdx.x<blockDim.x/2 ? (j_slot % smem_slots_per_arr) :
                // ((smem_slots_per_arr-j_slot) % smem_slots_per_arr);
                int j_slot_with_thread_offset = j_slot;

                int j_with_thread_offset = j_block + j_slot_with_thread_offset;
                A[j * n + i] = A[j * n + i] + smem_cache_u1[j_slot] * v1[i] + smem_cache_u2[j_slot] * v2[i];
                // A[j * n + i] = A[j * n + i] + smem_cache_u1[j_slot] * v1[i] + u2[j] * v2[i];
                // A[j_with_thread_offset * n + i] = A[j_with_thread_offset * n + i] +
                // smem_cache_u1[j_slot_with_thread_offset] * v1[i] + smem_cache_u2[j_slot_with_thread_offset] * v2[i];
                // A[j * n + i] = A[j * n + i] + u1[j] * v1[i] + u2[j] * v2[i];

                // j_slot_with_thread_offset++;
                // if (j_block + j_slot_with_thread_offset>=n || j_slot_with_thread_offset >= smem_slots_per_arr) {
                //     j_slot_with_thread_offset=0;
                // }
            }
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
    kernel_1<<<device_addrs.bpg_1, device_addrs.tpb_1>>>(n, device_addrs.A_dev, device_addrs.u1_dev,
                                                         device_addrs.v1_dev, device_addrs.u2_dev, device_addrs.v2_dev);
    kernel_2<<<device_addrs.bpg_2, device_addrs.tpb_2>>>(n, beta, device_addrs.A_dev, device_addrs.x_dev,
                                                         device_addrs.y_dev, device_addrs.z_dev);
    kernel_3<<<device_addrs.bpg_3, device_addrs.tpb_3>>>(n, alpha, device_addrs.A_dev, device_addrs.w_dev,
                                                         device_addrs.x_dev);

    gpuErrchk(cudaDeviceSynchronize());
}
