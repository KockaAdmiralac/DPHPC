#include <assert.h>
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
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n; j += blockDim.x * gridDim.x) {
        for (int i = 0; i < n; i++) {
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

    cudaMemcpy(A_dev, A, sizeof(DATA_TYPE) * n * n, cudaMemcpyHostToDevice);
    cudaMemcpy(u1_dev, u1, sizeof(DATA_TYPE) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(v1_dev, v1, sizeof(DATA_TYPE) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(u2_dev, u2, sizeof(DATA_TYPE) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(v2_dev, v2, sizeof(DATA_TYPE) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(w_dev, w, sizeof(DATA_TYPE) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(x_dev, x, sizeof(DATA_TYPE) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(y_dev, y, sizeof(DATA_TYPE) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(z_dev, z, sizeof(DATA_TYPE) * n, cudaMemcpyHostToDevice);

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
    kernel_Ax_combi<<<bpg1, tpb1, 3 * n * sizeof(DATA_TYPE)>>>(n, beta, A_dev, u1_dev, v1_dev, u2_dev, v2_dev, x_dev,
                                                               y_dev, z_dev);
    // kernel_1<<<bpg1, tpb1>>>(n, A_dev, u1_dev, v1_dev, u2_dev, v2_dev);
    // kernel_2<<<bpg2, tpb2>>>(n, beta, A_dev, x_dev, y_dev, z_dev);
    kernel_3<<<bpg3, tpb3>>>(n, alpha, A_dev, w_dev, x_dev);

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