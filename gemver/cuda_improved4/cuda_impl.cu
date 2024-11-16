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

__global__ void kernel_1(int n, DATA_TYPE *A, DATA_TYPE *u1, DATA_TYPE *v1, DATA_TYPE *u2, DATA_TYPE *v2) {
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n; j += blockDim.x * gridDim.x) {
        for (int i = blockIdx.y * blockDim.y + threadIdx.y; i < n; i += blockDim.y * gridDim.y) {
            A[i * n + j] = A[i * n + j] + u1[i] * v1[j] + u2[i] * v2[j];
        }
    }
}

inline __device__ void own_gemv(int n, DATA_TYPE *out_arr, const DATA_TYPE scale_by, const DATA_TYPE *arr,
                                const DATA_TYPE *mult_vect, const DATA_TYPE *add_vect) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < _PB_N; i += gridDim.x * blockDim.x) {
#define KERNEL2_ACCS 4
        DATA_TYPE x_incs[KERNEL2_ACCS] = {0.0};
        int j;
        for (j = 0; j < _PB_N - KERNEL2_ACCS + 1; j += KERNEL2_ACCS) {
            // could tile accesses to A so they're not done as strided.
            for (int acc = 0; acc < KERNEL2_ACCS; acc++) {
                x_incs[acc] += arr[(j + acc) * n + i] * mult_vect[(j + acc)];
            }
        }
        for (; j < _PB_N; j++) {
            x_incs[0] += arr[j * n + i] * mult_vect[j];
        }
        for (int k = 1; k < KERNEL2_ACCS; k++) {
            x_incs[0] += x_incs[k];
        }
        out_arr[i] += scale_by * x_incs[0] + add_vect[i];
    }
}

__global__ void kernel_2(int n, const DATA_TYPE beta, const DATA_TYPE *A, DATA_TYPE *x, const DATA_TYPE *y,
                         const DATA_TYPE *z) {
    own_gemv(n, x, beta, A, y, z);
}

#define TPB3 512
__global__ void kernel_3(const int n, const DATA_TYPE alpha, const DATA_TYPE *A, DATA_TYPE *w, const DATA_TYPE *x) {
    // build shared memory with al threads, with enough rows so each thread has a few js.

    const int smem_capacity = 48 * 1024;
    // const int smem_cells = smem_capacity / sizeof(DATA_TYPE);
    // const int smem_cols_per_thread = smem_cells / blockDim.x;
    const int smem_cols_per_thread = 8;
    const int smem_cells = smem_cols_per_thread * TPB3;
    __shared__ DATA_TYPE smem_A[smem_cells];
    for (int row_block = blockIdx.x * blockDim.x; row_block < _PB_N; row_block += gridDim.x * blockDim.x) {
        int row = row_block + threadIdx.x;

        DATA_TYPE w_inc = 0.0;

        const int rows_in_block = _PB_N - row_block >= blockDim.x ? blockDim.x : _PB_N - row_block;

        for (int col_block = 0; col_block < _PB_N; col_block += smem_cols_per_thread) {
            // for (int slot_local_row = 0; slot_local_row < blockDim.x; slot_local_row++) {
            //     for (int slot_local_col = threadIdx.x; slot_local_col < smem_cols_per_thread;
            //          slot_local_col += blockDim.x) {
            //         int from_A_row = row_block + slot_local_row;
            //         int from_A_col = col_block + slot_local_col;
            //         if (from_A_row >= _PB_N || from_A_col >= _PB_N) {
            //             continue;
            //         }
            //         smem_A[slot_local_row * smem_cols_per_thread + slot_local_col] = A[from_A_row * n + from_A_col];
            //     }
            // }
            const int cols_in_block =
                _PB_N - col_block >= smem_cols_per_thread ? smem_cols_per_thread : _PB_N - col_block;
            const int cells_in_block = rows_in_block * cols_in_block;

            for (int my_slot = threadIdx.x; my_slot < cells_in_block; my_slot += blockDim.x) {
                int slot_local_col = my_slot % cols_in_block;
                int slot_local_row = (my_slot - slot_local_col) / cols_in_block;
                int from_A_row = row_block + slot_local_row;
                int from_A_col = col_block + slot_local_col;
                // if (from_A_row >= _PB_N || from_A_col >= _PB_N) {
                //     break;
                // }
                smem_A[slot_local_row * cols_in_block + slot_local_col] = A[from_A_row * n + from_A_col];
            }
            __syncthreads();

            if (row < n) {
                for (int j = 0; j < cols_in_block; j++) {
                    w_inc += smem_A[threadIdx.x * cols_in_block + j] * x[col_block + j];
                    // w_inc += A[(row_block+threadIdx.x)*n+col_block+j]*x[col_block+j];
                }
            }
        }
        // for (int j = 0; j<_PB_N; j++) {
        //     int my_j = (j+threadIdx.x)%_PB_N;
        //     w_inc += A[i*n+my_j]*x[my_j];
        // }
        w[row] += alpha * w_inc;
    }
}

void initialise_benchmark(int argc, char **argv, int n, DATA_TYPE *alpha, DATA_TYPE *beta,
                          DATA_TYPE POLYBENCH_2D(A, N2, N2, n, n), DATA_TYPE POLYBENCH_1D(u1, N2, n),
                          DATA_TYPE POLYBENCH_1D(v1, N2, n), DATA_TYPE POLYBENCH_1D(u2, N2, n),
                          DATA_TYPE POLYBENCH_1D(v2, N2, n), DATA_TYPE POLYBENCH_1D(w, N2, n),
                          DATA_TYPE POLYBENCH_1D(x, N2, n), DATA_TYPE POLYBENCH_1D(y, N2, n),
                          DATA_TYPE POLYBENCH_1D(z, N2, n)) {
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
    int i, j;

    *alpha = 1.5;
    *beta = 1.2;

    DATA_TYPE fn = (DATA_TYPE)n;

    for (i = 0; i < n; i++) {
        u1[i] = i;
        u2[i] = ((i + 1) / fn) / 2.0;
        v1[i] = ((i + 1) / fn) / 4.0;
        v2[i] = ((i + 1) / fn) / 6.0;
        y[i] = ((i + 1) / fn) / 8.0;
        z[i] = ((i + 1) / fn) / 9.0;
        x[i] = 0.0;
        w[i] = 0.0;
        for (j = 0; j < n; j++) A[i][j] = (DATA_TYPE)(i * j % n) / n;
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

    device_addrs.tpb_1.x = tpb;
    device_addrs.bpg_1.x = max_threads_for_gpu / device_addrs.tpb_1.x;

    device_addrs.tpb_2.x = tpb;
    device_addrs.bpg_2.x = max_threads_for_gpu / device_addrs.tpb_2.x;

    device_addrs.tpb_3.x = TPB3;
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