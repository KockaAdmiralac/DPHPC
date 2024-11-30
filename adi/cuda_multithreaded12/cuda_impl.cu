/* Include polybench common header. */
#include <cublas_v2.h>
#include <polybench.h>

/* Include benchmark-specific header. */
#include <cuda_support.cuh>

#include "adi.h"

typedef struct {
    default_adi_data_t adi_data;
    cublasStatus_t stat;
    cublasHandle_t handle;
    DATA_TYPE *u_dev;
    DATA_TYPE *v_dev;
    DATA_TYPE *p_dev;
    DATA_TYPE *q_dev;
    DATA_TYPE *spare_arr;
} cuda_adi_data_t;

void transpose_in_place(DATA_TYPE POLYBENCH_2D(arr, N2, N2, n, n)) {
    for (int i = 0; i < N2; i++) {
        for (int j = i + 1; j < N2; j++) {
            DATA_TYPE upper_right = arr[i][j];
            DATA_TYPE lower_left = arr[j][i];
            arr[j][i] = upper_right;
            arr[i][j] = lower_left;
        }
    }
}

void transpose_oop(cublasHandle_t handle, double *src_arr, double *dst_arr, int n) {
    double c1 = 1;
    double c0 = 0;
    gpuCublasErrchk(cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, n, n, &c1, src_arr, n, &c0, src_arr, n, dst_arr, n));
}

void copy_array_transposed(DATA_TYPE POLYBENCH_2D(arr, N2, N2, n, n), double *device_addr, cudaMemcpyKind dir) {
    if (dir == cudaMemcpyHostToDevice) {
        transpose_in_place(arr);
        gpuErrchk(cudaMemcpy(device_addr, arr, sizeof(DATA_TYPE) * N2 * N2, cudaMemcpyHostToDevice));
    } else {
        gpuErrchk(cudaMemcpy(arr, device_addr, sizeof(DATA_TYPE) * N2 * N2, cudaMemcpyDeviceToHost));
        gpuErrchk(cudaDeviceSynchronize());
        transpose_in_place(arr);
    }
}

void initialise_benchmark(int argc, char **argv, int tsteps, int n, void **gen_data_ptr) {
    *gen_data_ptr = polybench_alloc_data(1, sizeof(cuda_adi_data_t));
    cuda_adi_data_t *data_ptr = (cuda_adi_data_t *)(*gen_data_ptr);
    default_initialise_benchmark(argc, argv, tsteps, n, &data_ptr->adi_data);

    gpuErrchk(cudaMalloc(&data_ptr->u_dev, sizeof(DATA_TYPE) * n * n));
    gpuErrchk(cudaMalloc(&data_ptr->v_dev, sizeof(DATA_TYPE) * n * n));
    gpuErrchk(cudaMalloc(&data_ptr->p_dev, sizeof(DATA_TYPE) * n * n));
    gpuErrchk(cudaMalloc(&data_ptr->q_dev, sizeof(DATA_TYPE) * n * n));
    gpuErrchk(cudaMalloc(&data_ptr->spare_arr, sizeof(DATA_TYPE) * n * n));

    gpuCublasErrchk(cublasCreate(&data_ptr->handle));

    gpuErrchk(cudaMemcpy(data_ptr->u_dev, data_ptr->adi_data.u, sizeof(DATA_TYPE) * n * n, cudaMemcpyHostToDevice));
    copy_array_transposed(data_ptr->adi_data.v, data_ptr->v_dev, cudaMemcpyHostToDevice);
    copy_array_transposed(data_ptr->adi_data.p, data_ptr->p_dev, cudaMemcpyHostToDevice);
    copy_array_transposed(data_ptr->adi_data.q, data_ptr->q_dev, cudaMemcpyHostToDevice);
    gpuErrchk(cudaDeviceSynchronize());
}

void finish_benchmark(void *gen_data_ptr) {
    cuda_adi_data_t *data_ptr = (cuda_adi_data_t *)gen_data_ptr;
    int n = data_ptr->adi_data.n;
    gpuErrchk(cudaMemcpy(data_ptr->adi_data.u, data_ptr->u_dev, sizeof(DATA_TYPE) * n * n, cudaMemcpyDeviceToHost));
    copy_array_transposed(data_ptr->adi_data.v, data_ptr->v_dev, cudaMemcpyDeviceToHost);
    copy_array_transposed(data_ptr->adi_data.p, data_ptr->p_dev, cudaMemcpyDeviceToHost);
    copy_array_transposed(data_ptr->adi_data.q, data_ptr->q_dev, cudaMemcpyDeviceToHost);
}

void print_data(int argc, char **argv, int n, void *gen_data_ptr) {
    default_print_data(argc, argv, n, &((cuda_adi_data_t *)gen_data_ptr)->adi_data);
}

void free_data(void *gen_data_ptr) {
    cuda_adi_data_t *data_ptr = (cuda_adi_data_t *)gen_data_ptr;
    gpuErrchk(cudaFree(data_ptr->u_dev));
    gpuErrchk(cudaFree(data_ptr->v_dev));
    gpuErrchk(cudaFree(data_ptr->p_dev));
    gpuErrchk(cudaFree(data_ptr->q_dev));
    gpuErrchk(cudaFree(data_ptr->spare_arr));
    gpuCublasErrchk(cublasDestroy(data_ptr->handle));
    free(gen_data_ptr);
}

__global__ void zero_out_arr_firstlast(int n, double *out_arr) {
    for (int i = 1 + blockDim.x * blockIdx.x + threadIdx.x; i < _PB_N - 1; i += gridDim.x * blockDim.x) {
        out_arr[0 * n + i] = SCALAR_VAL(1.0);
        out_arr[(_PB_N - 1) * n + i] = SCALAR_VAL(1.0);
    }
}

inline __device__ void thread_sweep_footer(double *out_arr, int n, int i, double *p, double *q) {
    DATA_TYPE prev_out = SCALAR_VAL(1.0);
    for (int j = _PB_N - 2; j >= 1; j--) {
        prev_out = p[j * n + i] * prev_out + q[j * n + i];
        out_arr[j * n + i] = prev_out;
    }
}

#define SWEEP_TPB 1024
inline __device__ void sweep_generic(int n, const double *in_arr, double *out_arr, double *p, double *q, double v1,
                                     double v2, double v3, double v4, double v5) {
    // col_sweep: in_arr = u, out_arr = v, v1=a, v2=b, v3=c, v4=d, v5=f
    // row_sweep: in_arr = v, out_arr = u, v1=d, v2=e, v3=f, v4=a, v5=c

    /*
    min bandwidth estimates per timestep:
    load in_arr (N^2)
    store p and q (each N^2)
    load p and q (each N^2)
    store out_arr (N^2)
    */

    const int threads_in_block = SWEEP_TPB;
    const int total_smem_slots = 48 * 1024 / sizeof(DATA_TYPE);
    const int js_at_time = total_smem_slots / threads_in_block;
    // const int js_at_time = 8;

    __shared__ DATA_TYPE smem_in_arr[threads_in_block * js_at_time];

    int base_iter = 1;
    int end_iter_excl = _PB_N - 1;
    for (int i = base_iter + blockDim.x * blockIdx.x + threadIdx.x; i < end_iter_excl; i += gridDim.x * blockDim.x) {
        int block_start_i = i - ((i - base_iter) % SWEEP_TPB);
        int block_end_i = block_start_i + SWEEP_TPB;
        int active_threads_this_block = block_end_i >= end_iter_excl ? end_iter_excl - block_start_i : SWEEP_TPB;

        DATA_TYPE prev_p = SCALAR_VAL(0.0);
        p[0 * n + i] = prev_p;
        DATA_TYPE prev_q = SCALAR_VAL(1.0);
        q[0 * n + i] = prev_q;

        // copy in_arr for next few iterations into smem

        int init_j = 1;
        int j_ctr = 0;
        for (int j = init_j; j < _PB_N - 1; j++) {
            if (j_ctr == 0 || j_ctr == js_at_time) {
                const int js_to_load = _PB_N - 1 - j >= js_at_time ? js_at_time : _PB_N - 1 - j;
                __syncthreads();  // avoid overwriting a slowpoke's shared memory
// #define SMEM_ADDR(thread, j_temp) ((thread)*js_at_time+(j_temp))
#define SMEM_ADDR(thread, j_temp) ((j_temp) * SWEEP_TPB + (thread))
                for (int j_temp = 0; j_temp < js_to_load && j_temp < js_at_time; j_temp++) {
                    // smem_in_arr[j_temp * SWEEP_TPB + threadIdx.x] =
                    smem_in_arr[SMEM_ADDR(threadIdx.x, j_temp)] =
                        in_arr[(j + j_temp) * n + i];  // note this i is >= 1, not >= 0
                }
                j_ctr = 0;
                __syncthreads();  // needed to ensure my neighbour threads don't go ahead until my data for them is
                                  // available.
            }

            DATA_TYPE new_prev_p = -v3 / (v1 * prev_p + v2);

            // only edges of block will use global memory because it simplifies storage into shared memory
            DATA_TYPE inm1 = threadIdx.x == 0 ? in_arr[j * n + i - 1] : smem_in_arr[SMEM_ADDR(threadIdx.x - 1, j_ctr)];
            DATA_TYPE in0 = smem_in_arr[SMEM_ADDR(threadIdx.x, j_ctr)];
            DATA_TYPE inp1 = threadIdx.x == active_threads_this_block - 1
                                 ? in_arr[j * n + i + 1]
                                 : smem_in_arr[SMEM_ADDR(threadIdx.x + 1, j_ctr)];

            DATA_TYPE new_prev_q =
                (-v4 * inm1 + (SCALAR_VAL(1.0) + SCALAR_VAL(2.0) * v4) * in0 - v5 * inp1 - v1 * prev_q) /
                (v1 * prev_p + v2);

            p[j * n + i] = new_prev_p;
            q[j * n + i] = new_prev_q;
            prev_p = new_prev_p;
            prev_q = new_prev_q;

            j_ctr++;
        }

        thread_sweep_footer(out_arr, n, i, p, q);
    }
}

__global__ void col_sweep(int tsteps, int n, const DATA_TYPE *u, DATA_TYPE *v, DATA_TYPE *p, DATA_TYPE *q, DATA_TYPE a,
                          DATA_TYPE b, DATA_TYPE c, DATA_TYPE d, DATA_TYPE f) {
    sweep_generic(n, u, v, p, q, a, b, c, d, f);
}

__global__ void row_sweep(int tsteps, int n, DATA_TYPE *u, const DATA_TYPE *v, DATA_TYPE *p, DATA_TYPE *q, DATA_TYPE a,
                          DATA_TYPE c, DATA_TYPE d, DATA_TYPE e, DATA_TYPE f) {
    sweep_generic(n, v, u, p, q, d, e, f, a, c);
}

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
/* Based on a Fortran code fragment from Figure 5 of
 * "Automatic Data and Computation Decomposition on Distributed Memory Parallel
 * Computers" by Peizong Lee and Zvi Meir Kedem, TOPLAS, 2002
 */
void kernel_adi_inner(cublasHandle_t handle, int tsteps, int n, DATA_TYPE *u, DATA_TYPE *v, DATA_TYPE *p, DATA_TYPE *q,
                      DATA_TYPE *spare_arr) {
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

    int sms = get_device_multiprocessors(0);

    // should be able to zero out only once during program run

    for (int t = 1; t <= _PB_TSTEPS; t++) {
        // Column Sweep
        // Using very high block/thread count to give CUDA lots of independent work
        // threads have negligible memory needs so 1024/block makes sense.
        double *temp_v = spare_arr;
        // col_sweep writes to v but never reads from it, so can drop the old v entirely except first/last row.
        transpose_oop(handle, v, temp_v, n);
        zero_out_arr_firstlast<<<sms, 1024>>>(n, temp_v);
        col_sweep<<<sms, SWEEP_TPB>>>(tsteps, n, u, temp_v, p, q, a, b, c, d, f);
        transpose_oop(handle, temp_v, v, n);

        // Row Sweep
        double *temp_u = spare_arr;
        // row_sweep only writes to u but never reads so no need to initialise temp_u.
        // currently transposing whole u but only really care about first/last rows.
        transpose_oop(handle, u, temp_u, n);
        zero_out_arr_firstlast<<<sms, 1024>>>(n, temp_u);
        row_sweep<<<sms, SWEEP_TPB>>>(tsteps, n, temp_u, v, p, q, a, c, d, e, f);
        transpose_oop(handle, temp_u, u, n);
    }
}

void kernel_adi(void *gen_data_ptr) {
    cuda_adi_data_t *data_ptr = (cuda_adi_data_t *)gen_data_ptr;
    kernel_adi_inner(data_ptr->handle, data_ptr->adi_data.tsteps, data_ptr->adi_data.n, data_ptr->u_dev,
                     data_ptr->v_dev, data_ptr->p_dev, data_ptr->q_dev, data_ptr->spare_arr);
    gpuErrchk(cudaDeviceSynchronize());
}
