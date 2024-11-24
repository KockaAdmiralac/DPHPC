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

__global__ void col_sweep(int tsteps, int n, DATA_TYPE *u, DATA_TYPE *v, DATA_TYPE *p, DATA_TYPE *q, DATA_TYPE a,
                          DATA_TYPE b, DATA_TYPE c, DATA_TYPE d, DATA_TYPE f) {
    for (int i = 1 + blockDim.x * blockIdx.x + threadIdx.x; i < _PB_N - 1; i += gridDim.x * blockDim.x) {
        DATA_TYPE prev_v = SCALAR_VAL(1.0);
        v[0 * n + i] = prev_v;
        DATA_TYPE prev_p = SCALAR_VAL(0.0);
        p[0 * n + i] = prev_p;
        DATA_TYPE prev_q = prev_v;
        q[0 * n + i] = prev_q;
        for (int j = 1; j < _PB_N - 1; j++) {
            DATA_TYPE new_prev_p = -c / (a * prev_p + b);
            DATA_TYPE um1 = u[j * n + (i - 1)];
            DATA_TYPE u0 = u[j * n + i];
            DATA_TYPE up1 = u[j * n + (i + 1)];

            DATA_TYPE new_prev_q =
                (-d * um1 + (SCALAR_VAL(1.0) + SCALAR_VAL(2.0) * d) * u0 - f * up1 - a * prev_q) / (a * prev_p + b);

            p[j * n + i] = new_prev_p;
            q[j * n + i] = new_prev_q;
            prev_p = new_prev_p;
            prev_q = new_prev_q;
        }

        prev_v = SCALAR_VAL(1.0);
        v[(_PB_N - 1) * n + i] = prev_v;
        for (int j = _PB_N - 2; j >= 1; j--) {
            prev_v = p[j * n + i] * prev_v + q[j * n + i];
            v[j * n + i] = prev_v;
        }
    }
}

__global__ void row_sweep(int tsteps, int n, DATA_TYPE *u, DATA_TYPE *v, DATA_TYPE *p, DATA_TYPE *q, DATA_TYPE a,
                          DATA_TYPE c, DATA_TYPE d, DATA_TYPE e, DATA_TYPE f) {
    for (int i = 1 + blockDim.x * blockIdx.x + threadIdx.x; i < _PB_N - 1; i += gridDim.x * blockDim.x) {
        u[0 * n + i] = SCALAR_VAL(1.0);
        DATA_TYPE prev_p = SCALAR_VAL(0.0);
        p[0 * n + i] = prev_p;
        DATA_TYPE prev_q = u[0 * n + i];
        q[0 * n + i] = prev_q;
        for (int j = 1; j < _PB_N - 1; j++) {
            DATA_TYPE new_prev_p = -f / (d * prev_p + e);
            DATA_TYPE vm1 = v[j * n + (i - 1)];
            DATA_TYPE v0 = v[j * n + i];
            DATA_TYPE vp1 = v[j * n + (i + 1)];

            DATA_TYPE new_prev_q =
                (-a * vm1 + (SCALAR_VAL(1.0) + SCALAR_VAL(2.0) * a) * v0 - c * vp1 - d * prev_q) / (d * prev_p + e);

            p[j * n + i] = new_prev_p;
            q[j * n + i] = new_prev_q;
            prev_p = new_prev_p;
            prev_q = new_prev_q;
        }
        DATA_TYPE prev_u = SCALAR_VAL(1.0);
        u[(_PB_N - 1) * n + i] = prev_u;
        for (int j = _PB_N - 2; j >= 1; j--) {
            prev_u = p[j * n + i] * prev_u + q[j * n + i];
            u[j * n + i] = prev_u;
        }
    }
}

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
/* Based on a Fortran code fragment from Figure 5 of
 * "Automatic Data and Computation Decomposition on Distributed Memory Parallel
 * Computers" by Peizong Lee and Zvi Meir Kedem, TOPLAS, 2002
 */
void kernel_adi_inner(cuda_adi_data_t *data_ptr, int tsteps, int n, DATA_TYPE *u, DATA_TYPE *v, DATA_TYPE *p,
                      DATA_TYPE *q) {
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
    for (int t = 1; t <= _PB_TSTEPS; t++) {
        // Column Sweep
        // Using very high block/thread count to give CUDA lots of independent work
        // threads have negligible memory needs so 1024/block makes sense.
        double *temp_v = data_ptr->spare_arr;
        // col_sweep writes to v but never reads from it, so can drop the old v entirely
        transpose_oop(data_ptr->handle, v, temp_v,
                      n);  // for some reason transpose_oop is needed before sweep else a race condition occurs
        col_sweep<<<sms, 1024>>>(tsteps, n, u, temp_v, p, q, a, b, c, d, f);
        transpose_oop(data_ptr->handle, temp_v, v, n);
        // Row Sweep
        double *temp_u = data_ptr->spare_arr;
        // row_sweep only writes to u but never reads so no need to initialise temp_u.
        transpose_oop(data_ptr->handle, u, temp_u, n);
        row_sweep<<<sms, 1024>>>(tsteps, n, temp_u, v, p, q, a, c, d, e, f);
        transpose_oop(data_ptr->handle, temp_u, u, n);
    }
}

void kernel_adi(void *gen_data_ptr) {
    cuda_adi_data_t *data_ptr = (cuda_adi_data_t *)gen_data_ptr;
    kernel_adi_inner(data_ptr, data_ptr->adi_data.tsteps, data_ptr->adi_data.n, data_ptr->u_dev, data_ptr->v_dev,
                     data_ptr->p_dev, data_ptr->q_dev);
    gpuErrchk(cudaDeviceSynchronize());
}
