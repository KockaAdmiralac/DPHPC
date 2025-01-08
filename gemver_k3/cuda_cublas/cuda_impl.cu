#include <assert.h>
#include <gemver.h>
#include <polybench.h>
#include <stdio.h>

#include <cuda_support.cuh>

#include "cublas_v2.h"

typedef struct {
    cublasStatus_t stat;
    cublasHandle_t handle;
    DATA_TYPE *A_dev;
    DATA_TYPE *u1_dev;
    DATA_TYPE *v1_dev;
    DATA_TYPE *u2_dev;
    DATA_TYPE *v2_dev;
    DATA_TYPE *w_dev;
    DATA_TYPE *x_dev;
    DATA_TYPE *y_dev;
    DATA_TYPE *z_dev;
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
    (void)alpha;
    (void)beta;
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

    gpuCublasErrchk(cublasCreate(&device_addrs.handle));
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

    gpuCublasErrchk(cublasDestroy(device_addrs.handle));
}

/* Main computational kernel. The whole function will be timed,
   including the call and return. */
void kernel_gemver(int n, DATA_TYPE alpha, DATA_TYPE beta, DATA_TYPE POLYBENCH_2D(A, N2, N2, n, n),
                   DATA_TYPE POLYBENCH_1D(u1, N2, n), DATA_TYPE POLYBENCH_1D(v1, N2, n),
                   DATA_TYPE POLYBENCH_1D(u2, N2, n), DATA_TYPE POLYBENCH_1D(v2, N2, n),
                   DATA_TYPE POLYBENCH_1D(w, N2, n), DATA_TYPE POLYBENCH_1D(x, N2, n), DATA_TYPE POLYBENCH_1D(y, N2, n),
                   DATA_TYPE POLYBENCH_1D(z, N2, n)) {
    // cublasDgemv isn't exactly the same as polybench gemver, Â first needs to be computed and passed as A.  Polybench
    // to cublas matching: ß->ɑ, y->x, z->ßy
    // cublasDgemm: C = ɑ op(A) op(B) + ßC
    //   op(A) mxk
    //   op(B) kxn
    //   C mxn
    // cublasDgemv: y = ɑ op(A) x + ßy
    //   op(A) mxn
    //   x nx1
    //   y mx1
    // steps:
    //   gemm on C=A, A=v1, B=u1, meaning A=A+u1.v1
    //   C mxn = N2xN2, B kxn = 1xN2, A mxk = N2x1
    //   gemm on C=A, A=v2, B=u2, meaning A=A+u2.v2, A at this point is gemver's Â output
    //   mxn = N2xN2, kxn = 1xN2, mxk = N2x1
    //   gemv on ɑ=ß, A=A, x=y, ß=1, y=z, z is now gemver's x output
    //   mxn = N2xN2, x is n, y is m
    //   gemv on ɑ=ɑ, A=A^T, x=z, ß=0, y=w, w is now gemver's w output

    const double cst1 = 1.0;
    const double cst0 = 0.0;

    gpuCublasErrchk(cublasDgemv_v2(device_addrs.handle, CUBLAS_OP_T, N2, N2, &alpha, device_addrs.A_dev, N2,
                                   device_addrs.x_dev, 1, &cst0, device_addrs.w_dev, 1));

    gpuErrchk(cudaDeviceSynchronize());
}
