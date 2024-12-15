/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include <mpi.h>
#include <stdio.h>
#include <string.h>

#include "adi.h"

// #define PRINT_DEBUG

/*
To split threads' print debug outputs:
def spl(r):
        a = []
        b = []
        last_found = 'a'
        s = r.splitlines()
        for l in s:
                if l.startswith('[0]'):
                        a.append(l)
                        last_found = 'a'
                elif l.startswith('[1]'):
                        b.append(l)
                        last_found = 'b'
                elif last_found == 'a':
                        a.append(l)
                else:
                        b.append(l)
        aj = '\n'.join(a)
        bj = '\n'.join(b)
        print(aj)
        print(bj)
*/

void print_data_pretty(double *arr, int rank, const char *arr_name, int rows, int cols) {
    char *buf = polybench_alloc_data(1000000, 1);
    int pos = 0;
    int out_pos = 0;
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            snprintf(buf + out_pos, 5, "%4f", arr[r * cols + c]);
            out_pos += 4;
            snprintf(buf + out_pos, 2, " ");
            out_pos++;
        }
        snprintf(buf + out_pos, 2, "\n");
        out_pos++;
    }
    printf("[%d] %s:\n%s\n", rank, arr_name, buf);
    free(buf);
}

typedef struct {
    int start;
    int end;
} mpi_bounds_t;
typedef struct {
    DATA_TYPE POLYBENCH_1D(p, N2, n);
    DATA_TYPE POLYBENCH_1D(q, N2, n);
    DATA_TYPE *u;
    DATA_TYPE *v;
    int tsteps;
    int n;
    int argc;
    char **argv;
    int rank, world_size;
    int start_i, end_i;
    mpi_bounds_t *bounds_per_process;
    MPI_Request *recv_requests_v;
    MPI_Request *send_requests_v;
    int recv_req_idx;
    int send_req_idx;
} mpi_adi_data_t;
/*
column sweep: takes wide u vertical band, outputs normal v vertical band
row sweep: takes wide v horizontal band, outputs normal u horizontal band

to receive wide horizontal after column sweep need:
  receive from v, my bounds + above/below rows and their bounds columns

to receive wide vertical after row sweep need:
  receive u, my bounds + left/right cols and their bounds rows
 */
void initialise_benchmark(int argc, char **argv, int tsteps, int n, void **gen_data_ptr) {
    *gen_data_ptr = polybench_alloc_data(1, sizeof(mpi_adi_data_t));
    mpi_adi_data_t *data_ptr = (mpi_adi_data_t *)(*gen_data_ptr);
    memset(data_ptr, 0, sizeof(mpi_adi_data_t));
    data_ptr->tsteps = tsteps;
    data_ptr->n = n;
    data_ptr->argc = argc;
    data_ptr->argv = argv;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &data_ptr->rank);
    // data_ptr->rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &data_ptr->world_size);
    // data_ptr->world_size = 2;

    data_ptr->bounds_per_process = polybench_alloc_data(data_ptr->world_size, sizeof(mpi_bounds_t));

    int cnt = 0;
    int min_elems_per_process = n / data_ptr->world_size;
    for (int pidx = 0; pidx < data_ptr->world_size; pidx++) {
        data_ptr->bounds_per_process[pidx].start = cnt;
        cnt = (pidx == data_ptr->world_size - 1) ? _PB_N : cnt + min_elems_per_process;
        data_ptr->bounds_per_process[pidx].end = cnt;
    }

    mpi_bounds_t *my_bounds = &data_ptr->bounds_per_process[data_ptr->rank];
    // int my_elem_cnt = my_bounds->end - my_bounds->start;

    // eventually I'd like u/v to be reduced size.
    // It should be possible to make each only as big as a "cross",
    // and even smaller to a single band if I can overwrite elements.
    data_ptr->u = polybench_alloc_data(data_ptr->n * data_ptr->n, sizeof(DATA_TYPE));
    data_ptr->v = polybench_alloc_data(data_ptr->n * data_ptr->n, sizeof(DATA_TYPE));
    // for now just making several requests for different parts of block sent/requested
    if (data_ptr->world_size > 1) {
        // int max_outstanding = (data_ptr->world_size-1)*my_elem_cnt;
        int max_outstanding = n * n;
        data_ptr->recv_requests_v = (MPI_Request *)polybench_alloc_data(max_outstanding, sizeof(MPI_Request));
        data_ptr->send_requests_v = (MPI_Request *)polybench_alloc_data(max_outstanding, sizeof(MPI_Request));
    }

    for (int i = 0; i < n; i++) {
        data_ptr->p[i] = 0.0;
        data_ptr->q[i] = 0.0;
        for (int j = 0; j < n; j++) {
            data_ptr->u[(i)*n + (j)] = (DATA_TYPE)(i + n - j) / n;
            data_ptr->v[(i)*n + (j)] = 0.0;
        }
    }
}

void finish_benchmark(void *gen_data_ptr) {
    mpi_adi_data_t *data_ptr = (mpi_adi_data_t *)gen_data_ptr;

    int *elems_per_process = polybench_alloc_data(data_ptr->world_size, sizeof(int));
    int *displs = polybench_alloc_data(data_ptr->world_size, sizeof(int));

    for (int pidx = 0; pidx < data_ptr->world_size; pidx++) {
        displs[pidx] = data_ptr->bounds_per_process[pidx].start * data_ptr->n;
        elems_per_process[pidx] =
            data_ptr->n * (data_ptr->bounds_per_process[pidx].end - data_ptr->bounds_per_process[pidx].start);
    }

    // Sync here the array u only once again!
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, data_ptr->u, elems_per_process, displs, MPI_DOUBLE,
                   MPI_COMM_WORLD);

#ifdef PRINT_DEBUG
    print_data_pretty(data_ptr->u, data_ptr->rank, "u after Allgatherv", data_ptr->n, data_ptr->n);
#endif

    free(elems_per_process);
    free(displs);

    MPI_Finalize();
}

bool should_print_counter(void *gen_data_ptr) {
#ifdef PRINT_DEBUG
    return false;
#else
    mpi_adi_data_t *data_ptr = (mpi_adi_data_t *)gen_data_ptr;
    return data_ptr->rank == 0;
#endif
}

void print_data(int argc, char **argv, int n, void *gen_data_ptr) {
#ifdef PRINT_DEBUG
    return;
#else
    fflush(stdout);
    mpi_adi_data_t *data_ptr = (mpi_adi_data_t *)gen_data_ptr;
    if (data_ptr->rank == 0) {
        default_adi_data_t *adi_data = polybench_alloc_data(1, sizeof(default_adi_data_t));
        memcpy(adi_data->u, data_ptr->u, n * n * sizeof(DATA_TYPE));
        memcpy(adi_data->v, data_ptr->v, n * n * sizeof(DATA_TYPE));
        default_print_data(argc, argv, n, adi_data);
        polybench_free_data(adi_data);
    }
#endif
}

void free_data(void *gen_data_ptr) {
    mpi_adi_data_t *data_ptr = (mpi_adi_data_t *)gen_data_ptr;
    if (data_ptr->world_size > 1) {
        free(data_ptr->recv_requests_v);
        free(data_ptr->send_requests_v);
    }
    free(data_ptr->u);
    free(data_ptr->v);
    free(data_ptr->bounds_per_process);
    free(gen_data_ptr);
}

void get_wide_vert_block(mpi_adi_data_t *data_ptr, int y_pidx) {
    int n = data_ptr->n;
    DATA_TYPE *dest_arr = data_ptr->u;
    mpi_bounds_t their_bounds = data_ptr->bounds_per_process[y_pidx];
    mpi_bounds_t my_bounds = data_ptr->bounds_per_process[data_ptr->rank];

    bool with_pre_col = my_bounds.start != 0;
    bool with_post_col = my_bounds.end != data_ptr->n;
    for (int row_req = their_bounds.start; row_req < their_bounds.end; row_req++) {
#ifdef PRINT_DEBUG
        printf("[%d] Receiving u[row %d, col %d], %d counts, recv req idx %d from source %d\n", data_ptr->rank, row_req,
               my_bounds.start - with_pre_col, my_bounds.end - my_bounds.start + with_pre_col + with_post_col,
               data_ptr->recv_req_idx, y_pidx);
#endif
        MPI_Irecv(&dest_arr[row_req * n + my_bounds.start - with_pre_col],
                  my_bounds.end - my_bounds.start + with_pre_col + with_post_col, MPI_DOUBLE, y_pidx, 0, MPI_COMM_WORLD,
                  &data_ptr->recv_requests_v[data_ptr->recv_req_idx]);
        data_ptr->recv_req_idx++;
    }
}

void get_wide_vert_band(mpi_adi_data_t *data_ptr) {
    for (int pidx = 0; pidx < data_ptr->world_size; pidx++) {
        if (pidx == data_ptr->rank) continue;
        get_wide_vert_block(data_ptr, pidx);
    }
    if (MPI_Waitall(data_ptr->recv_req_idx, data_ptr->recv_requests_v, MPI_STATUSES_IGNORE) != MPI_SUCCESS) {
        printf("Couldn't wait all on get wide vert band\n");
    }
}

// this is for receiving/completing a horizontal band of v, necessary before row sweep
void get_wide_horiz_block(mpi_adi_data_t *data_ptr, int x_pidx) {
    int n = data_ptr->n;
    DATA_TYPE *dest_arr = data_ptr->v;
    mpi_bounds_t their_bounds = data_ptr->bounds_per_process[x_pidx];
    mpi_bounds_t my_bounds = data_ptr->bounds_per_process[data_ptr->rank];

    bool with_pre_row = my_bounds.start != 0;
    bool with_post_row = my_bounds.end != data_ptr->n;
    for (int row_req = my_bounds.start - (int)with_pre_row; row_req < my_bounds.end + (int)with_post_row; row_req++) {
#ifdef PRINT_DEBUG
        printf("[%d] Receiving v[row %d, col %d], %d counts, recv req idx %d from source %d\n", data_ptr->rank, row_req,
               their_bounds.start, their_bounds.end - their_bounds.start, data_ptr->recv_req_idx, x_pidx);
#endif
        MPI_Irecv(&dest_arr[row_req * n + their_bounds.start], their_bounds.end - their_bounds.start, MPI_DOUBLE,
                  x_pidx, 0, MPI_COMM_WORLD, &data_ptr->recv_requests_v[data_ptr->recv_req_idx]);
        data_ptr->recv_req_idx++;
    }
}

void get_wide_horiz_band(mpi_adi_data_t *data_ptr) {
    for (int pidx = 0; pidx < data_ptr->world_size; pidx++) {
        if (pidx == data_ptr->rank) continue;
        get_wide_horiz_block(data_ptr, pidx);
    }
}

// this is for sending out a narrow vertical band of v, after column sweep
// the recipients will want wide horizontal, so send their bounds+2 rows
void send_vert_block(mpi_adi_data_t *data_ptr, int y_pidx) {
    int n = data_ptr->n;
    DATA_TYPE *src_arr = data_ptr->v;
    mpi_bounds_t their_bounds = data_ptr->bounds_per_process[y_pidx];
    mpi_bounds_t my_bounds = data_ptr->bounds_per_process[data_ptr->rank];

    bool with_pre_col = their_bounds.start != 0;
    bool with_post_col = their_bounds.end != data_ptr->n;
    for (int row_req = their_bounds.start - (int)with_pre_col; row_req < their_bounds.end + (int)with_post_col;
         row_req++) {
#ifdef PRINT_DEBUG
        printf("[%d] Sending v[row %d, col %d], %d counts, send req idx %d to destination %d\n", data_ptr->rank,
               row_req, my_bounds.start, my_bounds.end - my_bounds.start, data_ptr->send_req_idx, y_pidx);
#endif
        MPI_Isend(&src_arr[row_req * n + my_bounds.start], my_bounds.end - my_bounds.start, MPI_DOUBLE, y_pidx, 0,
                  MPI_COMM_WORLD, &data_ptr->send_requests_v[data_ptr->send_req_idx]);
        data_ptr->send_req_idx++;
    }
}

// this is for sending out a narrow horizontal band of u, after row sweep
// the recipients will want wide vertical, so send their bounds+2 cols
void send_horiz_block(mpi_adi_data_t *data_ptr, int x_pidx) {
    int n = data_ptr->n;
    DATA_TYPE *src_arr = data_ptr->u;
    mpi_bounds_t their_bounds = data_ptr->bounds_per_process[x_pidx];
    mpi_bounds_t my_bounds = data_ptr->bounds_per_process[data_ptr->rank];

    bool with_pre_col = their_bounds.start != 0;
    bool with_post_col = their_bounds.end != data_ptr->n;
    for (int row_req = my_bounds.start; row_req < my_bounds.end; row_req++) {
#ifdef PRINT_DEBUG
        printf("[%d] Sending u[row %d, col %d], %d counts, send req idx %d to destination %d\n", data_ptr->rank,
               row_req, their_bounds.start - with_pre_col,
               their_bounds.end - their_bounds.start + with_pre_col + with_post_col, data_ptr->send_req_idx, x_pidx);
#endif
        MPI_Isend(&src_arr[row_req * n + their_bounds.start - with_pre_col],
                  their_bounds.end - their_bounds.start + with_pre_col + with_post_col, MPI_DOUBLE, x_pidx, 0,
                  MPI_COMM_WORLD, &data_ptr->send_requests_v[data_ptr->send_req_idx]);
        data_ptr->send_req_idx++;
    }
}

void waitall_receive(mpi_adi_data_t *data_ptr) {
    if (MPI_Waitall(data_ptr->recv_req_idx, data_ptr->recv_requests_v, MPI_STATUSES_IGNORE) != MPI_SUCCESS) {
        printf("Couldn't wait all on waitall receive\n");
    }
}

void waitall_send(mpi_adi_data_t *data_ptr) {
    if (MPI_Waitall(data_ptr->send_req_idx, data_ptr->send_requests_v, MPI_STATUSES_IGNORE) != MPI_SUCCESS) {
        printf("Couldn't wait all on waitall send\n");
    }
}
/* Main computational kernel. The whole function will be timed,
   including the call and return. */
/* Based on a Fortran code fragment from Figure 5 of
 * "Automatic Data and Computation Decomposition on Distributed Memory Parallel
 * Computers" by Peizong Lee and Zvi Meir Kedem, TOPLAS, 2002
 */
void kernel_adi(void *gen_data_ptr) {
    mpi_adi_data_t *data_ptr = (mpi_adi_data_t *)gen_data_ptr;
    DATA_TYPE *u = data_ptr->u;
    DATA_TYPE *v = data_ptr->v;
    DATA_TYPE *p = data_ptr->p;
    DATA_TYPE *q = data_ptr->q;
    int n = data_ptr->n;
    int tsteps = data_ptr->tsteps;

    mpi_bounds_t my_bounds = data_ptr->bounds_per_process[data_ptr->rank];
    int start_comp_at = data_ptr->rank == 0 ? 1 : my_bounds.start;
    int end_comp_at = (data_ptr->rank == data_ptr->world_size - 1) ? _PB_N - 1 : my_bounds.end;

    int t, i, j;
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

    for (t = 1; t <= _PB_TSTEPS; t++) {
        // Column Sweep
        // fills out a full-height partial-width vertical band of v
        // requires a full-height partial-width vertical band of u, 1 element wider on each side
        // Request rests of rows needed for row sweep from the other ranks
        data_ptr->recv_req_idx = 0;
        for (int pidx = 0; pidx < data_ptr->world_size; pidx++) {
            if (pidx == data_ptr->rank) continue;
            get_wide_horiz_block(data_ptr, pidx);
        }
        for (i = start_comp_at; i < end_comp_at; i++) {
#ifdef PRINT_DEBUG
            printf(
                "[%d] Column sweep using u and filling v, columns %d-%d, now at i=%d and requiring adjacent columns %d "
                "and %d\n",
                data_ptr->rank, start_comp_at, end_comp_at, i, i - 1, i + 1);
#endif
            v[(0) * n + (i)] = SCALAR_VAL(1.0);
            p[0] = SCALAR_VAL(0.0);
            q[0] = v[(0) * n + (i)];
            for (j = 1; j < _PB_N - 1; j++) {
                p[j] = -c / (a * p[j - 1] + b);
                q[j] = (-d * u[(j)*n + (i - 1)] + (SCALAR_VAL(1.0) + SCALAR_VAL(2.0) * d) * u[(j)*n + (i)] -
                        f * u[(j)*n + (i + 1)] - a * q[j - 1]) /
                       (a * p[j - 1] + b);
            }

            v[(_PB_N - 1) * n + (i)] = SCALAR_VAL(1.0);
            for (j = _PB_N - 2; j >= 1; j--) {
                v[(j)*n + (i)] = p[j] * v[(j + 1) * n + (i)] + q[j];
            }
        }
#ifdef PRINT_DEBUG
        print_data_pretty(v, data_ptr->rank, "v after column sweep", n, n);
#endif
        data_ptr->send_req_idx = 0;
        // get_wide_horiz_band(data_ptr);
        for (int pidx = 0; pidx < data_ptr->world_size; pidx++) {
            if (pidx == data_ptr->rank) continue;
            send_vert_block(data_ptr, pidx);
        }
        waitall_send(data_ptr);
        waitall_receive(data_ptr);
#ifdef PRINT_DEBUG
        print_data_pretty(v, data_ptr->rank, "v after exchange", n, n);
#endif

        // Row Sweep
        // fills out a partial-height full-width horizontal band of u
        // requires a partial-height full-width horizontal band of v, 1 element taller on each side
        data_ptr->recv_req_idx = 0;
        for (int pidx = 0; pidx < data_ptr->world_size; pidx++) {
            if (pidx == data_ptr->rank) continue;
            get_wide_vert_block(data_ptr, pidx);
        }
        for (i = start_comp_at; i < end_comp_at; i++) {
#ifdef PRINT_DEBUG
            printf(
                "[%d] Row sweep using v and filling u, rows %d-%d, now at i=%d and requiring adjacent rows %d and %d\n",
                data_ptr->rank, start_comp_at, end_comp_at, i, i - 1, i + 1);
#endif
            u[(i)*n + (0)] = SCALAR_VAL(1.0);
            p[0] = SCALAR_VAL(0.0);
            q[0] = u[(i)*n + (0)];
            for (j = 1; j < _PB_N - 1; j++) {
                p[j] = -f / (d * p[j - 1] + e);
                q[j] = (-a * v[(i - 1) * n + (j)] + (SCALAR_VAL(1.0) + SCALAR_VAL(2.0) * a) * v[(i)*n + (j)] -
                        c * v[(i + 1) * n + (j)] - d * q[j - 1]) /
                       (d * p[j - 1] + e);
            }
            u[(i)*n + (_PB_N - 1)] = SCALAR_VAL(1.0);
            for (j = _PB_N - 2; j >= 1; j--) {
                u[(i)*n + (j)] = p[j] * u[(i)*n + (j + 1)] + q[j];
            }
        }
#ifdef PRINT_DEBUG
        print_data_pretty(u, data_ptr->rank, "u after row sweep", n, n);
#endif
        data_ptr->send_req_idx = 0;
        for (int pidx = 0; pidx < data_ptr->world_size; pidx++) {
            if (pidx == data_ptr->rank) continue;
            send_horiz_block(data_ptr, pidx);
        }
        waitall_send(data_ptr);
        waitall_receive(data_ptr);
#ifdef PRINT_DEBUG
        print_data_pretty(u, data_ptr->rank, "u after exchange", n, n);
#endif
    }
}