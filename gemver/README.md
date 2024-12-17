# gemver

| Variant                         | Based on                     | Speedup 1T | 2T   | 4T   | 8T   |
| ------------------------------- | ---------------------------- | ---------- | ---- | ---- | ---- |
| `serial_base`                   | -                            | -          | -    | -    | -    |
| `serial_neater_c`               | `serial_base`                | 1.0        | -    | -    | -    |
| `serial_block_first_loop`       | `serial_neater_c`            | ?          | ?    | ?    | ?    |
| `serial_vectorized_first_loop`  | `serial_block_first_loop`    | ?          | ?    | ?    | ?    |
| `serial_reorder_second_loop`    | `serial_neater_c`            | ?          | ?    | ?    | ?    |
| `serial_block_second_loop`      | `serial_reorder_second_loop` | ?          | ?    | ?    | ?    |
| `serial_vectorized_second_loop` | `serial_block_second_loop`   | ?          | ?    | ?    | ?    |
| `serial_merge_loops`            | `serial_reorder_second_loop` | 6.5        | ?    | ?    | ?    |
| `serial_extracted_beta`         | `serial_merge_loops`         | 6.5        | ?    | ?    | ?    |
| `serial_extracted_alpha`        | `serial_extracted_beta`      | 6.5        | ?    | ?    | ?    |
| `openmp_basic`                  | `serial_neater_c`            | 1.0        | 1.4  | 1.7  | 1.7  |
| `openmp_basic_all`              | `openmp_basic`               | 1.0        | 1.7  | 1.8  | 1.8  |
| `openmp_reorder_atomic`         | `serial_reorder_second_loop` | 6.0        | 1.8  | 2.2  | 3.0  |
| `openmp_reorder_reduce`         | `serial_reorder_second_loop` | 6.5        | 6.5  | 6.5  | 7.0  |
| `openmp_block_second_loop`      | `openmp_reorder_reduce`      | 6.0        | 6.5  | 6.5  | 6.5  |
| `openmp_merged`                 | `openmp_block_second_loop`   | 6.5        | 10.0 | 10.0 | 10.0 |
| `openmp_avx2`                   | `openmp_merged`              | ?          | ?    | ?    | ?    |
| `mpi_rows`                      | `mpi_cols`                   | ?          | ?    | ?    | ?    |
| `mpi_rows_outside_sync_A`       | `mpi_rows`                   | ?          | ?    | ?    | ?    |
| `mpi_columns_openmp`            | `mpi_cols`                   | ?          | ?    | ?    | ?    |
| `mpi_columns_non_blocking_communication_outside_sync_A`| `mpi_cols` | ?          | ?    | ?    | ?    |
