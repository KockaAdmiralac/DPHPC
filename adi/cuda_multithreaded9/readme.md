# CUDA multithreaded 9
A simplification step that unifies column and row sweeping into the same code with different data.
Also, the output array is written more efficiently cache-wise so it's written sequentially row by row rather than the first and last rows first.