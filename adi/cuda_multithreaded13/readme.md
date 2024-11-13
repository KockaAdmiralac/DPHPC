# CUDA multithreaded 13
Variant of 12 but where the first and last threads load the neighbour values into shared memory as well.  Performance is basically identical but makes the code nastier.