# CUDA minimise spare array population
By tracing out how the u/v array looks like at different parts of a half-timestep, one can tell the first and last rows are the only data that must be copied back and forth between u/v and the spare array.  These rows are critical because eventually they show up in the results as-is.  By copying just these rows over one can eliminate the entire array transposition in use so far.

Display name should be "Transposition limited to sides only"