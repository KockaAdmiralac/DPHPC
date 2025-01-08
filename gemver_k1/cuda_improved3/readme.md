In kernel1: Simply swapped loops so A doesn't have bank conflicts.
In kernel2: Used a local variable, and to get cublas performance used multiple accumulators to take advantage of non-blocking reads.

Display name should be "Fewer A bank conflicts and multiple accumulators"