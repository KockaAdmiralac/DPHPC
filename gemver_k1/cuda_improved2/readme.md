Attempt at addressing bank conflicts on u1/u2 by putting them into shared memory.  No performance uplift but far more bugs and difficult code.  There's probably no performance uplift because global memory coalescing detects the "broadcast" of all threads reading the same value.

Display name should be "u1/u2 in shared memory"