# CUDA multithreaded 12
Version using shared memory to temporarily store the input array, gets great speedup because loading into shared memory is fast and except for edges the data gets reused 3x.

On the GTX 680 N2=7000, TSTEPS=15 gets about 0.193 s.
bandwidth estimates per sweep incl footer:
    load in_arr (N^2)
    store p and q (each N^2)
    load p and q (each N^2)
    store out_arr (N^2)

so 3N^2 load and store each
3*7000²*8 = 1.095 GB loaded/sweep.
Each timestep has 2 sweeps and with TSTEPS=15 there are 30 sweeps total.
So 32.9 GB/run
Including the time for random auxiliary kernels like transpose/initialise at t=0.193 s this gets 170 GB/s bandwidth used for each of load/store.

According to https://www.nvidia.com/content/PDF/product-specifications/GeForce_GTX_680_Whitepaper_FINAL.pdf, pg6 the GTX 680 aka GK104 has a bandwidth of 192.26 GB/s.