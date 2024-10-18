#!/bin/sh

# Example script of something batch-runnable on Euler.
# sbatch run_gemver_v0_serial.sh

# first run this on the login node:
#module load stack/2024-06 python/3.11.6

python3 test.py --benchmark gemver --variants 0_serial --runs 1000