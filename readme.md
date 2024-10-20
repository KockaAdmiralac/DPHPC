# Running test.py on Euler
Save the shell script below to run_gemver_vbase_serial.sh.
```
#!/bin/sh
python3 test.py --benchmark gemver --variants base_serial --runs 1000
```

First run this on the login node:

    module load stack/2024-06 python/3.11.6

Now run the shell script on the CPU hosts with:

    sbatch run_gemver_vbase_serial.sh

