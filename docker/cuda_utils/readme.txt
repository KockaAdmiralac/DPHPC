Stripped version of https://gitlab.com/nvidia/container-images/cuda for DPHPC course
It has a slightly modified set of Dockerfiles for Ubuntu 24.04 with CUDA 9.1 normally built for Ubuntu 16.04.

Run with:
./build.sh -d --image-name dphpc-cuda --cuda-version 9.1 --os ubuntu --os-version 24.04 --arch amd64 --load
