#!/bin/sh
set -e

cd cuda_utils
./build.sh -d --image-name dphpc-cuda --cuda-version 8.0 --os ubuntu --os-version 16.04 --arch amd64 --load
./build.sh -d --image-name dphpc-cuda --cuda-version 8.0 --os ubuntu --os-version 24.04 --arch amd64 --load
cd ..
DOCKER_BUILDKIT=0 docker compose build --build-arg DOCKER_GID=$(getent group docker | cut -f3 -d:)
