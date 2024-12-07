FROM dphpc-cuda:8.0-devel-ubuntu24.04

RUN apt update --allow-unauthenticated
RUN apt install bash openssh-server -y
RUN mkdir /var/run/sshd && \
    ssh-keygen -A

RUN useradd --shell /usr/bin/bash --create-home dphpc && \
    echo 'dphpc:k8ngucf2pd' | chpasswd

EXPOSE 22

RUN apt install -y --allow-unauthenticated sudo unminimize
RUN usermod -aG sudo dphpc && \
    yes | unminimize
RUN echo "PATH=\"${PATH}:/usr/local/nvidia/bin:/usr/local/cuda/bin\"" > /etc/profile.d/cuda_path.sh

# Download Docker gpg key
RUN apt-get install ca-certificates curl && \
    install -m 0755 -d /etc/apt/keyrings && \
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc && \
    sudo chmod a+r /etc/apt/keyrings/docker.asc
# Add the repository to Apt sources:
RUN echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  tee /etc/apt/sources.list.d/docker.list > /dev/null && \
    apt-get update --allow-unauthenticated
# Now install docker
ARG DOCKER_GID
RUN apt-get install -y --allow-unauthenticated docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin && \
    groupmod -g $DOCKER_GID docker && \
    usermod -aG docker dphpc

RUN apt install --allow-unauthenticated -y \
    software-properties-common \
    nano \
    python3 \
    python3-pip \
    python3-venv \
    screen \
    tmux \
    command-not-found \
    openmpi-bin \
    openmpi-common \
    libopenmpi-dev \
    rsync \
    git \
    python3-numpy \
    python3-tabulate \
    python3-marshmallow-dataclass \
    python3-typeguard \
    file \
    mypy \
    clang-format \
    black \
    valgrind \
    apt-file && \
    apt-file update

ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

ENTRYPOINT ["/usr/sbin/sshd", "-D"]
