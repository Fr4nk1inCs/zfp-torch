FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

LABEL maintainer="fushen@mail.ustc.edu.cn"

# Use USTC Ubuntu mirror
RUN sed -i 's@//.*archive.ubuntu.com@//mirrors.ustc.edu.cn@g' /etc/apt/sources.list \
    && apt update -y \
    && apt upgrade -y

ARG PYTHON_PACKAGES="python3 python3-pip python3-dev python3-venv python-is-python3"
ARG CXX_TOOLCHAIN="build-essential cmake ninja-build"
ARG LIBRARIES="libomp-dev"
ARG TOOLS="git wget curl vim fd-find ripgrep fzf"

# Install packages
RUN apt install -y $PYTHON_PACKAGES $CXX_TOOLCHAIN $LIBRARIES $TOOLS

# Use USTC Pypi mirror
RUN pip install -i https://mirrors.ustc.edu.cn/pypi/simple pip -U \
    && pip config set global.index-url https://mirrors.ustc.edu.cn/pypi/simple

# Clone and build zfp
RUN git clone https://sciproxy.com/https://github.com/llnl/zfp.git --branch release1.0.1 /workspace/zfp \
    && cd /workspace/zfp \
    && mkdir build && cd build \
    && cmake .. -DZFP_WITH_CUDA=ON \
    && cmake --build . \
    && cmake --install .

# MISC: direnv, bashrc
RUN apt install -y direnv \
    && direnv hook bash >> /root/.bashrc \
    && sed -i 's/xterm-color/xterm*/g' /root/.bashrc
