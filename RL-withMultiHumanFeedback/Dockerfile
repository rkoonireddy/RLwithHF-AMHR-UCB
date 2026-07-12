FROM python:3.10-slim

ARG DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ffmpeg \
    git \
    libgl1 \
    libglfw3-dev \
    libglew-dev \
    libegl1 \
    libxext6 \
    libglib2.0-0 \
    libosmesa6 \
    patchelf \
    swig \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MUJOCO_GL=egl \
    PATH="/app:${PATH}"

WORKDIR /app
COPY . /app

RUN pip install --upgrade pip setuptools wheel && pip install .

CMD ["/bin/bash"]
