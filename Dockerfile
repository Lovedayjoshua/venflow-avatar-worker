FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python

RUN python -m pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA 12.1 support.
RUN pip install \
    torch==2.4.1 \
    torchvision==0.19.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Clone the official MuseTalk repository.
RUN git clone --depth 1 https://github.com/TMElyralab/MuseTalk.git /app/MuseTalk

WORKDIR /app/MuseTalk

# Install MuseTalk's own dependencies.
RUN pip install -r requirements.txt

# Install RunPod worker SDK and worker dependencies.
COPY requirements.txt /app/venflow-requirements.txt
RUN pip install -r /app/venflow-requirements.txt

# Create runtime directories.
RUN mkdir -p \
    /models \
    /workspace/input \
    /workspace/output \
    /workspace/avatar-cache

# MuseTalk model directory.
ENV MUSETALK_MODEL_DIR=/models

# Venflow runtime directories.
ENV VENFLOW_INPUT_DIR=/workspace/input
ENV VENFLOW_OUTPUT_DIR=/workspace/output
ENV VENFLOW_AVATAR_CACHE=/workspace/avatar-cache

# Copy our RunPod worker.
COPY src/handler.py /app/handler.py

WORKDIR /app

CMD ["python", "-u", "/app/handler.py"]
