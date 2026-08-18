FROM python:3.10-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    MPLCONFIGDIR=/tmp/matplotlib

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        libgl1 \
        libgomp1 \
        tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/fibar-requirements.txt

# Use PyTorch's CPU wheel repository to avoid bundling CUDA libraries.
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        torch==2.2.0+cpu \
    && python -m pip install --requirement /tmp/fibar-requirements.txt \
    && python -m pip check \
    && tesseract --version

WORKDIR /data
