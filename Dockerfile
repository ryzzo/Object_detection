###############################################################################
#  Jupyter Notebook image (Python 3.8 + TensorFlow 2.8.2 + TF-OD-API)         #
###############################################################################
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# ───────────────────── avoid interactive tzdata prompt ───────────────────────
ENV DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC
USER root

# ─────────────  ❶ add deadsnakes PPA  ────────────────────────────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.8 python3.8-distutils python3.8-venv wget git && \
    ln -sf /usr/bin/python3.8 /usr/local/bin/python

RUN wget https://bootstrap.pypa.io/pip/3.8/get-pip.py -O - | python && \
    pip install --no-cache-dir "pip<24" wheel \
        notebook jupyterlab ipywidgets

ENV DEBIAN_FRONTEND=noninteractive

RUN ln -fs /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        tzdata \
        software-properties-common \
        build-essential \
        libjpeg-dev \
        zlib1g-dev \
        libusb-1.0.0-dev \
        wget git \
        protobuf-compiler \
        ca-certificates \
        curl \
        libgl1 \
        libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt

RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir -r /app/requirements.txt

# ─────────────  TensorFlow Object-Detection API  ─────────────────────────────
RUN git clone --depth 1 https://github.com/tensorflow/models.git /tmp/models && \
    cd /tmp/models/research && \
    protoc object_detection/protos/*.proto --python_out=. && \
    cp object_detection/packages/tf2/setup.py . && \
    python -m pip install --no-cache-dir . && \
    cd / && rm -rf /tmp/models

# tf-models-official matching TensorFlow 2.8.x
RUN python -m pip install --no-cache-dir tf-models-official==2.8.0

COPY . /app

EXPOSE 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--no-browser", "--allow-root", "--NotebookApp.token=''"]
