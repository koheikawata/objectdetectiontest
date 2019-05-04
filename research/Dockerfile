# FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04
FROM ubuntu:16.04
MAINTAINER Kohei Kawata

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \ 
        python-dev \
        python-pil \
        python-lxml \
        python-tk \
        build-essential \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        cmake \ 
        curl \
        git \ 
        wget \
        vim \
        rsync \
        software-properties-common \
        unzip \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN python --version && \
    ls -l /usr/bin/python
RUN unlink /usr/bin/python && \
    ln -s /usr/bin/python3.5 /usr/bin/python && \
    python --version

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

RUN pip install \
#    tensorflow_gpu==1.12.0 \
    tensorflow \
    numpy \
    cython \
    contextlib2 \
    pillow \
    lxml \
    jupyter \
    matplotlib \
    Flask

RUN mkdir /app && \
    git clone https://github.com/tensorflow/models /app/models

WORKDIR /app/models/research

RUN wget -O ./protobuf.zip https://github.com/google/protobuf/releases/download/v3.0.0/protoc-3.0.0-linux-x86_64.zip && \
    unzip ./protobuf.zip && \
    ./bin/protoc ./object_detection/protos/*.proto --python_out=.

RUN export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

COPY ./object_detection/test1 ./object_detection/test1
COPY ./object_detection_app_test1_docker.py ./object_detection_app_test1_docker.py

EXPOSE 80

CMD ["python", "object_detection_app_test1_docker.py"]
