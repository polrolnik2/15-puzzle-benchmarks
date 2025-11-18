FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    libgtest-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install required packages
RUN apt-get update && \
    apt-get install -y cmake g++ libgtest-dev && \
    apt-get clean

#install Debuggers
RUN apt-get update && \
    apt-get install -y gdb lldb && \
    apt-get clean

# Set up workdir and copy project
WORKDIR /workspace
COPY . /workspace

SHELL ["/bin/bash", "-c"]

