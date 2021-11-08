# Base image must at least have pytorch and CUDA installed.
ARG BASE_IMAGE=nvidia/cuda:10.2-devel-ubuntu18.04
FROM $BASE_IMAGE
ARG BASE_IMAGE
RUN echo "Installing Apex on top of ${BASE_IMAGE}"

RUN apt-get update && apt-get install -y vim libbz2-dev python3-pip wget git tar zlib1g-dev
RUN wget https://www.python.org/ftp/python/3.8.9/Python-3.8.9.tgz
RUN tar xvfz Python-3.8.9.tgz
RUN cd Python-3.8.9 && ./configure && make && make install

RUN pip install --upgrade pip

# make sure we don't overwrite some existing directory called "apex"
WORKDIR /tmp/unique_for_apex
# uninstall Apex if present, twice to make absolutely sure :)
RUN pip uninstall -y apex || :
RUN pip uninstall -y apex || :
RUN pip install torch==1.7.1
# SHA is something the user can touch to force recreation of this Docker layer,
# and therefore force cloning of the latest version of Apex
RUN SHA=ToUcHMe git clone https://github.com/NVIDIA/apex.git
WORKDIR /tmp/unique_for_apex/apex
RUN pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .
RUN git clone --recursive https://github.com/parlance/ctcdecode.git
RUN cd ctcdecode && pip install .

WORKDIR /workspace
