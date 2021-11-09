# Base image must at least have pytorch and CUDA installed.
ARG BASE_IMAGE=nvidia/cuda:10.2-devel-ubuntu18.04
FROM $BASE_IMAGE
ARG BASE_IMAGE
RUN echo "Installing Apex on top of ${BASE_IMAGE}"

ADD for_docker.zip /workspace
#RUN unzip for_docker.zip -d /workspace

#RUN apt-get --purge autoremove python-pip
RUN apt-get update && DEBIAN_FRONTEND="noninteractive" apt-get install -y vim libffi-dev lzma liblzma-dev libsndfile1 libbz2-dev wget git tar zlib1g-dev build-essential libreadline-gplv2-dev libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev
RUN wget https://www.python.org/ftp/python/3.8.9/Python-3.8.9.tgz
RUN tar xvfz Python-3.8.9.tgz
RUN cd Python-3.8.9 && ./configure && make && make install

RUN update-alternatives --install /usr/bin/python python /usr/local/bin/python3.8 1
RUN python --version
RUN apt-get install -y python3-pip
#RUN pip3 install pip
RUN echo $(pip3 -V)
#RUN pip -V
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1
RUN python3.8 -m pip install --upgrade pip
RUN echo $(pip -V)
# make sure we don't overwrite some existing directory called "apex"
RUN unzip for_docker.zip -d /workspace

WORKDIR /tmp/unique_for_apex
# uninstall Apex if present, twice to make absolutely sure :)
RUN pip install numpy==1.20.1
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

RUN pip install https://github.com/kpu/kenlm/archive/master.zip 

#RUN pip3 install -r requirements.txt
WORKDIR /workspace

#RUN unzip for_docker.zip 
