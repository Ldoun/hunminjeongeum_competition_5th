FROM dacon/nia-pytorch:1.0

# python lib
#COPY requirements.txt ./

# pip3 install and apt-get update
RUN apt-get update && apt-get install -y vim libbz2-dev python3-pip
RUN wget https://www.python.org/ftp/python/3.6.3/Python-3.6.3.tgz
RUN tar xvfz Python-3.6.3.tgz
RUN cd Python-3.6.3 && ./configure && make && make install

# pip upgrade
RUN pip install --upgrade pip

# install Python Packages
# RUN pip install -r requirements.txt

RUN git clone https://github.com/NVIDIA/apex
RUN cd apex
RUN pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
