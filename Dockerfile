FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04

ENV CUDA_ROOT /usr/local/cuda/bin/
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib64/
RUN ldconfig

ENV TZ=Europe/Minsk
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get update && apt-get install -y \
  software-properties-common \
  && \
  rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y \
  ca-certificates \
 # libjasper-runtime \
  openssh-server \
  git \
  wget \
  libopenblas-dev \
  python3.6-dev \
  python3-pip \
  gcc \
  g++ \
  make \
  gfortran \
  libatlas-base-dev \
  libblas-dev \
  liblapack-dev \
  build-essential \
  curl \
  libfreetype6 \
  libfreetype6-dev \
  libjpeg62-dev \
  libjpeg8 \
  #libpng12-dev \
  libzmq3-dev \
  pkg-config \
  python-opencv \
  software-properties-common \
  unzip \
  zip \
  sudo \
  && \
  rm -rf /var/lib/apt/lists/*

# install latest cmake 3.*
#RUN add-apt-repository ppa:george-edison55/cmake-3.x
#RUN apt-get update && apt-get install -y cmake && \
#  rm -rf /var/lib/apt/lists/*

# Setup ssh
RUN mkdir /var/run/sshd
RUN service ssh stop

# Add user with valid passwrd
RUN useradd -ms /bin/bash user
RUN (echo user ; echo user) | passwd user

# Configure sudo
RUN usermod -a -G sudo user

RUN python3.6 -m pip install pip --upgrade


RUN python3.6 -m pip install pillow
RUN python3.6 -m pip install evalutils
RUN python3.6 -m pip install SimpleITK==1.0.0
RUN python3.6 -m pip install scikit-learn
RUN python3.6 -m pip install opencv-python==3.4.2.17

RUN python3.6 -m pip install pydicom


RUN python3.6 -m pip install networkx




# install PyTorch
RUN python3.6 -m pip install torch===1.6.0 torchvision===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html

RUN python3.6 -m pip uninstall -y dgl
#RUN python3.6 -m pip install https://s3.us-east-2.amazonaws.com/dgl.ai/wheels/cuda10.0/dgl-0.3-cp36-cp36m-manylinux1_x86_64.whl
RUN python3.6 -m pip install dgl-cu100

# Add tiger
#RUN git clone https://github.com/xieweiyi/msk-tiger.git /opt/tiger/
#RUN cd /opt/tiger && python3.6 -m pip install .
# Add Apax for mixed precision training support in PyTorch
#RUN git clone https://github.com/NVIDIA/apex.git /home/root
#RUN cd /home/root && python3.6 -m pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .

ENV PATH "$PATH:$CUDA_ROOT"
RUN echo "PATH=$PATH" > /etc/environment
RUN echo "CUDA_ROOT=$CUDA_ROOT" >> /etc/environment
RUN echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH" >> /etc/environment


RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm

RUN mkdir -p /opt/algorithm /input /output \
    && chown algorithm:algorithm /opt/algorithm /input /output

USER algorithm

WORKDIR /opt/algorithm
#
#ENV PATH="/home/algorithm/.local/bin:${PATH}"
#
#RUN python -m pip install --user -U pip



COPY --chown=algorithm:algorithm requirements.txt /opt/algorithm/
#RUN python -m pip install --user -rrequirements.txt

#COPY --chown=algorithm:algorithm process.py /opt/algorithm/

COPY models.py /opt/algorithm/models.py
COPY test.py /opt/algorithm/test.py
COPY best.pth /opt/algorithm/best.pth
COPY settings.py /opt/algorithm/settings.py

#ENTRYPOINT python -m process $0 $@
ENTRYPOINT ["/usr/bin/python3.6", "/opt/algorithm/test.py"]
## ALGORITHM LABELS ##

# These labels are required
LABEL nl.diagnijmegen.rse.algorithm.name=lobeseg
LABEL nl.diagnijmegen.rse.algorithm.author="Weiyi Xie"
LABEL nl.diagnijmegen.rse.algorithm.ticket=7838

# These labels are required and describe what kind of hardware your algorithm requires to run.
LABEL nl.diagnijmegen.rse.algorithm.hardware.cpu.count=2
LABEL nl.diagnijmegen.rse.algorithm.hardware.cpu.capabilities=""
LABEL nl.diagnijmegen.rse.algorithm.hardware.memory=82G
LABEL nl.diagnijmegen.rse.algorithm.hardware.gpu.count=1
LABEL nl.diagnijmegen.rse.algorithm.hardware.gpu.cuda_compute_capability=""
LABEL nl.diagnijmegen.rse.algorithm.hardware.gpu.memory=11G


