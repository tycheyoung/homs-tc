FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

ENV PATH /usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

RUN apt-get update && apt-get install -y python3 python3-dev python3-pip libbz2-dev git cmake

# Clone the code
RUN git clone --recurse-submodules https://github.com/tycheyoung/homs-tc.git 

# set working dir
WORKDIR /homs-tc

RUN apt-get clean && apt-get autoremove  && rm -rf /var/lib/apt/lists/*

# Install and build  
RUN bash ./install.sh
