#!/usr/bin/env bash

# For Cython
pip3 install -r requirements.txt --upgrade
python3 setup.py build_ext --inplace

# For cnpy
cd cnpy
mkdir -p build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../../cnpy_lib
make
make install

# For CUDA binary
cd ../..
make all
