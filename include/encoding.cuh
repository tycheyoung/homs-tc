#ifndef ENCODING_CUH_
#define ENCODING_CUH_

#include <stdio.h>
#include <iostream>

#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>

#include <cuda_fp16.h>

#define PACK_UNIT_SIZE 32


template<typename T>
struct square                                                                   
{                                                                               
    __host__ __device__                                                         
    T operator()(const T& x) const                                              
    {                                                                           
        return x*x;                                                             
    }                                                                           
}; 

__global__ void normMatRow_int8(float* result, int8_t* inputMat, int setNum, int colNum);
__global__ void normMatRow_fp16(half* result, half* inputMat, int setNum, int colNum);
__global__ void normMatRow_fp32(float* result, float* inputMat, int setNum, int colNum);


__global__ void encodeLevelIdSparse_int8_soa(
    unsigned int* level_hvs_packed, float* id_hvs, int* peak_index, float* peak_intensities, unsigned int* csr_info, int8_t* hv_matrix,
    int N, int Q, int D, int totalFeature, int csr_offset, int bin_offset);
__global__ void encodeLevelIdSparse_fp16_soa(
    unsigned int* level_hvs_packed, float* id_hvs, int* peak_index, float* peak_intensities, unsigned int* csr_info, half* hv_matrix,
    int N, int Q, int D, int totalFeature, int csr_offset, int bin_offset);
__global__ void encodeLevelIdSparse_fp32_soa(
    unsigned int* level_hvs_packed, float* id_hvs, int* peak_index, float* peak_intensities, unsigned int* csr_info, float* hv_matrix,
    int N, int Q, int D, int totalFeature, int csr_offset, int bin_offset);

#endif
