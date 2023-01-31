#include "../include/encoding.cuh"

#include "../include/fp16_conversion.h"


__device__ float* get2df(float* p, const int x, int y, const int stride) {
    return (float*)((char*)p + x*stride) + y;
}


__device__ char get2d_bin(unsigned int* p, const int i, const int DIM, const int d) {
    unsigned int v = ((*(p + i * ((DIM + PACK_UNIT_SIZE-1)/PACK_UNIT_SIZE) + d/PACK_UNIT_SIZE)) >> ((PACK_UNIT_SIZE-1) - d % PACK_UNIT_SIZE)) & 0x01;
    if (v == 0) {
        return -1;
    } else {
        return 1;
    }
}

__global__ void encodeLevelIdSparse_int8_soa(
    unsigned int* level_hvs_packed, float* id_hvs, int* peak_index, float* peak_intensities, unsigned int* csr_info, int8_t* hv_matrix,
    int N, int Q, int D, int totalFeature, int csr_offset, int bin_offset) {

    const int d = threadIdx.x + blockIdx.x * blockDim.x;
    if (d >= D)
        return;

    // we traverse [start, end-1]
    for (int sample_idx = blockIdx.y; sample_idx < N; sample_idx += blockDim.y * gridDim.y) 
    {
        int8_t encoded_hv_e = 0.0;
        unsigned int start_range = csr_info[csr_offset + sample_idx] - bin_offset;
        unsigned int end_range = (sample_idx + 1 < N) ? csr_info[csr_offset + sample_idx + 1] - bin_offset : totalFeature;
    
        #pragma unroll 1
        for (int f = start_range; f < end_range; ++f) {
            encoded_hv_e += get2d_bin(level_hvs_packed, (int)(peak_intensities[f] * Q), D, d) * \
                            id_hvs[peak_index[f] * D + d];
        }
        hv_matrix[sample_idx * D + d] = encoded_hv_e;
    }
}


__global__ void encodeLevelIdSparse_fp16_soa(
    unsigned int* level_hvs_packed, float* id_hvs, int* peak_index, float* peak_intensities, unsigned int* csr_info, half* hv_matrix,
    int N, int Q, int D, int totalFeature, int csr_offset, int bin_offset) {

    const int d = threadIdx.x + blockIdx.x * blockDim.x;
    if (d >= D)
        return;

    // we traverse [start, end-1]
    for (int sample_idx = blockIdx.y; sample_idx < N; sample_idx += blockDim.y * gridDim.y) 
    {
        float encoded_hv_e = 0.0;
        unsigned int start_range = csr_info[csr_offset + sample_idx] - bin_offset;
        unsigned int end_range = (sample_idx + 1 < N) ? csr_info[csr_offset + sample_idx + 1] - bin_offset : totalFeature;
    
        #pragma unroll 1
        for (int f = start_range; f < end_range; ++f) {
            encoded_hv_e += get2d_bin(level_hvs_packed, (int)(peak_intensities[f] * Q), D, d) * \
                            id_hvs[peak_index[f] * D + d];
        }
        hv_matrix[sample_idx * D + d] = __float2half(encoded_hv_e);
    }
}

__global__ void encodeLevelIdSparse_fp32_soa(
    unsigned int* level_hvs_packed, float* id_hvs, int* peak_index, float* peak_intensities, unsigned int* csr_info, float* hv_matrix,
    int N, int Q, int D, int totalFeature, int csr_offset, int bin_offset) {

    const int d = threadIdx.x + blockIdx.x * blockDim.x;
    if (d >= D)
        return;

    // we traverse [start, end-1]
    for (int sample_idx = blockIdx.y; sample_idx < N; sample_idx += blockDim.y * gridDim.y) 
    {
        float encoded_hv_e = 0.0;
        unsigned int start_range = csr_info[csr_offset + sample_idx] - bin_offset;
        unsigned int end_range = (sample_idx + 1 < N) ? csr_info[csr_offset + sample_idx + 1] - bin_offset : totalFeature;
    
        #pragma unroll 1
        for (int f = start_range; f < end_range; ++f) {
            encoded_hv_e += get2d_bin(level_hvs_packed, (int)(peak_intensities[f] * Q), D, d) * \
                            id_hvs[peak_index[f] * D + d];
        }
        hv_matrix[sample_idx * D + d] = encoded_hv_e;
    }
}


__global__ void normMatRow_int8(float* result, int8_t* inputMat, int setNum, int colNum) {
    for (int rowNum = blockIdx.x * blockDim.x + threadIdx.x; 
        rowNum < setNum; 
        rowNum += blockDim.x * gridDim.x)
    {
        square<float> unary_op;
        thrust::plus<float> binary_op;
        float init = 0;

        result[rowNum] = sqrt(thrust::transform_reduce(thrust::device, inputMat + rowNum * colNum, inputMat + (rowNum + 1) * colNum, unary_op, init, binary_op));
    }
}

__global__ void normMatRow_fp16(half* result, half* inputMat, int setNum, int colNum) {
    for (int rowNum = blockIdx.x * blockDim.x + threadIdx.x; 
        rowNum < setNum; 
        rowNum += blockDim.x * gridDim.x)
    {
        square<float> unary_op;
        thrust::plus<float> binary_op;
        float init = 0;

        result[rowNum] = __float2half(sqrt(thrust::transform_reduce(thrust::device, inputMat + rowNum * colNum, inputMat + (rowNum + 1) * colNum, unary_op, init, binary_op)));
    }
}

__global__ void normMatRow_fp32(float* result, float* inputMat, int setNum, int colNum) {
    for (int rowNum = blockIdx.x * blockDim.x + threadIdx.x; 
        rowNum < setNum; 
        rowNum += blockDim.x * gridDim.x)
    {
        result[rowNum] = sqrt(thrust::transform_reduce(thrust::device, inputMat + rowNum * colNum, inputMat + (rowNum + 1) * colNum, square<float>(), 0.0f, thrust::plus<float>()));
    }
}
