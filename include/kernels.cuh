#ifndef KERNELS_CUH_
#define KERNELS_CUH_

#include <stdio.h>
#include <iostream>
#include <cuda_fp16.h>


template <typename T>
__global__ void cosSimCalculator(T* sim_mat, T* norm_vec, int Nq, int N) {
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; 
        idx < N;
        idx += blockDim.x * gridDim.x)
    {
        T norm_s = norm_vec[idx];
        for (int i = 0; i < Nq; ++i)
            sim_mat[i * N + idx] /= norm_s;
    }
}

__global__ void packing(unsigned int* output, float* arr, int origLength, int packLength, int numVec);


template<typename T>
__global__ void getTopItemIndex(int* topKIndex, T* topKScore, T* __restrict__ Y_item, 
                                float* __restrict__ spectra_precursormz, float* __restrict__ query_precursormz, 
                                const int charge, const float tol_val_ppm, const float tol_val_da,  const int rowNum, 
                                const int colNum, const int batch_offset, const int ref_batch_offset) {

    for (int dx = blockIdx.x * blockDim.x + threadIdx.x; 
         dx < rowNum; // Nq
         dx += blockDim.x * gridDim.x) 
    {
        float query_precursor_mz = query_precursormz[batch_offset + dx];

        // argmax per row
        T max_value_open = -99999.;
        int max_idx_open = -1;
        T max_value_std = -99999.;
        int max_idx_std = -1;
        #pragma unroll
        for (int j = 0; j < colNum; j++) {
            // evaluate tolerance "ppm"
            float library_mzs = spectra_precursormz[ref_batch_offset + j];
            float abs_precursor_diff = fabsf(query_precursor_mz - library_mzs);

            if (abs_precursor_diff/library_mzs * 1e6 <= tol_val_ppm) {  // standard search
                T val_to_compare = Y_item[colNum * dx + j];
                if (max_value_std < val_to_compare) {
                    max_value_std = val_to_compare;
                    max_idx_std = j;
                }
            }
            if (abs_precursor_diff * charge <= tol_val_da) {  // open search
                T val_to_compare = Y_item[colNum * dx + j];
                if (max_value_open < val_to_compare) {
                    max_value_open = val_to_compare;
                    max_idx_open = j;
                }

            }
        }
        topKIndex[dx] = max_idx_std;
        topKIndex[dx + rowNum] = max_idx_open;
        if (max_idx_std != -1) {
            topKScore[dx] = max_value_std;
            topKScore[dx + rowNum] = max_value_open;
        }
    }
}

template<typename T>
__global__ void getTopItemIndex_int8(int* topKIndex, T* topKScore, int* __restrict__ Y_item, 
                                T* __restrict__ Y_item_norm,
                                float* __restrict__ spectra_precursormz, float* __restrict__ query_precursormz, 
                                const int charge, const float tol_val_ppm, const float tol_val_da,  const int rowNum, 
                                const int colNum, const int batch_offset, const int ref_batch_offset) {

    for (int dx = blockIdx.x * blockDim.x + threadIdx.x; 
         dx < rowNum; // Nq
         dx += blockDim.x * gridDim.x) 
    {
        float query_precursor_mz = query_precursormz[batch_offset + dx];

        // argmax per row
        T max_value_open = -99999.;
        int max_idx_open = -1;
        T max_value_std = -99999.;
        int max_idx_std = -1;
        #pragma unroll
        for (int j = 0; j < colNum; j++) {
            // evaluate tolerance "ppm"
            float library_mzs = spectra_precursormz[ref_batch_offset + j];
            float abs_precursor_diff = fabsf(query_precursor_mz - library_mzs);

            T curr_norm = Y_item_norm[j];

            if (abs_precursor_diff/library_mzs * 1e6 <= tol_val_ppm) {  // standard search
                T val_to_compare = ((float) Y_item[colNum * dx + j]) / curr_norm;
                if (max_value_std < val_to_compare) {
                    max_value_std = val_to_compare;
                    max_idx_std = j;
                }
            }
            if (abs_precursor_diff * charge <= tol_val_da) {  // open search
                T val_to_compare = ((float) Y_item[colNum * dx + j]) / curr_norm;
                if (max_value_open < val_to_compare) {
                    max_value_open = val_to_compare;
                    max_idx_open = j;
                }

            }
        }
        topKIndex[dx] = max_idx_std;
        topKIndex[dx + rowNum] = max_idx_open;
        if (max_idx_std != -1) {
            topKScore[dx] = max_value_std;
            topKScore[dx + rowNum] = max_value_open;
        }
    }
}

#endif
