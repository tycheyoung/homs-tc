#include <stdio.h>
#include <iostream>
#include <string>
#include <algorithm>
#include <vector>
#include <string>
#include <chrono>
#include <fstream>
#include <experimental/filesystem>

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/generate.h>
#include <thrust/sequence.h>

#include "cnpy_lib/include/cnpy.h"

#include "include/openhd.cuh"
#include "include/openhd_rand.cuh"
#include "include/cudadebug.cuh"
#include "include/encoding.cuh"
#include "include/kernels.cuh"
#include "include/fp16_conversion.h"
#include "include/dataio.hpp"

#define __MAX_THREADS__ 1024
#define MIN(x, y) ((x < y) ? x : y)

// USE_INT8 or USE_FLOAT or USE_HALF will be defined in the Makefile

// #define VERBOSE
#define CSV_OUTPUT

int main(int argc, char * argv[]) {

    if (argc < 14) {
        std::cerr << "Usage: " << argv[0] << " <encoded splib filename> <encoded query filename> <splib charge> <query charge> ";
        std::cerr << "<vector_length> <HD Dimen> <Q> <tol_val_ppm> <tol_val_Da> <BATCH SPLIB SIZE> <BATCH SIZE>";
        std::cerr << "<output_std filename> <output_open filename> <dump ref_hvs>\n";
        std::cerr << "Usage: " << argv[0] << " ./dataset/human_yeast_targetdecoy ./dataset/iPRG2012 3 3";
        std::cerr << "34976 1000 100 20 500 2000 1000 ./dataset/output ./dataset/output_open\n";
        exit(1);
    }

    // Argument read
    std::string vec_len = argv[5];  // vector_length
    std::string fs = argv[1];
    fs = fs + "_vec_" + vec_len;
    std::string fq = argv[2];
    fq = fq + "_vec_" + vec_len;
    std::string splibcharge = argv[3];
    int splibcharge_i = atoi(argv[3]);
    std::string querycharge = argv[4];
    unsigned int F = atoi(argv[5]);
    unsigned int D = atoi(argv[6]);
    unsigned int Q = atoi(argv[7]);

    float tol_val_ppm = atof(argv[8]);
    float tol_val_da = atof(argv[9]);

    const unsigned int BATCH_SPLIB_SIZE = atoi(argv[10]);
    const unsigned int BATCH_SIZE = atoi(argv[11]);
    std::ofstream outputfile_std(argv[12], std::ios::app);
    std::ofstream outputfile_open(argv[13], std::ios::app);
    bool use_precomputed_ref_hvs = (atoi(argv[14]) != 0);

    std::string splib_filename = fs + ".charge" + splibcharge + ".npz";
    std::string query_filename = fq + ".charge" + querycharge + ".npz";

    // Load ref npz
    cnpy::npz_t ref_npz = cnpy::npz_load(splib_filename);
    std::vector<float> vectorized_ref_intensities = ref_npz["spectra_intensities"].as_vec<float>();
    std::vector<int> vectorized_ref_index = ref_npz["spectra_idx"].as_vec<int>();
    std::vector<float> vectorized_ref_precursormz = ref_npz["pr_mzs"].as_vec<float>();
    std::vector<int> vectorized_ref_identifier = ref_npz["spectra_identifier"].as_vec<int>();

    unsigned int* csr_info = ref_npz["csr_info"].data<unsigned int>();
    unsigned int N = ref_npz["pr_mzs"].shape[0];
    unsigned int ref_bins = ref_npz["spectra_intensities"].shape[0];

    // Load query npz
    cnpy::npz_t query_npz = cnpy::npz_load(query_filename);
    std::vector<float> vectorized_query_intensities = query_npz["spectra_intensities"].as_vec<float>();
    std::vector<int> vectorized_query_index = query_npz["spectra_idx"].as_vec<int>();
    std::vector<float> vectorized_query_precursormz = query_npz["pr_mzs"].as_vec<float>();

    unsigned int* q_csr_info = query_npz["csr_info"].data<unsigned int>();
    unsigned int Nq = query_npz["pr_mzs"].shape[0];
    unsigned int query_bins = query_npz["spectra_intensities"].shape[0];

    #ifdef VERBOSE
    std::cout << "Splib batch uses: " << BATCH_SPLIB_SIZE << " / Query batch uses: " << BATCH_SIZE << std::endl;
    std::cout << "splib: " << N << " spectras / " << ref_bins << " bins"   << std::endl;
    std::cout << "query: " << Nq << " spectras / " << query_bins << " bins"   << std::endl;
    #endif

    
    const unsigned int TOTAL_REF_BATCH_NUM = (N + BATCH_SPLIB_SIZE - 1) / BATCH_SPLIB_SIZE;
    const unsigned int TOTAL_BATCH_NUM = (Nq + BATCH_SIZE - 1) / BATCH_SIZE;

    // Check if there's precomputed reference HVs
    // if use_precomputed_ref_hvs is true, but there's no precomputed reference HVs,
    // then we will dump the reference HVs to disk
    bool dump_ref_hvs = false;
    if (use_precomputed_ref_hvs) {
        printf("Checking if precomputed reference HVs exist...\n");
        for (int i = 0; i < TOTAL_REF_BATCH_NUM; i++) {
            std::string filename = fs + "_hvs_d" + std::to_string(D) +  "_c" + splibcharge + "_B" \
                                + std::to_string(BATCH_SPLIB_SIZE) + "_" + std::to_string(i) + ".bin";
            if (!std::experimental::filesystem::exists(filename)) {
                printf("Precomputed reference HVs not found, will dump them to disk.\n");
                dump_ref_hvs = true;
                break;
            }
        }
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties( &prop, 0);

    cudaError_t err = cudaSuccess;
    curandState* devState;
    HANDLE_ERROR(cudaMalloc((void**)&devState, D * sizeof(curandState)));
    float* d_id_hvs;
    float* d_level_hvs;
    HANDLE_ERROR(cudaMalloc((void **)&d_id_hvs, F * D * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void **)&d_level_hvs, (Q + 1) * D * sizeof(float)));

    // generate id base and level base in gpu
    dim3 nBlock(__MAX_THREADS__, 1, 1);
    dim3 nGrid((D + __MAX_THREADS__ - 1) / __MAX_THREADS__, 1);

    init_rand<<<nGrid, nBlock>>>(devState, D, 0);

    #ifndef USE_INT8
        generate_large_hvs<<<nGrid, nBlock>>>(d_id_hvs, devState, F, D);
    #else
        generate_large_hvs_binary<<<nGrid, nBlock>>>(d_id_hvs, devState, F, D);
    #endif
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch kernel (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    std::vector<float> bases;  // flattened
    std::vector<float> base_v1(D/2, 1);
    std::vector<float> base_v2(D/2, -1);
    base_v1.insert(base_v1.end(), base_v2.begin(), base_v2.end());
    std::vector<float> level_base(base_v1);
    std::vector<float> level_hvs;
    for (int q = 0; q <= Q; ++q) {
        int flip = (int) (q/float(Q) * D) / 2;
        std::vector<float> level_hv(level_base);
        // + flip will transform (flip) number of elements
        std::transform(level_hv.begin(), level_hv.begin() + flip, level_hv.begin(), bind2nd(std::multiplies<float>(), -1)); 
        level_hvs.insert(level_hvs.end(), level_hv.begin(), level_hv.end());
    }
    HANDLE_ERROR(cudaMemcpy(d_level_hvs, level_hvs.data(), level_hvs.size() * sizeof(float), cudaMemcpyHostToDevice));

    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch kernel (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Pack ID HVS & LV HVS
    const int pack_len = (D+PACK_UNIT_SIZE-1)/PACK_UNIT_SIZE;
    unsigned int* d_level_hvs_packed;
    HANDLE_ERROR(cudaMalloc((void **)&d_level_hvs_packed, (Q+1) * pack_len * sizeof(unsigned int)));
    dim3 nGridMulti2((D + __MAX_THREADS__ - 1) / __MAX_THREADS__, Q+1);
    packing<<<nGridMulti2, nBlock>>>(d_level_hvs_packed, d_level_hvs, D, pack_len, Q+1);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch kernel (error code %s)!\n",
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    HANDLE_ERROR(cudaFree(d_level_hvs));


    #ifndef USE_INT8
    const float alf = 1.;
    const float bet = 0.;
    const float *alpha = &alf;
    const float *beta = &bet;
    #else
    const int alf = 1;
    const int bet = 0;
    const int *alpha = &alf;
    const int *beta = &bet;
    #endif
    unsigned int* d_q_csr_info;
    HANDLE_ERROR(cudaMalloc((void **)&d_q_csr_info, Nq * sizeof(unsigned int)));
    HANDLE_ERROR(cudaMemcpy(d_q_csr_info, q_csr_info, Nq * sizeof(unsigned int), cudaMemcpyHostToDevice));

    float* d_vectorized_ref_precursormz = NULL;
    HANDLE_ERROR(cudaMalloc((void **)&d_vectorized_ref_precursormz, N * sizeof(float)));
    HANDLE_ERROR(cudaMemcpy(d_vectorized_ref_precursormz, vectorized_ref_precursormz.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    float* d_vectorized_query_precursormz = NULL;
    HANDLE_ERROR(cudaMalloc((void **)&d_vectorized_query_precursormz, Nq * sizeof(float)));
    HANDLE_ERROR(cudaMemcpy(d_vectorized_query_precursormz, vectorized_query_precursormz.data(), Nq * sizeof(float), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    cublasCreate(&handle);

    float* d_vectorized_ref_intensities = NULL;
    int* d_vectorized_ref_index = NULL;
    unsigned int* d_csr_info;
    if (!use_precomputed_ref_hvs) {
        HANDLE_ERROR(cudaMalloc((void **)&d_csr_info, N * sizeof(unsigned int)));
        HANDLE_ERROR(cudaMalloc((void **)&d_vectorized_ref_intensities, vectorized_ref_intensities.size() * sizeof(float)));
        HANDLE_ERROR(cudaMalloc((void **)&d_vectorized_ref_index, vectorized_ref_index.size() * sizeof(int)));
        HANDLE_ERROR(cudaMemcpy(d_vectorized_ref_intensities, vectorized_ref_intensities.data(), vectorized_ref_intensities.size() * sizeof(float), cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(d_vectorized_ref_index, vectorized_ref_index.data(), vectorized_ref_index.size() * sizeof(int), cudaMemcpyHostToDevice));
        HANDLE_ERROR(cudaMemcpy(d_csr_info, csr_info, N * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
        

    float* d_vectorized_query_intensities = NULL;
    int* d_vectorized_query_index = NULL;
    HANDLE_ERROR(cudaMalloc((void **)&d_vectorized_query_intensities, query_bins * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void **)&d_vectorized_query_index, query_bins * sizeof(int)));
    HANDLE_ERROR(cudaMemcpy(d_vectorized_query_intensities, vectorized_query_intensities.data(), query_bins * sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_vectorized_query_index, vectorized_query_index.data(), query_bins * sizeof(int), cudaMemcpyHostToDevice));

    int processed_ref_batch_size = 0;

    // Measure Time per stage
    cudaEvent_t stop1, stop2;
    cudaEventCreate(&stop1);
    cudaEventCreate(&stop2);

    float tmp_time = 0;

    float ref_encode_time = 0;
    float query_encode_time = 0;
    float search_time = 0;

    #ifdef USE_HALF
    __half* d_hv_matrix; __half* d_q_hv_matrix;
    float* d_guess_vec = NULL;
    __half* d_splibNorm;
    #endif
    #ifdef USE_FLOAT
    float* d_hv_matrix; float* d_q_hv_matrix;
    float* d_guess_vec = NULL;
    float* d_splibNorm;
    #endif
    #ifdef USE_INT8
    int8_t* d_hv_matrix; int8_t* d_q_hv_matrix;
    int* d_guess_vec = NULL;
    float* d_splibNorm;
    #endif

    int* d_topIndex = NULL;
    float* d_topScore = NULL;
    
    float* d_vectorized_ref_intensities_batch = d_vectorized_ref_intensities;
    int* d_vectorized_ref_index_batch = d_vectorized_ref_index;

    int prev_N_batch = 0;
    int prev_Nq_batch = 0;

    auto start = std::chrono::high_resolution_clock::now();

    for (int ref_batch_num = 0; ref_batch_num < TOTAL_REF_BATCH_NUM; ref_batch_num++) {
        // This includes bin size
        int ref_batch_size = (ref_batch_num != TOTAL_REF_BATCH_NUM - 1) ? 
                        (csr_info[(ref_batch_num + 1)* BATCH_SPLIB_SIZE] - csr_info[ref_batch_num * BATCH_SPLIB_SIZE]) 
                      : (ref_bins - csr_info[ref_batch_num * BATCH_SPLIB_SIZE]);
        int N_batch = (ref_batch_num != TOTAL_REF_BATCH_NUM - 1) ? BATCH_SPLIB_SIZE: (N % BATCH_SPLIB_SIZE);
        if ((ref_batch_num == TOTAL_REF_BATCH_NUM - 1) && (N_batch == 0)) { 
            N_batch = BATCH_SPLIB_SIZE;
        }
        if (ref_batch_num == 0)
            prev_N_batch = N_batch;

        if (N_batch != prev_N_batch) {
            HANDLE_ERROR(cudaFree(d_hv_matrix));
            HANDLE_ERROR(cudaFree(d_splibNorm));
        }
        #ifdef USE_HALF
        if (N_batch != prev_N_batch || ref_batch_num == 0)
            HANDLE_ERROR(cudaMalloc((void **)&d_hv_matrix, D * N_batch * sizeof(__half)));
        #endif

        #ifdef USE_FLOAT
        if (N_batch != prev_N_batch || ref_batch_num == 0)
            HANDLE_ERROR(cudaMalloc((void **)&d_hv_matrix, D * N_batch * sizeof(float)));
        #endif

        #ifdef USE_INT8
        if (N_batch != prev_N_batch || ref_batch_num == 0)
            HANDLE_ERROR(cudaMalloc((void **)&d_hv_matrix, D * N_batch * sizeof(int8_t)));
        #endif

        dim3 encodeSplibDim((D + __MAX_THREADS__ - 1) / __MAX_THREADS__, MIN(N_batch, prop.maxGridSize[1]));
        cudaEventRecord(stop1);
        if (!use_precomputed_ref_hvs) {
            #ifdef USE_HALF
            encodeLevelIdSparse_fp16_soa<<<encodeSplibDim, nBlock>>>(d_level_hvs_packed, d_id_hvs, d_vectorized_ref_index_batch, d_vectorized_ref_intensities_batch, d_csr_info, d_hv_matrix,
                                                            N_batch, Q, D, ref_batch_size, ref_batch_num * BATCH_SPLIB_SIZE,
                                                            processed_ref_batch_size);
            #endif

            #ifdef USE_FLOAT
            encodeLevelIdSparse_fp32_soa<<<encodeSplibDim, nBlock>>>(d_level_hvs_packed, d_id_hvs, d_vectorized_ref_index_batch, d_vectorized_ref_intensities_batch, d_csr_info, d_hv_matrix,
                                                            N_batch, Q, D, ref_batch_size, ref_batch_num * BATCH_SPLIB_SIZE,
                                                            processed_ref_batch_size);
            #endif

            #ifdef USE_INT8
            encodeLevelIdSparse_int8_soa<<<encodeSplibDim, nBlock>>>(d_level_hvs_packed, d_id_hvs, d_vectorized_ref_index_batch, d_vectorized_ref_intensities_batch, d_csr_info, d_hv_matrix,
                                                            N_batch, Q, D, ref_batch_size, ref_batch_num * BATCH_SPLIB_SIZE,
                                                            processed_ref_batch_size);
            #endif
            if (dump_ref_hvs) {
                std::string dump_fname = fs + "_hvs_d" + std::to_string(D) +  "_c" + splibcharge + "_B" \
                                         + std::to_string(BATCH_SPLIB_SIZE) + "_" + std::to_string(ref_batch_num) + ".bin";
                std::cout << "save reference HVs to " << dump_fname << std::endl;
                #ifdef USE_HALF
                __half* hv_matrix = (__half*)malloc(D * N_batch * sizeof(__half));
                HANDLE_ERROR(cudaMemcpy(hv_matrix, d_hv_matrix, D * N_batch * sizeof(__half), cudaMemcpyDeviceToHost));
                saveArr<half>(dump_fname, hv_matrix, D * N_batch);
                free(hv_matrix);
                #endif
                #ifdef USE_FLOAT
                float* hv_matrix = (float*)malloc(D * N_batch * sizeof(float));
                HANDLE_ERROR(cudaMemcpy(hv_matrix, d_hv_matrix, D * N_batch * sizeof(float), cudaMemcpyDeviceToHost));
                saveArr<float>(dump_fname, hv_matrix, D * N_batch);
                free(hv_matrix);
                #endif
                #ifdef USE_INT8
                int8_t* hv_matrix = (int8_t*)malloc(D * N_batch * sizeof(int8_t));  // TOOD: error exception
                HANDLE_ERROR(cudaMemcpy(hv_matrix, d_hv_matrix, D * N_batch * sizeof(int8_t), cudaMemcpyDeviceToHost));
                saveArr<int8_t>(dump_fname, hv_matrix, D * N_batch);
                free(hv_matrix);
                #endif
            }
        } else {
            // Load precomputed ref hv
            std::string dump_fname = fs + "_hvs_d" + std::to_string(D) +  "_c" + splibcharge + "_B" \
                                     + std::to_string(BATCH_SPLIB_SIZE) + "_" + std::to_string(ref_batch_num) + ".bin";
            std::cout << "load precomputed reference HVs from " << dump_fname << std::endl;
            #ifdef USE_HALF
            __half* hv_matrix = (__half*)malloc(D * N_batch * sizeof(__half));
            loadArr<half>(dump_fname, hv_matrix, D * N_batch);
            HANDLE_ERROR(cudaMemcpy(d_hv_matrix, hv_matrix, D * N_batch * sizeof(__half), cudaMemcpyHostToDevice));
            free(hv_matrix);
            #endif
            #ifdef USE_FLOAT
            float* hv_matrix = (float*)malloc(D * N_batch * sizeof(float));
            loadArr<float>(dump_fname, hv_matrix, D * N_batch);
            HANDLE_ERROR(cudaMemcpy(d_hv_matrix, hv_matrix, D * N_batch * sizeof(float), cudaMemcpyHostToDevice));
            free(hv_matrix);
            #endif
            #ifdef USE_INT8
            int8_t* hv_matrix = (int8_t*)malloc(D * N_batch * sizeof(int8_t));
            loadArr<int8_t>(dump_fname, hv_matrix, D * N_batch);  // TOOD: error exception
            HANDLE_ERROR(cudaMemcpy(d_hv_matrix, hv_matrix, D * N_batch * sizeof(int8_t), cudaMemcpyHostToDevice));
            free(hv_matrix);
            #endif
        }

        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to launch kernel (error code %s)!\n",
                    cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        #ifdef USE_HALF
        if (BATCH_SPLIB_SIZE != N_batch || ref_batch_num == 0)
            HANDLE_ERROR(cudaMalloc((void **)&d_splibNorm, N_batch * sizeof(half)));
        normMatRow_fp16<<<(N_batch + __MAX_THREADS__ - 1) / __MAX_THREADS__, __MAX_THREADS__>>>(d_splibNorm, d_hv_matrix, N_batch, D);
        cosSimCalculator<half><<<(N_batch + __MAX_THREADS__ - 1) / __MAX_THREADS__, __MAX_THREADS__>>>(d_hv_matrix, d_splibNorm, D, N_batch);
        #endif

        #ifdef USE_FLOAT
        if (N_batch != prev_N_batch || ref_batch_num == 0)
            HANDLE_ERROR(cudaMalloc((void **)&d_splibNorm, N_batch * sizeof(float)));
        normMatRow_fp32<<<(N_batch + __MAX_THREADS__ - 1) / __MAX_THREADS__, __MAX_THREADS__>>>(d_splibNorm, d_hv_matrix, N_batch, D);
        cosSimCalculator<float><<<(N_batch + __MAX_THREADS__ - 1) / __MAX_THREADS__, __MAX_THREADS__>>>(d_hv_matrix, d_splibNorm, D, N_batch);
        #endif

        #ifdef USE_INT8
        if (N_batch != prev_N_batch || ref_batch_num == 0)
            HANDLE_ERROR(cudaMalloc((void **)&d_splibNorm, N_batch * sizeof(float)));
        normMatRow_int8<<<(N_batch + __MAX_THREADS__ - 1) / __MAX_THREADS__, __MAX_THREADS__>>>(d_splibNorm, d_hv_matrix, N_batch, D);
        // cosSimCalculator<int8_t><<<(N_batch + __MAX_THREADS__ - 1) / __MAX_THREADS__, __MAX_THREADS__>>>(d_hv_matrix, d_splibNorm, D, N_batch);
        #endif

        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to launch kernel (error code %s)!\n",
                    cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        cudaEventRecord(stop2);
        cudaEventSynchronize(stop2);
        cudaEventElapsedTime(&tmp_time, stop1, stop2);
        ref_encode_time += tmp_time;

        int processed_query_batch_size = 0;
        float* d_vectorized_query_intensities_batch = d_vectorized_query_intensities;
        int* d_vectorized_query_index_batch = d_vectorized_query_index;

        for (int batch_num = 0; batch_num < TOTAL_BATCH_NUM; batch_num++) {
            // This includes bin size
            int query_batch_size = (batch_num != TOTAL_BATCH_NUM - 1) ? 
                            (q_csr_info[(batch_num + 1)* BATCH_SIZE] - q_csr_info[batch_num * BATCH_SIZE]) 
                        : (query_bins - q_csr_info[batch_num * BATCH_SIZE]);
            int Nq_batch = (batch_num != TOTAL_BATCH_NUM - 1) ? BATCH_SIZE: (Nq % BATCH_SIZE);
            if ((batch_num == TOTAL_BATCH_NUM - 1) && (Nq_batch == 0)) { 
                Nq_batch = BATCH_SIZE;
            }
            if (batch_num == 0 && ref_batch_num == 0)
                prev_Nq_batch = Nq_batch;


            #ifdef USE_HALF
            if (Nq_batch != prev_Nq_batch || N_batch != prev_N_batch || (batch_num == 0 && ref_batch_num == 0)){
                if (Nq_batch != prev_Nq_batch || N_batch != prev_N_batch)
                    HANDLE_ERROR(cudaFree(d_guess_vec));
                HANDLE_ERROR(cudaMalloc((void **)&d_guess_vec, Nq_batch * N_batch * sizeof(float)));
            }
            if (Nq_batch != prev_Nq_batch || batch_num == 0) {
                if (Nq_batch != prev_Nq_batch)
                    HANDLE_ERROR(cudaFree(d_q_hv_matrix));
                HANDLE_ERROR(cudaMalloc((void **)&d_q_hv_matrix, Nq_batch * D * sizeof(__half)));
            }
            #endif
            
            #ifdef USE_FLOAT
            if (Nq_batch != prev_Nq_batch || N_batch != prev_N_batch || (batch_num == 0 && ref_batch_num == 0)) {
                if (Nq_batch != prev_Nq_batch || N_batch != prev_N_batch)
                    HANDLE_ERROR(cudaFree(d_guess_vec));
                HANDLE_ERROR(cudaMalloc((void **)&d_guess_vec, Nq_batch * N_batch * sizeof(float)));
            }
            if (Nq_batch != prev_Nq_batch || batch_num == 0) {
                if (Nq_batch != prev_Nq_batch)
                    HANDLE_ERROR(cudaFree(d_q_hv_matrix));
                HANDLE_ERROR(cudaMalloc((void **)&d_q_hv_matrix, Nq_batch * D * sizeof(float)));
            }
            #endif

            #ifdef USE_INT8
            if (Nq_batch != prev_Nq_batch || N_batch != prev_N_batch || (batch_num == 0 && ref_batch_num == 0)){
                if (Nq_batch != prev_Nq_batch || N_batch != prev_N_batch)
                    HANDLE_ERROR(cudaFree(d_guess_vec));
                HANDLE_ERROR(cudaMalloc((void **)&d_guess_vec, Nq_batch * N_batch * sizeof(int))); 
            }
            if (Nq_batch != prev_Nq_batch || batch_num == 0) {
                if (Nq_batch != prev_Nq_batch)
                    HANDLE_ERROR(cudaFree(d_q_hv_matrix));
                HANDLE_ERROR(cudaMalloc((void **)&d_q_hv_matrix, Nq_batch * D * sizeof(int8_t)));
            }
            #endif

            if (Nq_batch != prev_Nq_batch || batch_num == 0) {
                if (Nq_batch != prev_Nq_batch) {
                    HANDLE_ERROR(cudaFree(d_topIndex));
                    HANDLE_ERROR(cudaFree(d_topScore));
                }
                HANDLE_ERROR(cudaMalloc((void **)&d_topIndex, 2*Nq_batch * sizeof(int)));
                HANDLE_ERROR(cudaMalloc((void **)&d_topScore, 2*Nq_batch * sizeof(float)));
            }

            dim3 encodeQueryDim((D + __MAX_THREADS__ - 1) / __MAX_THREADS__, MIN(Nq_batch, prop.maxGridSize[1]));
            cudaEventRecord(stop1);
    #ifdef USE_HALF
            encodeLevelIdSparse_fp16_soa<<<encodeQueryDim, nBlock>>>(d_level_hvs_packed, d_id_hvs, d_vectorized_query_index_batch, d_vectorized_query_intensities_batch, d_q_csr_info, d_q_hv_matrix,
                                                            Nq_batch, Q, D, query_batch_size, batch_num * BATCH_SIZE, processed_query_batch_size);
    #endif
    #ifdef USE_FLOAT
            encodeLevelIdSparse_fp32_soa<<<encodeQueryDim, nBlock>>>(d_level_hvs_packed, d_id_hvs, d_vectorized_query_index_batch, d_vectorized_query_intensities_batch, d_q_csr_info, d_q_hv_matrix,
                                                            Nq_batch, Q, D, query_batch_size, batch_num * BATCH_SIZE, processed_query_batch_size);
    #endif
    #ifdef USE_INT8
            encodeLevelIdSparse_int8_soa<<<encodeQueryDim, nBlock>>>(d_level_hvs_packed, d_id_hvs, d_vectorized_query_index_batch, d_vectorized_query_intensities_batch, d_q_csr_info, d_q_hv_matrix,
                                                            Nq_batch, Q, D, query_batch_size, batch_num * BATCH_SIZE, processed_query_batch_size);
    #endif
    
            err = cudaGetLastError();
            if (err != cudaSuccess)
            {
                fprintf(stderr, "Failed to launch kernel (error code %s)!\n",
                        cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }
            cudaEventRecord(stop2);
            cudaEventSynchronize(stop2);
            cudaEventElapsedTime(&tmp_time, stop1, stop2);
            query_encode_time += tmp_time;

            cudaEventRecord(stop1);
            #ifdef USE_HALF
            cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, 
                    N_batch, Nq_batch, D,
                    alpha, d_hv_matrix, CUDA_R_16F, D,
                    d_q_hv_matrix, CUDA_R_16F, D, beta, 
                    d_guess_vec, CUDA_R_32F, N_batch,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);    // Nq * N_batch
            #endif
            #ifdef USE_FLOAT
            cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, 
                    N_batch, Nq_batch, D, 
                    alpha, d_hv_matrix, D, 
                    d_q_hv_matrix, D, beta, 
                    d_guess_vec, N_batch);  // Nq * N_batch
            #endif
            #ifdef USE_INT8
            cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, 
                    N_batch, Nq_batch, D,
                    alpha, d_hv_matrix, CUDA_R_8I, D,
                    d_q_hv_matrix, CUDA_R_8I, D, beta, 
                    d_guess_vec, CUDA_R_32I, N_batch,
                    CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP);    // Nq * N_batch
            #endif

            err = cudaGetLastError();
            if (err != cudaSuccess)
            {
                fprintf(stderr, "Failed to launch kernel (error code %s)!\n",
                        cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }

            dim3 nTopGrid(MIN((Nq_batch + __MAX_THREADS__ - 1) / __MAX_THREADS__, prop.maxGridSize[0]), 1);

            #ifndef USE_INT8
            getTopItemIndex<float><<<nTopGrid, nBlock>>> (d_topIndex, d_topScore, d_guess_vec, 
                                                          d_vectorized_ref_precursormz, 
                                                          d_vectorized_query_precursormz, 
                                                          splibcharge_i /*splib charge*/, tol_val_ppm, tol_val_da, 
                                                          Nq_batch, N_batch, batch_num * BATCH_SIZE, ref_batch_num * BATCH_SPLIB_SIZE);
            #else
            getTopItemIndex_int8<float><<<nTopGrid, nBlock>>>(d_topIndex, d_topScore, d_guess_vec, 
                                                              d_splibNorm,
                                                              d_vectorized_ref_precursormz, 
                                                              d_vectorized_query_precursormz, 
                                                              splibcharge_i /*splib charge*/, tol_val_ppm, tol_val_da, 
                                                              Nq_batch, N_batch, batch_num * BATCH_SIZE, ref_batch_num * BATCH_SPLIB_SIZE);
            #endif

            err = cudaGetLastError();
            if (err != cudaSuccess)
            {
                fprintf(stderr, "Failed to launch kernel (error code %s)!\n",
                        cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }
            cudaEventRecord(stop2);
            cudaEventSynchronize(stop2);
            cudaEventElapsedTime(&tmp_time, stop1, stop2);
            search_time += tmp_time;

            prev_Nq_batch = Nq_batch;
            prev_N_batch = N_batch;

            if (batch_num < TOTAL_BATCH_NUM - 1) {
                d_vectorized_query_intensities_batch = d_vectorized_query_intensities_batch + query_batch_size;  // Update for next batch
                d_vectorized_query_index_batch = d_vectorized_query_index_batch + query_batch_size;  // Update for next batch
                processed_query_batch_size += query_batch_size;
            }
            
            #ifdef CSV_OUTPUT
            int* search_idx = (int*) malloc(2*Nq_batch * sizeof(int));
            float* similarity_score = (float*) malloc(2*Nq_batch * sizeof(float));
            // Copy results on CPU
            HANDLE_ERROR(cudaMemcpy(search_idx, d_topIndex, 2*Nq_batch * sizeof(int), cudaMemcpyDeviceToHost));
            HANDLE_ERROR(cudaMemcpy(similarity_score, d_topScore, 2*Nq_batch * sizeof(float), cudaMemcpyDeviceToHost));

            // record as csv
            // query idx, query charge, ref spec_id, spectra charge, score
            for (int x = 0 ; x < Nq_batch; x++) {
                int splib_idx = search_idx[x];
                if (splib_idx >= 0) {
                    outputfile_std << (batch_num * BATCH_SIZE + x) << "\t" << querycharge << "\t";
                    outputfile_std << vectorized_ref_identifier[ref_batch_num * BATCH_SPLIB_SIZE + splib_idx] << "\t" << splibcharge << "\t";
                    outputfile_std << similarity_score[x] << "\n";
                }
                splib_idx = search_idx[Nq_batch + x];
                if (splib_idx >= 0) {
                    outputfile_open << (batch_num * BATCH_SIZE + x) << "\t" << querycharge << "\t";
                    outputfile_open << vectorized_ref_identifier[ref_batch_num * BATCH_SPLIB_SIZE + splib_idx] << "\t" << splibcharge << "\t";
                    outputfile_open << similarity_score[Nq_batch + x] << "\n";
                }
            }

            free(search_idx);
            free(similarity_score);
            #endif
        } // end of batch query loop

        // prev_N_batch = N_batch;

        if (ref_batch_num < TOTAL_REF_BATCH_NUM - 1) {
            d_vectorized_ref_intensities_batch = d_vectorized_ref_intensities_batch + ref_batch_size;  // Update for next batch
            d_vectorized_ref_index_batch = d_vectorized_ref_index_batch + ref_batch_size;  // Update for next batch
            processed_ref_batch_size += ref_batch_size;
        }

    } // end of batch ref loop


    auto stop = std::chrono::high_resolution_clock::now();
    double time_taken_ms = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count() * 1e-6;
    // std::cout << "Time taken by function: " << time_taken_ms << " ms" << std::endl;
    // std::cout << "ref_charge,ref_encode_time,query_encode_time,search_time,time_taken_ms" << std::endl;
    std::cout << splibcharge_i << "," << ref_encode_time << "," << query_encode_time << "," << search_time << "," << time_taken_ms <<  std::endl;

    // Free
    if (!use_precomputed_ref_hvs) {
        HANDLE_ERROR(cudaFree(d_csr_info));
        HANDLE_ERROR(cudaFree(d_vectorized_ref_index));
        HANDLE_ERROR(cudaFree(d_vectorized_ref_intensities));
    }
    HANDLE_ERROR(cudaFree(d_q_csr_info));
    HANDLE_ERROR(cudaFree(d_vectorized_query_index));
    HANDLE_ERROR(cudaFree(d_vectorized_query_intensities));

    HANDLE_ERROR(cudaFree(d_id_hvs));
    HANDLE_ERROR(cudaFree(d_level_hvs_packed));
    HANDLE_ERROR(cudaFree(devState));

    HANDLE_ERROR(cudaFree(d_hv_matrix));
    HANDLE_ERROR(cudaFree(d_q_hv_matrix));
    HANDLE_ERROR(cudaFree(d_guess_vec));
    HANDLE_ERROR(cudaFree(d_topIndex));
    HANDLE_ERROR(cudaFree(d_topScore));
    HANDLE_ERROR(cudaFree(d_splibNorm));

    cublasDestroy(handle);

    HANDLE_ERROR(cudaFree(d_vectorized_ref_precursormz));
    HANDLE_ERROR(cudaFree(d_vectorized_query_precursormz));

    cudaEventDestroy(stop1);
    cudaEventDestroy(stop2);

    return 0;
}
