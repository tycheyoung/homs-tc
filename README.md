Feeding BER into the HOMS-TC code
=======================================================

Modify _HOMS-TC_ code to recreate some error results. Bit error rate (BER) comes from RRAM, which might effect the data stored in memory. Initially the code assume 0 BER, I modified the code to inject noise into the encoding phase and observe the differences. To inject BER into the code, I flipped the binary value [-1,1] randomly with 50% of it being 1 and 50% being -1.

System Requirements
------------------------------------------------------

_HOMS-TC_ requires `Python 3.8+` with `CUDA v11+` environment. A NVIDIA GPU with Tensor Core is required for the best performance. _HOMS-TC_ has been tested on NVIDIA RTX 3090 and NVIDIA RTX 4090 with CUDA v11.8. 

Installation
------------------------------------------------------

Install via Docker
*********************

We recommend installing _HOMS-TC_ via docker using the following command:

```bash
git clone --recurse-submodules https://github.com/tycheyoung/homs-tc.git
cd homs-tc
docker build -f ./docker/Dockerfile -t homs_tc .
docker run --gpus all -it homs_tc /bin/bash  # Make sure to mount dataset folder
```

Install from Source
*********************
First, be sure to install all dependencies (Python and CUDA). In Ubuntu:

```bash
sudo apt-get update
sudo apt-get install python3 python3-dev python3-pip 
sudo apt-get install nvidia-cuda-toolkit  # This will install the latest version of CUDA. Read below before proceed
```
For CUDA installation, refer to the documentation [\[LINK\]](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) to install a specific version.

Then, to install `homs-tc`:

```bash
git clone --recurse-submodules https://github.com/tycheyoung/homs-tc.git
cd homs-tc
./install.sh
```

Usage
------------------------------------------------------

    usage: python run.py [-h] [--ref REF] [--query QUERY] 
                              [--config CONFIG] [--output OUTPUT]
                
    Positional arguments:
    ref         The path of spectral library file (in `.splib` format) to be used to identify the experimental spectra (reference).
    query       The path of the spectral file (in `.mgf` format) to be searched (query).
    config      The path of config file (in `.ini` format). 
    output      The path of the `mztab` output file containing the search result.

Config file example
------------------------------------------------------

See `configs/iprg2012.ini` for the example config file. All parameters should be defined.
* preprocessing
  - min_spectra_ref: the minimum number of referenece spectra required for each charge. If it is below the threshold, the reference spectra with the corresponding charge are excluded from the search.
  - min_spectra_query: the minimum number of query spectra required for each charge. If it is below the threshold, the query spectra with the corresponding charge are excluded from the search.
  - resolution: the number of decimal places to round the `m/z`
  - bin_size: the bin width (in `Da`) for raw spectra to spectrum vector conversion
  - min_mz: the min m/z value to consider (inclusive) during preprocessing
  - max_mz: the max m/z value to consider (inclusive) during preprocessing
  - remove_precursor: a flag (boolean) to eliminate peaks near the precursor mass. Can be either `true` or `false`
  - remove_precursor_tolerance: a `m/z` window to eliminate peaks near the precursor mass
  - min_intensity: remove peaks with a lower intensity relative to the base peak intensity
  - min_peaks: a cutoff for discarding spectra with with fewer peaks
  - min_mz_range: a threshold to discard low-quality spectra with narrow mass range
  - max_peaks_used: the specified limit of the most intense peaks to retain for query spectra
  - max_peaks_used_library: the specified limit of the most intense peaks to retain for reference spectra
  - scaling: the method of scaling peak intensities, either square root ("sqrt") or rank-based ("rank")

* search
  - precursor_tolerance_mass_ppm: narrow window size (standard search - 1st stage of cascade search)
  - precursor_tolerance_mass_open_da: wide window size (open search - 2nd stage of cascade search)
  - max_ref_batch_size: batch size for reference hypervectors during the search stage
  - max_query_batch_size: batch size for query hypervectors during the search stage
  - hv_quantize_level: the quantization level of hypervector during the encoding stage
  - hv_dimensionality: the hypervector dimensionality
  - hv_precision: the hypervector precision. Can be either `fp32` or `fp16` or `int8`
  - use_precomputed_ref_hvs: a flag (boolean) to use dumped reference HVs. If there are no existing dump files, it will generate dump first. Can be either `true` or `false`. Usually, generating reference on-the-fly is faster than loading precomputed HVs.

* fdr
  - fdr_threshold: the FDR threshold for each stage of the cascade search
  - fdr_tolerance_mass: the bin width to group SSMs for subgroup FDR calculation during the second stage of the cascade search.
  - fdr_tolerance_mode: the unit of `fdr_tolerance_mass`. Can be either `Da` or `ppm`
  - fdr_min_group_size: the minimum group size to perform FDR control individually for that subgroup.

Result
----
|Run/BER| Run 1 | Run 2 | Run 3 | Run 4 | Run 5 |
|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
|  0    | 4289  |  4436 |  4667 |  4508 | 4727  |
|  0.5  | 4589  | 4675  | 4607  | 4428  | 4713  |



