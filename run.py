from homs_tc.definition import Config
from homs_tc.reader import SpectralLibraryReader
from homs_tc.fdr_postprocessing import fdr_postprocessing
from homs_tc.preprocessor import mgf_preprocess, splib_preprocess

import configparser
import argparse
import subprocess
import os
import gc


CNPY_LIB_PATH = "./cnpy_lib/lib"
old = os.environ.get("LD_LIBRARY_PATH")
if old:
    os.environ["LD_LIBRARY_PATH"] = old + ":" + "CNPY_LIB_PATH"
else:
    os.environ["LD_LIBRARY_PATH"] = "CNPY_LIB_PATH"


def config_parse(file_path):
    _config = configparser.ConfigParser()
    _config.read(file_path)
    return Config(_config)

def main():
    # get arguments: spectral_library_filename, query_filename, config_filename, output_filename
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", help="Spectral library filename (splib)", type=str)
    parser.add_argument("--query", help="Query filename (mgf)", type=str)
    parser.add_argument("--config", help="Config filename (ini)", type=str)
    parser.add_argument("--output", help="Output filename (mztab)", type=str)

    args = parser.parse_args()
    config = config_parse(args.config)
    # Open spectral library, get ref charges. Discard charges with less than min_spectra_ref spectra
    library_reader = SpectralLibraryReader(args.ref, config)
    library_reader.open()
    ref_charges = [charge for charge, charge_info in library_reader.spec_info['charge'].items() 
                                            if len(charge_info['id']) >= config.min_spectra_ref]

    # Check if preprocessing is done
    vec_len = config.spectrum_vector_dim
    preprocessed_query_pkl = args.query.replace(".mgf", "_vec_") + str(vec_len) + ".pkl"
    if not os.path.exists(preprocessed_query_pkl):
        print(preprocessed_query_pkl, "does not exist. Running query preprocessing first...")
        query_charges = mgf_preprocess(args.query, config)
    else:
        import pickle
        with open(preprocessed_query_pkl, "rb") as f:
            query_spectra_raw = pickle.load(f)
        query_charges = list(query_spectra_raw.keys())
        del query_spectra_raw
        gc.collect()

    # Intersection of ref_charges and query_charges
    charges_to_search = list(set(ref_charges) & set(query_charges))

    preprocessed_query_fnames = []
    for charge in charges_to_search:
        preprocessed_query_fnames.append(args.query.replace(".mgf", "_vec_") + \
                                         str(vec_len)+ ".charge" + str(charge) + ".npz")
    for preprocessed_query_fname in preprocessed_query_fnames:
        if not os.path.exists(preprocessed_query_fname):
            print(preprocessed_query_fname, "does not exist. Running query preprocessing first...")
            mgf_preprocess(args.query, config)
            break

    preprocessed_ref_fnames = []
    for charge in charges_to_search:
        preprocessed_ref_fnames.append(args.ref.replace(".splib", "_vec_") + \
                                       str(vec_len)+ ".charge" + str(charge) + ".npz")
    for preprocessed_ref_fname in preprocessed_ref_fnames:
        if not os.path.exists(preprocessed_ref_fname):
            print(preprocessed_ref_fname, "does not exist. Running ref preprocessing first...")
            splib_preprocess(args.ref, library_reader, config)
            break

    library_reader.close()  # To save memory

    # Run CUDA code
    if config.hv_precision == "fp32":
        exec_fname = "./fp32main"
    elif config.hv_precision == "fp16":
        exec_fname = "./fp16main"
    elif config.hv_precision == "int8":
        exec_fname = "./int8main"
    else:
        print("Invalid hv_precision in config file.")
        exit(-1)

    # Remove interim files
    std_interim_fname = args.output.replace(".mztab", "_std.interim")
    open_interim_fname = args.output.replace(".mztab", "_open.interim")
    if os.path.exists(std_interim_fname):
        os.remove(std_interim_fname)
    if os.path.exists(open_interim_fname):
        os.remove(open_interim_fname)
    ref_fname_wo_ext = args.ref.replace(".splib", "")
    query_fname_wo_ext = args.query.replace(".mgf", "")

    print("charge,ref_encode_time,query_encode_time,search time,total time")
    for charge in charges_to_search:
        result = subprocess.run([exec_fname, ref_fname_wo_ext, query_fname_wo_ext, str(charge), str(charge),
                                 str(config.spectrum_vector_dim), 
                                 str(config.hv_dimensionality), str(config.hv_quantize_level),
                                 str(config.precursor_tolerance_mass_ppm), str(config.precursor_tolerance_mass_open_da),
                                 str(config.max_ref_batch_size), str(config.max_query_batch_size),
                                 std_interim_fname, open_interim_fname, str(config.use_precomputed_ref_hvs)
                                ], capture_output=True, text=True)
        print(result.stdout.strip("\n"))
        if result.returncode != 0:
            print("CUDA code failed. Exiting...")
            exit(-1)

    # Run FDR filtration, Save output in mztab format
    output_fname = fdr_postprocessing(std_interim_fname, open_interim_fname, 
                                      preprocessed_query_pkl,
                                      args, config)
    print("Done. Output saved in", output_fname)

    # Remove interim files
    os.remove(std_interim_fname)
    os.remove(open_interim_fname)

if __name__ == '__main__':
    main()
