from collections import defaultdict
from homs_tc.preprocess_utils import process_spectrum, spectrum_to_vector
from homs_tc.reader import read_mgf

import numpy as np
import copy
import pickle


def mgf_preprocess(query_filename, config):
    vectorized_spectra_idx = defaultdict(list)
    vectorized_spectra_intensities = defaultdict(list)
    csr_info = defaultdict(list)
    spectra_precursor_mz = defaultdict(list)
    spectra_identifier = defaultdict(list)

    # build MsmsSpectrum List of query
    # classify by its charge
    print("[Query] Classify query by its charge")
    query_spectra = defaultdict(list)
    for query_spectrum in read_mgf(query_filename):

        # For queries with an unknown charge, try all possible charges.
        if query_spectrum.precursor_charge is not None:
            query_spectra_charge = [query_spectrum]
        else:
            query_spectra_charge = []
            for charge in (2, 3):
                query_spectra_charge.append(copy.copy(query_spectrum))
                query_spectra_charge[-1].precursor_charge = charge
        for query_spectrum_charge in query_spectra_charge:
            # Discard low-quality spectra.
            if process_spectrum(query_spectrum_charge, config, False).is_valid:
                (query_spectra[query_spectrum_charge.precursor_charge].append(query_spectrum_charge))

    print("[Query] Vectorizing: Raw spectra to spectrum vector")
    for charge in query_spectra.keys():
        csr_info[charge].append(0)  # put 0 to the start of each list in csr_info
        for query_spectrum in query_spectra[charge]:
            precursor_charge = query_spectrum.precursor_charge
            vec_tmp = spectrum_to_vector(query_spectrum, config.min_mz, config.max_mz, config.bin_size, config.spectrum_vector_dim, config.min_bound)
            
            vectorized_spectra_idx[precursor_charge].extend([x[0] for x in vec_tmp])
            vectorized_spectra_intensities[precursor_charge].extend([x[1] for x in vec_tmp])
            csr_info[precursor_charge].append(csr_info[precursor_charge][-1] + len(vec_tmp))
            spectra_identifier[precursor_charge].append(query_spectrum.identifier)
            spectra_precursor_mz[precursor_charge].append(query_spectrum.precursor_mz)

        charge_pr_mzs = np.array(spectra_precursor_mz[charge], dtype=np.float32)
        charge_spectra_idx = np.array(vectorized_spectra_idx[charge], dtype=np.int32)
        charge_spectra_intensities = np.array(vectorized_spectra_intensities[charge], dtype=np.float32)
        charge_csr_info = np.array(csr_info[charge], dtype=np.uint32)

        print("[Query] Convert charge: " + str(charge) + " to npz")
        # save to npz
        np.savez_compressed(query_filename.replace(".mgf", "_vec_") + str(config.spectrum_vector_dim)+ ".charge" + str(charge),
                            pr_mzs=charge_pr_mzs,
                            spectra_idx=charge_spectra_idx,
                            spectra_intensities=charge_spectra_intensities,
                            csr_info=charge_csr_info)

    # save query_spectra to pkl
    print("[Query] Save query_spectra to pkl")
    with open(query_filename.replace(".mgf", "_vec_") + str(config.spectrum_vector_dim)+ ".pkl", 'wb') as f:
        pickle.dump(query_spectra, f)

    return list(query_spectra.keys())  # Return list of query charges


def splib_preprocess(ref_filename, library_reader, config):
    create_ann_charges = []
    ann_charges = [charge for charge, charge_info in library_reader.spec_info['charge'].items() 
                                            if len(charge_info['id']) >= config.min_spectra_ref]
    for charge in sorted(ann_charges):
        create_ann_charges.append(charge)


    vectorized_spectra_idx = defaultdict(list)
    vectorized_spectra_intensities = defaultdict(list)
    csr_info = defaultdict(list)
    spectra_precursor_mz = defaultdict(list)
    spectra_identifier = defaultdict(list)

    for charge in create_ann_charges:
        csr_info[charge].append(0)

    print("[Ref] Vectorizing: Raw spectra to spectrum vector")
    for lib_spectrum, _ in library_reader.get_all_spectra():
        precursor_charge = lib_spectrum.precursor_charge
        if precursor_charge in create_ann_charges:
            a = process_spectrum(lib_spectrum, config, True)
            if a.is_valid:
                vec_tmp = spectrum_to_vector(a, config.min_mz, config.max_mz, config.bin_size, config.spectrum_vector_dim, config.min_bound)

                vectorized_spectra_idx[precursor_charge].extend([x[0] for x in vec_tmp])
                vectorized_spectra_intensities[precursor_charge].extend([x[1] for x in vec_tmp])
                csr_info[precursor_charge].append(csr_info[precursor_charge][-1] + len(vec_tmp))
                spectra_identifier[precursor_charge].append(lib_spectrum.identifier)
                spectra_precursor_mz[precursor_charge].append(lib_spectrum.precursor_mz)

    for charge in create_ann_charges:
        print("[Ref] Convert charge: " + str(charge) + " to npz")
        charge_pr_mzs = np.array(spectra_precursor_mz[charge], dtype=np.float32)
        charge_spectra_idx = np.array(vectorized_spectra_idx[charge], dtype=np.int32)
        charge_spectra_intensities = np.array(vectorized_spectra_intensities[charge], dtype=np.float32)
        charge_spectra_identifier = np.array(spectra_identifier[charge], dtype=np.int32)
        charge_csr_info = np.array(csr_info[charge], dtype=np.uint32)

        # save to npz
        np.savez_compressed(ref_filename.replace(".splib", "_vec_") + str(config.spectrum_vector_dim)+ ".charge" + str(charge),
                            pr_mzs=charge_pr_mzs,
                            spectra_idx=charge_spectra_idx,
                            spectra_intensities=charge_spectra_intensities,
                            spectra_identifier=charge_spectra_identifier,
                            csr_info=charge_csr_info)
