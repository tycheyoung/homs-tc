import homs_tc.writer
from homs_tc.reader import SpectralLibraryReader
from homs_tc.definition import SpectrumSpectrumMatch
from homs_tc.filter import filter_fdr, filter_group_fdr

import pandas as pd
import pickle
import numpy as np


def cuda_output_to_df(cuda_output_fname):
    header_list = ["query_idx", "query_charge", "splib_identifier", "splib_charge", "score"]
    df_dtypes = {"query_idx":np.int32, "query_charge":np.int32, "splib_identifier":np.int64, "splib_charge":np.int32, "score":np.float64}
    df = pd.read_csv(cuda_output_fname, names=header_list, dtype=df_dtypes, delimiter="\t", header=None)
    c_maxes = df.groupby(['query_charge', 'query_idx']).score.transform(max)  # if batched, multiple max found in the subset of ref
    df = df.loc[df.score == c_maxes]
    return df


def build_ssm(df_std, df_open, library_reader, query_pkl_name):
    with open(query_pkl_name, 'rb') as f:
        query_spectra = pickle.load(f)
    # Build SSM
    print("Build SSM (std)")
    ssms_std = []
    for row in df_std.itertuples():
        found_query_charge = row.query_charge
        query_msms = query_spectra[found_query_charge][row.query_idx]
        
        ref_msms = None
        if row.splib_identifier != -1:
            ref_tmp = library_reader.get_spectrum(row.splib_identifier, True)
            if ref_tmp.is_valid:
                ref_msms = ref_tmp
        else:
            print(row.splib_identifier, ": null splib")
        
        if ref_msms is not None:
            ssm = SpectrumSpectrumMatch(query_msms, ref_msms, row.score)
            ssms_std.append(ssm)

    print("Build SSM (open)")
    ssms_open = []
    for row in df_open.itertuples():
        found_query_charge = row.query_charge
        query_msms = query_spectra[found_query_charge][row.query_idx]
        
        ref_msms = None
        if row.splib_identifier != -1:
            ref_tmp = library_reader.get_spectrum(row.splib_identifier, True)
            if ref_tmp.is_valid:
                ref_msms = ref_tmp
        else:
            print(row.splib_identifier, ": null splib")

        if ref_msms is not None:
            ssm = SpectrumSpectrumMatch(query_msms, ref_msms, row.score)
            ssms_open.append(ssm)

    return ssms_std, ssms_open

def filter_fdr_wrapper(ssms, fdr):
    identifications_list = []
    identified_queries_std = set()
    for ssm in filter_fdr(ssms, fdr):
        if ssm.query_identifier not in identified_queries_std:
            identified_queries_std.add(ssm.query_identifier)
        identifications_list.append(ssm)
    return identifications_list, identified_queries_std

def filter_group_fdr_wrapper(identifications_list, identified_queries_std, ssms, 
                             fdr, fdr_tolerance_mass, fdr_tolerance_mode, fdr_min_group_size):
    for ssm in filter_group_fdr(ssms, fdr, fdr_tolerance_mass, fdr_tolerance_mode, fdr_min_group_size):
        if ssm.query_identifier not in identified_queries_std:
            identifications_list.append(ssm)
    return identifications_list

def fdr_postprocessing(cuda_std_output, cuda_open_output, query_pkl_name, args, config):
    df_std = cuda_output_to_df(cuda_std_output)
    df_open = cuda_output_to_df(cuda_open_output)

    # Build SSM
    # build MsmsSpectrum List of reference
    library_reader = SpectralLibraryReader(args.ref, config)
    library_reader.open()

    ssms_std, ssms_open = build_ssm(df_std, df_open, library_reader, query_pkl_name)
    library_reader.close()

    # Filter FDR
    identifications, identified_queries_std = filter_fdr_wrapper(ssms_std, config.fdr_threshold)

    # Filter group FDR
    identifications = filter_group_fdr_wrapper(identifications, identified_queries_std, ssms_open, 
                                               config.fdr_threshold,
                                               config.fdr_tolerance_mass,
                                               config.fdr_tolerance_mode,
                                               config.fdr_min_group_size)
    # Write to file
    print("Total identifications: ", len(identifications))
    output_fname = homs_tc.writer.write_mztab(identifications, args, config)

    return output_fname
