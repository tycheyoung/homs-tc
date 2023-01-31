from homs_tc.preprocess_utils import get_dim
from spectrum_utils.spectrum import MsmsSpectrum

import math


class Config:
    def __init__(self, _config):
        self.conf_dict = _config._sections

        self.min_spectra_ref = int(_config['preprocessing']['min_spectra_ref'])
        self.min_spectra_query = int(_config['preprocessing']['min_spectra_query'])

        self.resolution = int(_config['preprocessing']['resolution'])
        self.bin_size = float(_config['preprocessing']['bin_size'])
        self.min_mz = int(_config['preprocessing']['min_mz'])
        self.max_mz = int(_config['preprocessing']['max_mz'])
        self.remove_precursor = 1 if _config['preprocessing']['remove_precursor'].lower() in ("yes", "true", "t", "1") else 0
        self.remove_precursor_tolerance = float(_config['preprocessing']['remove_precursor_tolerance'])
        self.min_intensity = float(_config['preprocessing']['min_intensity'])
        self.min_peaks = int(_config['preprocessing']['min_peaks'])
        self.min_mz_range = float(_config['preprocessing']['min_mz_range'])
        self.max_peaks_used = int(_config['preprocessing']['max_peaks_used'])
        self.max_peaks_used_library = int(_config['preprocessing']['max_peaks_used_library'])
        self.scaling = _config['preprocessing']['scaling']

        self.precursor_tolerance_mass_ppm = float(_config['search']['precursor_tolerance_mass_ppm'])
        self.precursor_tolerance_mass_open_da = float(_config['search']['precursor_tolerance_mass_open_da'])

        self.max_ref_batch_size = _config['search']['max_ref_batch_size']
        self.max_query_batch_size = _config['search']['max_query_batch_size']
        self.hv_quantize_level = int(_config['search']['hv_quantize_level'])
        self.hv_dimensionality = int(_config['search']['hv_dimensionality'])
        self.hv_precision = _config['search']['hv_precision'].lower()
        self.use_precomputed_ref_hvs = 1 if _config['search']['use_precomputed_ref_hvs'].lower() in ("yes", "true", "t", "1") else 0
        self.spectrum_vector_dim, self.min_bound, _ = get_dim(self.min_mz, self.max_mz, self.bin_size)

        self.fdr_threshold = float(_config['fdr']['fdr_threshold'])
        self.fdr_tolerance_mass = float(_config['fdr']['fdr_tolerance_mass'])
        self.fdr_tolerance_mode = _config['fdr']['fdr_tolerance_mode']
        self.fdr_min_group_size = int(_config['fdr']['fdr_min_group_size'])
    
    def __str__(self):
        return str(self.__dict__)


class SpectrumSpectrumMatch:
    # Imported from ANN-SoLo (https://github.com/bittremieux/ANN-SoLo)
    def __init__(self, query_spectrum: MsmsSpectrum,
                 library_spectrum: MsmsSpectrum = None,
                 search_engine_score: float = math.nan,
                 q: float = math.nan):
        self.query_spectrum = query_spectrum
        self.library_spectrum = library_spectrum
        self.search_engine_score = search_engine_score
        self.q = q

    @property
    def sequence(self):
        return (self.library_spectrum.peptide if self.library_spectrum is not None else None)

    @property
    def query_identifier(self):
        return self.query_spectrum.identifier

    @property
    def query_index(self):
        return self.query_spectrum.index

    @property
    def library_identifier(self):
        return (self.library_spectrum.identifier
                if self.library_spectrum is not None else None)

    @property
    def retention_time(self):
        return self.query_spectrum.retention_time

    @property
    def charge(self):
        return self.query_spectrum.precursor_charge

    @property
    def exp_mass_to_charge(self):
        return self.query_spectrum.precursor_mz

    @property
    def calc_mass_to_charge(self):
        return (self.library_spectrum.precursor_mz if self.library_spectrum is not None else None)

    @property
    def is_decoy(self):
        return (self.library_spectrum.is_decoy if self.library_spectrum is not None else None)
