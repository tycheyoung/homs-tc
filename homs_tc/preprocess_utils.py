import functools
import numpy as np
import numba as nb
import math

from spectrum_utils.spectrum import MsmsSpectrum


@nb.njit
def _check_spectrum_valid(spectrum_mz: np.ndarray, min_peaks: int,
                          min_mz_range: float) -> bool:
    """
    Check whether a spectrum is of high enough quality to be used for matching.
    Imported from ANN-SoLo (https://github.com/bittremieux/ANN-SoLo)
    """
    return (len(spectrum_mz) >= min_peaks and spectrum_mz[-1] - spectrum_mz[0] >= min_mz_range)


@nb.njit
def _norm_intensity(spectrum_intensity: np.ndarray) -> np.ndarray:
    """
    Normalize spectrum peak intensities.
    Imported from ANN-SoLo (https://github.com/bittremieux/ANN-SoLo)
    """
    return spectrum_intensity / np.linalg.norm(spectrum_intensity)


def process_spectrum(spectrum: MsmsSpectrum, config, is_library: bool) -> MsmsSpectrum:
    """
    Process the peaks of the MS/MS spectrum according to the config.
    Imported from ANN-SoLo (https://github.com/bittremieux/ANN-SoLo)
    """
    if spectrum.is_processed:
        return spectrum

    min_peaks = config.min_peaks
    min_mz_range = config.min_mz_range

    spectrum = spectrum.set_mz_range(config.min_mz, config.max_mz)
    if not _check_spectrum_valid(spectrum.mz, min_peaks, min_mz_range):
        spectrum.is_valid = False
        spectrum.is_processed = True
        return spectrum
    if config.resolution is not None:
        spectrum = spectrum.round(config.resolution, 'sum')
        if not _check_spectrum_valid(spectrum.mz, min_peaks, min_mz_range):
            spectrum.is_valid = False
            spectrum.is_processed = True
            return spectrum

    if config.remove_precursor:
        spectrum = spectrum.remove_precursor_peak(
            config.remove_precursor_tolerance, 'Da', 2)
        if not _check_spectrum_valid(spectrum.mz, min_peaks, min_mz_range):
            spectrum.is_valid = False
            spectrum.is_processed = True
            return spectrum

    spectrum = spectrum.filter_intensity(
        config.min_intensity, (config.max_peaks_used_library if is_library else
                               config.max_peaks_used))
    if not _check_spectrum_valid(spectrum.mz, min_peaks, min_mz_range):
        spectrum.is_valid = False
        spectrum.is_processed = True
        return spectrum

    scaling = config.scaling
    if scaling == 'sqrt':
        scaling = 'root'
    if scaling is not None:
        spectrum = spectrum.scale_intensity(
            scaling, max_rank=(config.max_peaks_used_library if is_library else
                               config.max_peaks_used))

    spectrum.intensity = _norm_intensity(spectrum.intensity)

    # Set a flag to indicate that the spectrum has been processed to avoid
    # reprocessing of library spectra for multiple queries.
    spectrum.is_valid = True
    spectrum.is_processed = True

    return spectrum


@functools.lru_cache(maxsize=None)
def get_dim(min_mz, max_mz, bin_size):
    min_mz, max_mz = float(min_mz), float(max_mz)
    start_dim = min_mz - min_mz % bin_size
    end_dim = max_mz + bin_size - max_mz % bin_size
    return round((end_dim - start_dim) / bin_size), start_dim, end_dim


def spectrum_to_vector(spectrum: MsmsSpectrum, min_mz: float, max_mz: float,
                       bin_size: float, vec_len: int, min_bound) -> np.ndarray:
    mz_map = {}
    for mz, intensity in zip(spectrum.mz, spectrum.intensity):
        bin_idx = math.floor((mz - min_bound) // bin_size)
        if bin_idx not in mz_map:
            mz_map[bin_idx] = intensity
        else:
            mz_map[bin_idx] += intensity
    vec_tmp = [(k, v) for k, v in mz_map.items()]
    return vec_tmp
