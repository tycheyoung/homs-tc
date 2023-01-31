from pyteomics import mgf
from spectrum_utils.spectrum import MsmsSpectrum
from homs_tc.definition import Config
from homs_tc.preprocess_utils import process_spectrum
from homs_tc.parser import SplibParser

import collections
import logging
import os
import pickle
from functools import lru_cache
from typing import Iterator, Tuple
import joblib
import numpy as np


def read_mgf(filename: str) -> Iterator[MsmsSpectrum]:
    """
    Read all spectra from the given mgf file.
    Imported from ANN-SoLo (https://github.com/bittremieux/ANN-SoLo)
    """
    # Test if the given file is an mgf file.
    _, ext = os.path.splitext(filename)
    assert (ext in ['.mgf'])

    # Get all query spectra.
    for i, mgf_spectrum in enumerate(mgf.read(filename)):
        # Create spectrum.
        identifier = mgf_spectrum['params']['title']
        precursor_mz = float(mgf_spectrum['params']['pepmass'][0])
        retention_time = float(mgf_spectrum['params']['rtinseconds'])
        if 'charge' in mgf_spectrum['params']:
            precursor_charge = int(mgf_spectrum['params']['charge'][0])
        else:
            precursor_charge = None

        spectrum = MsmsSpectrum(identifier, precursor_mz, precursor_charge,
                                mgf_spectrum['m/z array'],
                                mgf_spectrum['intensity array'],
                                retention_time=retention_time)
        spectrum.index = i
        spectrum.is_processed = False

        yield spectrum


class SpectralLibraryReader:
    """
    Read spectra from a SpectraST spectral library .splib file.
    Imported from ANN-SoLo (https://github.com/bittremieux/ANN-SoLo)
    """
    is_recreated = False

    def __init__(self, filename: str, homs_tc_config: Config, config_hash: str = None) -> None:
        self.homs_tc_config = homs_tc_config
        self._filename = filename
        self._config_hash = config_hash
        self._parser = None
        do_create = False

        # Test if the given spectral library file is in a supported format.
        _, ext = os.path.splitext(self._filename)
        assert (ext in ['.splib'])
        logging.debug('Load the spectral library configuration')

        # Verify that the configuration file
        # corresponding to this spectral library is present.
        config_filename = self._get_config_filename()
        if not os.path.isfile(config_filename):
            # If not we should recreate this file
            # prior to using the spectral library.
            do_create = True
            logging.warning('Missing spectral library configuration file')
        else:
            # Load the configuration file.
            config_lib_filename, self.spec_info, load_hash =\
                joblib.load(config_filename)

            # Check that the same spectral library file format is used.
            if config_lib_filename != os.path.basename(self._filename):
                do_create = True
                logging.warning('The configuration corresponds to a different '
                                'file format of this spectral library')
            # Verify that the runtime settings match the loaded settings.
            if self._config_hash != load_hash:
                do_create = True
                logging.warning('The spectral library search engine was '
                                'created using non-compatible settings')

        # (Re)create the spectral library configuration
        # if it is missing or invalid.
        if do_create:
            self._create_config()

    def _get_config_filename(self) -> str:
        """
        Gets the configuration filename (.spcfg) for the spectral library with the
        current configuration. 
        """
        if self._config_hash is not None:
            return (f'{os.path.splitext(self._filename)[0]}_'
                    f'{self._config_hash[:7]}.spcfg')
        else:
            return f'{os.path.splitext(self._filename)[0]}.spcfg'

    def _create_config(self) -> None:
        """
        Create a new configuration file for the spectral library.
        The configuration file contains for each spectrum in the spectral
        library its offset for quick random-access reading, and its precursor
        m/z for filtering using a precursor mass window. Finally, it also
        contains the settings used to construct this spectral library to make
        sure these match the runtime settings.
        """
        logging.info('Create the spectral library configuration for file %s',
                     self._filename)

        self.is_recreated = True

        # Read all the spectra in the spectral library.
        temp_info = collections.defaultdict(lambda: {'id': [], 'precursor_mz': []})
        offsets = {}
        with self as lib_reader:
            for spectrum, offset in lib_reader.get_all_spectra():
                # Store the spectrum information for easy retrieval.
                info_charge = temp_info[spectrum.precursor_charge]
                info_charge['id'].append(spectrum.identifier)
                info_charge['precursor_mz'].append(spectrum.precursor_mz)
                offsets[spectrum.identifier] = offset
        self.spec_info = {
            'charge': {charge: {'id': np.asarray(charge_info['id'], np.uint32),
                                'precursor_mz': np.asarray(charge_info['precursor_mz'], np.float32)} for charge, charge_info in temp_info.items()},
            'offset': offsets}

        # Store the configuration.
        config_filename = self._get_config_filename()
        logging.debug('Save the spectral library configuration to file %s', config_filename)
        joblib.dump(
            (os.path.basename(self._filename), self.spec_info,
             self._config_hash),
            config_filename, compress=9, protocol=pickle.DEFAULT_PROTOCOL)

    def open(self) -> None:
        self._parser = SplibParser(self._filename.encode())

    def close(self) -> None:
        if self._parser is not None:
            del self._parser

    def __enter__(self) -> 'SpectralLibraryReader':
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    @lru_cache(maxsize=None)
    def get_spectrum(self, spec_id: int, process_peaks: bool = False)\
            -> MsmsSpectrum:
        """
        Read the spectrum with the specified identifier from the spectral library file.
        """
        spectrum = self._parser.read_spectrum(
            self.spec_info['offset'][spec_id])[0]
        spectrum.is_processed = False
        if process_peaks:
            process_spectrum(spectrum, self.homs_tc_config, True)

        return spectrum

    def get_all_spectra(self) -> Iterator[Tuple[MsmsSpectrum, int]]:
        """
        Generates all spectra from the spectral library file.
        """
        self._parser.seek_first_spectrum()
        try:
            while True:
                spectrum, offset = self._parser.read_spectrum()
                spectrum.is_processed = False
                yield spectrum, offset
        except StopIteration:
            return
