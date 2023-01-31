from homs_tc.definition import SpectrumSpectrumMatch, Config

from argparse import ArgumentParser
import pathlib
import os
import re
from typing import Union, List, Pattern, AnyStr

from . import __version__


def natural_sort_key(s: str, _nsre: Pattern[AnyStr] = re.compile('([0-9]+)'))\
        -> List[Union[str, int]]:
    """
    Key to be used for natural sorting of mixed alphanumeric strings.
    A list of separate int (numeric) and string (alphabetic) parts of the given string.

    Source: https://stackoverflow.com/a/16090640
    """
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(_nsre, s)]

def write_mztab(identifications: List[SpectrumSpectrumMatch],
                args: ArgumentParser,
                config: Config) -> str:
    """
    Write the given SSMs to an mzTab file.
    Imported from ANN-SoLo (https://github.com/bittremieux/ANN-SoLo)
    """
    filename = args.output
    splib_fname = args.ref
    query_fname = args.query

    # Check if the filename contains the mztab extension and add if required.
    if os.path.splitext(filename)[1].lower() != '.mztab':
        filename += '.mztab'

    # Collect the necessary metadata.
    metadata = [
        ('mzTab-version', '1.0.0'),
        ('mzTab-mode', 'Summary'),
        ('mzTab-type', 'Identification'),
        ('mzTab-ID', f'HOMS-TC_{filename}'),
        ('title', f'HOMS-TC identification file "{filename}"'),
        ('description', f'Identification results of file '
                        f'"{os.path.split(query_fname)[1]}" against '
                        f'spectral library file '
                        f'"{os.path.split(splib_fname)[1]}"'),
        ('software[1]', f'[MS, MS:1001456, HOMS-TC, {__version__}]'),
        ('psm_search_engine_score[1]', '[MS, MS:1001143, search engine '
                                       'specific score for PSMs,]'),
        ('psm_search_engine_score[2]', '[MS, MS:1002354, PSM-level q-value,]'),
        ('ms_run[1]-format', '[MS, MS:1001062, Mascot MGF file,]'),
        ('ms_run[1]-location', pathlib.Path(
            os.path.abspath(query_fname)).as_uri()),
        ('ms_run[1]-id_format', '[MS, MS:1000774, multiple peak list nativeID '
                                'format,]'),
        ('fixed_mod[1]', '[MS, MS:1002453, No fixed modifications searched,]'),
        ('variable_mod[1]', '[MS, MS:1002454, No variable modifications '
                            'searched,]'),
        ('false_discovery_rate', f'[MS, MS:1002350, PSM-level global FDR, '
                                 f'{config.fdr_threshold}]'),
    ]

    i = 0
    for section_key in config.conf_dict.keys():
        for key, item in config.conf_dict[section_key].items():
            metadata.append((f'software[1]-setting[{i}]', f'{key} = {item}'))
            i += 1

    database_version = 'null'  # the spectral library version

    with open(filename, 'w') as f_out:
        # Metadata section.
        for m in metadata:
            f_out.write('\t'.join(['MTD'] + list(m)) + '\n')

        # SSMs.
        f_out.write('\t'.join([
            'PSH', 'sequence', 'PSM_ID', 
            'accession', 'unique', 'database',
            'database_version', 'search_engine', 'search_engine_score[1]',
            'search_engine_score[2]', 'modifications', 'retention_time',
            'charge', 'exp_mass_to_charge', 'calc_mass_to_charge',
            'spectra_ref', 'pre', 'post', 'start', 'end',
            'opt_ms_run[1]_cv_MS:1003062_spectrum_index',
            'opt_ms_run[1]_cv_MS:1002217_decoy_peptide', 'opt_ms_run[1]_num_candidates']) + '\n')
        # SSMs sorted by their query identifier.
        for ssm in sorted(identifications, key=lambda s: natural_sort_key(s.query_identifier)):
            f_out.write('\t'.join(['PSM', ssm.sequence, str(ssm.query_identifier),
                                   'null', 'null', pathlib.Path(os.path.abspath(splib_fname)).as_uri(),
                                   database_version, '[MS, MS:1001456, HOMS-TC,]', str(ssm.search_engine_score),
                                   str(ssm.q), 'null', str(ssm.retention_time),
                                   str(ssm.charge), str(ssm.exp_mass_to_charge), str(ssm.calc_mass_to_charge),
                                   f'ms_run[1]:index={ssm.query_index}', 'null', 'null', 'null', 'null',
                                   str(ssm.library_identifier),
                                   f'{ssm.is_decoy:d}', "0"]) + '\n')

    return filename