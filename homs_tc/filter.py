from homs_tc.definition import SpectrumSpectrumMatch

from typing import Iterator
import pyteomics.auxiliary
import numpy as np
import operator
import itertools


def filter_fdr(ssms: Iterator[SpectrumSpectrumMatch], fdr: float = 0.01)\
        -> Iterator[SpectrumSpectrumMatch]:
    """
    Filter SSMs exceeding the given FDR using pyteomics. (for std search)
    """
    for _, _, q, ssm in pyteomics.auxiliary.qvalues(
            ssms, key=operator.attrgetter('search_engine_score'), reverse=True,
            is_decoy=operator.attrgetter('is_decoy'), remove_decoy=True,
            formula=1, correction=0, full_output=True):
        ssm.q = q
        if q <= fdr:
            yield ssm
        else:
            break


def filter_group_fdr(ssms: Iterator[SpectrumSpectrumMatch], fdr: float = 0.01,
                     tol_mass: float = None, tol_mode: str = None,
                     min_group_size: int = None)\
        -> Iterator[SpectrumSpectrumMatch]:
    """
    Filter SSMs exceeding the given FDR using pyteomics. (for open search)
    Imported from ANN-SoLo (https://github.com/bittremieux/ANN-SoLo)
    """
    ssms_remaining = np.asarray(sorted(
        ssms, key=operator.attrgetter('search_engine_score'), reverse=True))
    exp_masses = np.asarray([ssm.exp_mass_to_charge for ssm in ssms_remaining])
    mass_diffs = np.asarray([(ssm.exp_mass_to_charge - ssm.calc_mass_to_charge)
                             * ssm.charge for ssm in ssms_remaining])

    # Start with the highest ranked SSM.
    groups_common, groups_uncommon = [], []
    while ssms_remaining.size > 0:
        # Find all remaining PSMs within the mass difference window.
        if (tol_mass is None or tol_mode not in ('Da', 'ppm') or min_group_size is None):
            mask = np.full(len(ssms_remaining), True, dtype=bool)
        elif tol_mode == 'Da':
            mask = np.fabs(mass_diffs - mass_diffs[0]) <= tol_mass
        elif tol_mode == 'ppm':
            mask = (np.fabs(mass_diffs - mass_diffs[0]) / exp_masses * 10 ** 6 <= tol_mass)
        if np.count_nonzero(mask) >= min_group_size:
            groups_common.append(ssms_remaining[mask])
        else:
            groups_uncommon.extend(ssms_remaining[mask])
        # Exclude the selected SSMs from further selections.
        ssms_remaining = ssms_remaining[~mask]
        exp_masses = exp_masses[~mask]
        mass_diffs = mass_diffs[~mask]

    # Calculate the FDR combined for all uncommon mass difference groups
    # and separately for each common mass difference group.
    for ssm in itertools.chain(filter_fdr(groups_uncommon, fdr),
                               *[filter_fdr(group, fdr) for group in groups_common]):
        yield ssm
