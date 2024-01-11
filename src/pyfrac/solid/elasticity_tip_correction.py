# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Carlo Peruzzo on Tue Dec 15 10:18:56 2021.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2022.
All rights reserved. See the LICENSE.TXT file for more details.
"""

# external imports
import numpy as np
from numba import prange, njit
from solid.elasticity_kernels.isotropic_R0_elem import load_isotropic_elasticity_matrix


@njit(parallel=True, cache = True, nogil=True, fastmath=True)
def tip_correction_factors(FillFrac):
    """
    This function is used to apply the <<filling fraction tip correction>> according to:
    Rider & Napier, 1985, Error Analysis and Design of Large-Scale Tabular Mining Stress Analyser.

    :param FillFrac: vector containing the filling fraction
    :return: array called diag_corr containing the diagonal factors such that

             with NO tip correction it would be:
                C * v = res


             WITH tip correction (two steps procedure):
                indxs = EltTip (EltTip=list of tip elements IDs)
                C * v = res
                res[indxs] = res[indxs] + C[0,0] * diag_corr * v[indxs]

             Note:
                - C[0,0] is the self effect in the elasticity matrix. Remember that it is constant.
                - the above procedure is equivalent to:
                    for e in EltTip:
                        C[EltTip[e], EltTip[e]] *= (1. + diag_corr[e])
                    C * v = res
                - this function is applied to provide a correction to all the elements at the front:
                   diag_corr[e] == 0. if FillFrac[e] == 1.
                               != 0. if FillFrac[e] < 1.


    """

    coeff = np.pi / 4.

    n_EltTip = len(FillFrac)
    diag_corr = np.zeros(n_EltTip)

    r = FillFrac - 0.25

    for e in prange(n_EltTip):
        r_e = r[e]
        if r_e < 0.1:
            r_e = 0.1
        ac = (1. - r_e) / r_e
        diag_corr[e] = ac * coeff

    return diag_corr


def tip_correction_(C_Crack, EltCrack, EltTip, FillFrac):
    """
    This function is used to apply the <<filling fraction tip correction>> according to:
    Rider & Napier, 1985, Error Analysis and Design of Large-Scale Tabular Mining Stress Analyser.

    :param C_Crack: squared elasticity matrix C[np.ix_(EltCrack, EltCrack)] of size len(EltCrack) x len(EltCrack)
    :param EltCrack: list of elements in the crack
    :param EltTip: list of elements in the tip
    :param FillFrac: list of filling fractions ordered as EltTip. 1 means
    :return: corrected matrix ready for the dot product

    """
    coeff = np.pi / 4.

    EltTip_positions = np.where(np.in1d(EltCrack, EltTip))[0]

    # filling fraction correction for element in the tip region
    r = FillFrac - .25
    indx = np.where(np.less(r, 0.1))[0]
    r[indx] = 0.1
    ac = (1 - r) / r
    C_Crack[EltTip_positions, EltTip_positions] = C_Crack[EltTip_positions, EltTip_positions] * (1. + ac * coeff)

    return C_Crack



def full_C_and_tip_correction_for_test(Mesh, Ep, EltTip, FillFrac):
    """
    This function is used to apply the <<filling fraction tip correction>> according to:
    Rider & Napier, 1985, Error Analysis and Design of Large-Scale Tabular Mining Stress Analyser.

    :return: corrected matrix ready for the dot product

    """
    C = load_isotropic_elasticity_matrix(Mesh, Ep, C_precision=np.float64)

    coeff = np.pi / 4.

    #EltTip_positions = np.where(np.in1d(EltCrack, EltTip))[0]

    # filling fraction correction for element in the tip region
    r = FillFrac - .25
    indx = np.where(np.less(r, 0.1))[0]
    r[indx] = 0.1
    ac = (1 - r) / r
    C[EltTip, EltTip] = C[EltTip, EltTip] * (1. + ac * coeff)

    return C