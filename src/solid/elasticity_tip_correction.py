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

@njit(parallel=True, fastmath=True, nogil=True, cache=True)
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