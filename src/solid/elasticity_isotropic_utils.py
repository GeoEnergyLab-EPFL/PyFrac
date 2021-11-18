# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Carlo Peruzzo on Tue Nov 2 15:09:38 2021.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2020.
All rights reserved. See the LICENSE.TXT file for more details.
"""

# External imports
import numpy as np

def get_isotropic_el_self_eff(hx, hy, Ep):
    """
    Evaluate the self effect term (diagonal value) for isotropic elasticity.
    Arguments:
        hx (float):                     -- x size of a mesh cell
        hy (float):                     -- y size of a mesh cell
        Ep (float):                     -- plain strain modulus.
    Returns:
        ndarray-float:                  -- the diagonal term.
    """

    a = hx / 2.  # Lx/nx-1
    b = hy / 2.  # Ly/ny-1
    bb = b * b
    aa = a * a
    sqrt_aa_p_bb = np.sqrt(aa + bb) / (a * b)
    return sqrt_aa_p_bb * Ep / (2. * np.pi)