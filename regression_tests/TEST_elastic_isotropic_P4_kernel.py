# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Carlo Peruzzo on 01.09.20.
Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory, 2016-2020.
All rights reserved. See the LICENSE.TXT file for more details.
"""

# external imports
import numpy as np

# local imports
from solid.elasticity_kernels.isotropic_R4_elem import sig_zz_Dz_11, sig_zz_Dz_12, sig_zz_Dz_13, sig_zz_Dz_21, sig_zz_Dz_22, sig_zz_Dz_23, sig_zz_Dz_31, sig_zz_Dz_32, sig_zz_Dz_33

###### TESTING ######

# common parameeters
nu = 0.4                            # Poisson's ratio
youngs_mod = 3.3e10                 # Young's modulus
Ep = youngs_mod / (1 - nu ** 2) # plain strain modulus
toll = 1.e-6

def test_sig_zz_Dz_11():
    a = 1.8
    b = 2.6
    x1 = 0.1
    x2 = 0.3
    val = sig_zz_Dz_11(a, b, x1, x2)
    true_val = 0.00630201
    diff = np.abs(val-true_val)
    assert diff < toll


def test_sig_zz_Dz_12():
    a = 1.8
    b = 2.6
    x1 = 0.1
    x2 = 0.3
    val = sig_zz_Dz_12(a, b, x1, x2)
    true_val = -0.315158
    diff = np.abs(val-true_val)
    assert diff < toll


def test_sig_zz_Dz_13():
    a = 1.8
    b = 2.6
    x1 = 0.1
    x2 = 0.3
    val = sig_zz_Dz_13(a, b, x1, x2)
    true_val = -0.010563
    diff = np.abs(val-true_val)
    assert diff < toll


def test_sig_zz_Dz_21():
    a = 1.8
    b = 2.6
    x1 = 0.1
    x2 = 0.3
    val = sig_zz_Dz_21(a, b, x1, x2)
    true_val = -0.256847
    diff = np.abs(val - true_val)
    assert diff < toll


def test_sig_zz_Dz_22():
    a = 1.8
    b = 2.6
    x1 = 0.1
    x2 = 0.3
    val = sig_zz_Dz_22(a, b, x1, x2)
    true_val = 3.517325122370536
    diff = np.abs(val - true_val)
    assert diff < toll

    a = 0.3
    b = 0.4
    x1 = 0.1
    x2 = 0.3
    val = sig_zz_Dz_22(a, b, x1, x2)
    true_val = 25.77633909108604
    diff = np.abs(val - true_val)
    assert diff < toll


def test_sig_zz_Dz_23():
    a = 1.8
    b = 2.6
    x1 = 0.1
    x2 = 0.3
    val = sig_zz_Dz_23(a, b, x1, x2)
    true_val = -0.022717055840082906
    diff = np.abs(val - true_val)
    assert diff < toll

def test_sig_zz_Dz_31():
    a = 1.8
    b = 2.6
    x1 = 0.1
    x2 = 0.3
    val = sig_zz_Dz_31(a, b, x1, x2)
    true_val = -0.00036087223943224653
    diff = np.abs(val - true_val)
    assert diff < toll

def test_sig_zz_Dz_32():
    a = 1.8
    b = 2.6
    x1 = 0.1
    x2 = 0.3
    val = sig_zz_Dz_32(a, b, x1, x2)
    true_val = -0.183284340030993
    diff = np.abs(val - true_val)
    assert diff < toll


def test_sig_zz_Dz_33():
    a = 1.8
    b = 2.6
    x1 = 0.1
    x2 = 0.3
    val = sig_zz_Dz_33(a, b, x1, x2)
    true_val = -0.00947443096063772
    diff = np.abs(val - true_val)
    assert diff < toll