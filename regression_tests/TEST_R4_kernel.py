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
from solid.elsticity_kernels.isotropic_R4_elem import sig_zz_Dz_11, sig_zz_Dz_12, sig_zz_Dz_13
from solid.elsticity_kernels.isotropic_R4_elem import sig_zz_Dz_21, sig_zz_Dz_22, sig_zz_Dz_23
from solid.elsticity_kernels.isotropic_R4_elem import sig_zz_Dz_31, sig_zz_Dz_32, sig_zz_Dz_33

toll_abs_err = 5*10.**-6
toll_rel_err = 5*10.**-6

def common_test_for_all_sig_zz_Dz_ij(func_, a, b, x1, x2, ref_val):

    to_check_val = func_(a, b, x1, x2)

    print("\n  Testing: " + str(func_))
    if ref_val > 1.:
        print("  ... checking rel error")
        # compute relative error
        err = np.abs(ref_val - to_check_val)/np.abs(ref_val)
        assert err < toll_rel_err
    else:
        # comput abs error
        print("  ... checking abs error")
        err = np.abs(ref_val - to_check_val)
        assert err < toll_abs_err



def test01_sig_zz_Dz_ij_():

    a = 2.; b = 1.
    x1 = 1.20; x2 = 0.95

    ref_val = { 'sigzzdz_11':  0.608749,
                'sigzzdz_12': -3.31887,
                'sigzzdz_13': -1.86861,
                'sigzzdz_21': -5.21919,
                'sigzzdz_22': 27.5081,
                'sigzzdz_23': 15.9903,
                'sigzzdz_31': -1.11429,
                'sigzzdz_32':  5.98227,
                'sigzzdz_33':  3.42465}

    common_test_for_all_sig_zz_Dz_ij(sig_zz_Dz_11, a, b, x1, x2, ref_val['sigzzdz_11'])
    common_test_for_all_sig_zz_Dz_ij(sig_zz_Dz_12, a, b, x1, x2, ref_val['sigzzdz_12'])
    common_test_for_all_sig_zz_Dz_ij(sig_zz_Dz_13, a, b, x1, x2, ref_val['sigzzdz_13'])
    common_test_for_all_sig_zz_Dz_ij(sig_zz_Dz_21, a, b, x1, x2, ref_val['sigzzdz_21'])
    common_test_for_all_sig_zz_Dz_ij(sig_zz_Dz_22, a, b, x1, x2, ref_val['sigzzdz_22'])
    common_test_for_all_sig_zz_Dz_ij(sig_zz_Dz_23, a, b, x1, x2, ref_val['sigzzdz_23'])
    common_test_for_all_sig_zz_Dz_ij(sig_zz_Dz_31, a, b, x1, x2, ref_val['sigzzdz_31'])
    common_test_for_all_sig_zz_Dz_ij(sig_zz_Dz_32, a, b, x1, x2, ref_val['sigzzdz_32'])
    common_test_for_all_sig_zz_Dz_ij(sig_zz_Dz_33, a, b, x1, x2, ref_val['sigzzdz_33'])



def test02_sig_zz_Dz_ij_():

    a = 1.; b = 1.5
    x1 = 0.95; x2 = 1.20

    ref_val = { 'sigzzdz_11':  0.724974,
                'sigzzdz_12': -4.96946,
                'sigzzdz_13': -1.68614,
                'sigzzdz_21': -4.05978,
                'sigzzdz_22': 26.8092,
                'sigzzdz_23': 9.41524,
                'sigzzdz_31': -2.20641,
                'sigzzdz_32':  15.1308,
                'sigzzdz_33':  5.12593}

    common_test_for_all_sig_zz_Dz_ij(sig_zz_Dz_11, a, b, x1, x2, ref_val['sigzzdz_11'])
    common_test_for_all_sig_zz_Dz_ij(sig_zz_Dz_12, a, b, x1, x2, ref_val['sigzzdz_12'])
    common_test_for_all_sig_zz_Dz_ij(sig_zz_Dz_13, a, b, x1, x2, ref_val['sigzzdz_13'])
    common_test_for_all_sig_zz_Dz_ij(sig_zz_Dz_21, a, b, x1, x2, ref_val['sigzzdz_21'])
    common_test_for_all_sig_zz_Dz_ij(sig_zz_Dz_22, a, b, x1, x2, ref_val['sigzzdz_22'])
    common_test_for_all_sig_zz_Dz_ij(sig_zz_Dz_23, a, b, x1, x2, ref_val['sigzzdz_23'])
    common_test_for_all_sig_zz_Dz_ij(sig_zz_Dz_31, a, b, x1, x2, ref_val['sigzzdz_31'])
    common_test_for_all_sig_zz_Dz_ij(sig_zz_Dz_32, a, b, x1, x2, ref_val['sigzzdz_32'])
    common_test_for_all_sig_zz_Dz_ij(sig_zz_Dz_33, a, b, x1, x2, ref_val['sigzzdz_33'])
