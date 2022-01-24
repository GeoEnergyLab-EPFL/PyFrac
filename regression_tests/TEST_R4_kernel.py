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
from solid.elasticity_kernels.isotropic_R4_elem import sig_zz_Dz_11, sig_zz_Dz_12, sig_zz_Dz_13, isotropic_R4_kernel
from solid.elasticity_kernels.isotropic_R4_elem import sig_zz_Dz_21, sig_zz_Dz_22, sig_zz_Dz_23
from solid.elasticity_kernels.isotropic_R4_elem import sig_zz_Dz_31, sig_zz_Dz_32, sig_zz_Dz_33

toll_abs_err = 5*10.**-13
toll_rel_err = 5*10.**-13

def common_test_for_all_sig_zz_Dz_ij(func_, a, b, x1, x2, ref_val):

    to_check_val = func_(a, b, x1, x2)

    print("\n  Testing: " + str(func_))
    if ref_val > 1.:
        print("  ... checking rel error")
        # compute relative error
        err = np.abs(ref_val - to_check_val)/np.abs(ref_val)
        assert err < toll_rel_err
        return err
    else:
        # comput abs error
        print("  ... checking abs error")
        err = np.abs(ref_val - to_check_val)
        assert err < toll_abs_err
        return err



def test01_sig_zz_Dz_ij_():

    a = 2.; b = 1.
    x1 = 1.20; x2 = 0.95

    ref_val = { 'sigzzdz_11':  0.6087493633105447,
                'sigzzdz_12': -3.3188681869484165,
                'sigzzdz_13': -1.8686128498352588,
                'sigzzdz_21': -5.219194144980523,
                'sigzzdz_22': 27.508078977224827,
                'sigzzdz_23': 15.990313481397791,
                'sigzzdz_31': -1.114290789271492,
                'sigzzdz_32':  5.982274003903131,
                'sigzzdz_33':  3.4246549240151216}

    err_set = []
    err_set.append(common_test_for_all_sig_zz_Dz_ij(sig_zz_Dz_11, a, b, x1, x2, ref_val['sigzzdz_11']))
    err_set.append(common_test_for_all_sig_zz_Dz_ij(sig_zz_Dz_12, a, b, x1, x2, ref_val['sigzzdz_12']))
    err_set.append(common_test_for_all_sig_zz_Dz_ij(sig_zz_Dz_13, a, b, x1, x2, ref_val['sigzzdz_13']))
    err_set.append(common_test_for_all_sig_zz_Dz_ij(sig_zz_Dz_21, a, b, x1, x2, ref_val['sigzzdz_21']))
    err_set.append(common_test_for_all_sig_zz_Dz_ij(sig_zz_Dz_22, a, b, x1, x2, ref_val['sigzzdz_22']))
    err_set.append(common_test_for_all_sig_zz_Dz_ij(sig_zz_Dz_23, a, b, x1, x2, ref_val['sigzzdz_23']))
    err_set.append(common_test_for_all_sig_zz_Dz_ij(sig_zz_Dz_31, a, b, x1, x2, ref_val['sigzzdz_31']))
    err_set.append(common_test_for_all_sig_zz_Dz_ij(sig_zz_Dz_32, a, b, x1, x2, ref_val['sigzzdz_32']))
    err_set.append(common_test_for_all_sig_zz_Dz_ij(sig_zz_Dz_33, a, b, x1, x2, ref_val['sigzzdz_33']))
    err_set = np.asarray(err_set)
    print(f" max err is: {err_set.max()}")


def test02_sig_zz_Dz_ij_():

    a = 1.; b = 1.5
    x1 = 0.95; x2 = 1.20

    # 14 digits accurate
    ref_val = { 'sigzzdz_11':  0.7249736004591366,
                'sigzzdz_12': -4.969461391461541,
                'sigzzdz_13': -1.6861383447281155,
                'sigzzdz_21': -4.059782906575605,
                'sigzzdz_22': 26.809247876344845,
                'sigzzdz_23': 9.415244161241898,
                'sigzzdz_31': -2.206408995916106,
                'sigzzdz_32':  15.130836585058026,
                'sigzzdz_33':  5.125925237517022}

    err_set = []
    err_set.append(common_test_for_all_sig_zz_Dz_ij(sig_zz_Dz_11, a, b, x1, x2, ref_val['sigzzdz_11']))
    err_set.append(common_test_for_all_sig_zz_Dz_ij(sig_zz_Dz_12, a, b, x1, x2, ref_val['sigzzdz_12']))
    err_set.append(common_test_for_all_sig_zz_Dz_ij(sig_zz_Dz_13, a, b, x1, x2, ref_val['sigzzdz_13']))
    err_set.append(common_test_for_all_sig_zz_Dz_ij(sig_zz_Dz_21, a, b, x1, x2, ref_val['sigzzdz_21']))
    err_set.append(common_test_for_all_sig_zz_Dz_ij(sig_zz_Dz_22, a, b, x1, x2, ref_val['sigzzdz_22']))
    err_set.append(common_test_for_all_sig_zz_Dz_ij(sig_zz_Dz_23, a, b, x1, x2, ref_val['sigzzdz_23']))
    err_set.append(common_test_for_all_sig_zz_Dz_ij(sig_zz_Dz_31, a, b, x1, x2, ref_val['sigzzdz_31']))
    err_set.append(common_test_for_all_sig_zz_Dz_ij(sig_zz_Dz_32, a, b, x1, x2, ref_val['sigzzdz_32']))
    err_set.append(common_test_for_all_sig_zz_Dz_ij(sig_zz_Dz_33, a, b, x1, x2, ref_val['sigzzdz_33']))
    err_set = np.asarray(err_set)
    print(f" max err is: {err_set.max()}") # must be < 10.^-14


"""
    -------------
    those functions can be used to show that the kernel for the R4 element is not dependent only on the distance 
    between source (A) and receiver (B) but it depends also on the relative position between the two points, 
    i.e. the parameter:
    
    sign(xA-xB) * sign(yA-yB)
    -------------
    
def base_test_toeplitz_R4(dx, dy, hx, hy):

    a = hx / 2.
    b = hy / 2.
    const = 1.
    pp = isotropic_R4_kernel(+dx, +dy, hx, hy, a, b, const)
    mm = isotropic_R4_kernel(-dx, -dy, hx, hy, a, b, const)
    mp = isotropic_R4_kernel(-dx, +dy, hx, hy, a, b, const)
    pm = isotropic_R4_kernel(+dx, -dy, hx, hy, a, b, const)
    print(f'data: ')
    print(f' ++ res: {pp}')
    print(f' -- res: {mm}')
    print(f' -+ res: {mp}')
    print(f' +- res: {pm}')
    print(f'done')
    assert 1 < 2

def test01_toeplitz_R4():
    dx = 2.6
    dy = 2.6
    hx = 2.
    hy = 2.
    base_test_toeplitz_R4(dx, dy, hx, hy)

    dx = 2.6
    dy = 2.6
    hx = 6.
    hy = 6.
    base_test_toeplitz_R4(dx, dy, hx, hy)

    dx = 2.1
    dy = 2.1
    hx = 2.
    hy = 3.
    base_test_toeplitz_R4(dx, dy, hx, hy)

    dx = 4.1
    dy = 6.1
    hx = 2.
    hy = 3.
    base_test_toeplitz_R4(dx, dy, hx, hy)

    dx = 0.
    dy = 6.1
    hx = 2.
    hy = 3.
    base_test_toeplitz_R4(dx, dy, hx, hy)

    dx = 6.1
    dy = 0.
    hx = 2.
    hy = 3.
    base_test_toeplitz_R4(dx, dy, hx, hy)
"""
