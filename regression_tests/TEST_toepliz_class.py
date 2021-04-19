# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Carlo Peruzzo on 01.09.20.
Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory, 2016-2020.
All rights reserved. See the LICENSE.TXT file for more details.
"""

import pytest

# local imports
from mesh import CartesianMesh
import numpy as np
from elasticity import load_isotropic_elasticity_matrix
from elasticity import load_isotropic_elasticity_matrix_toepliz

def common_test_for_all_toepliz_tests(C, C_new, expect_simmetric=False):

    size0 = C.shape[0]
    size1 = C.shape[1]
    # check that the dimensions of the two matrices is the same
    assert C.shape[0] == C_new.shape[0] and C.shape[1] == C_new.shape[1]

    sum = 0.
    for i in range(size0):
        for j in range(size1):
            sum = sum + abs(C_new[i, j] - C[i, j])
            if size0 == size1 and expect_simmetric:
                #check symmetry in the case of a square matrix
                assert C_new[i, j] == C_new[j, i]
    # check that the sum of the differences between any entry of both matrices is zero
    assert sum == 0.


###### TESTING ######

# common parameeters
nu = 0.4                            # Poisson's ratio
youngs_mod = 3.3e10                 # Young's modulus
Ep = youngs_mod / (1 - nu ** 2) # plain strain modulus

def test_toepliz_get_submatrix_hy_eq_hx_and_nx_eq_ny():
    # Mesh hx=hy, nx=ny
    # creating mesh
    print("Mesh hx=hy, nx=ny ...")
    Mesh = CartesianMesh(0.6, 0.6, 19, 19)
    # old way
    C = load_isotropic_elasticity_matrix(Mesh, Ep)
    # new way
    C_obj = load_isotropic_elasticity_matrix_toepliz(Mesh,Ep)
    slice = np.asarray(range(Mesh.NumberOfElts))
    C_new = C_obj[np.ix_(slice, slice)]
    common_test_for_all_toepliz_tests(C, C_new, expect_simmetric=True)

def test_toepliz_get_submatrix_hy_noteq_hx_and_nx_noteq_ny():
    # Mesh hx!=hy, nx!=ny
    Mesh = CartesianMesh(0.5, 0.6, 23, 27)
    # old way
    C = load_isotropic_elasticity_matrix(Mesh, Ep)
    # new way
    C_obj = load_isotropic_elasticity_matrix_toepliz(Mesh,Ep)
    slice = np.asarray(range(Mesh.NumberOfElts))
    C_new = C_obj[np.ix_(slice, slice)]
    common_test_for_all_toepliz_tests(C, C_new, expect_simmetric=True)

def test_toepliz_get_submatrix_hy_eq_hx_and_nx_noteq_ny():
    # Mesh hx=hy, nx!=ny
    Mesh = CartesianMesh(31*1/63, 1, 11, 23)
    # old way
    C = load_isotropic_elasticity_matrix(Mesh, Ep)
    # new way
    C_obj = load_isotropic_elasticity_matrix_toepliz(Mesh,Ep)
    slice = np.asarray(range(Mesh.NumberOfElts))
    C_new = C_obj[np.ix_(slice, slice)]
    common_test_for_all_toepliz_tests(C, C_new, expect_simmetric=True)

def test_toepliz_get_submatrix_hy_noteq_hx_and_nx_eq_ny():
    # Mesh hx!=hy, nx=ny
    Mesh = CartesianMesh(1.4, 1.6, 19, 19)
    # old way
    C = load_isotropic_elasticity_matrix(Mesh, Ep)
    # new way
    C_obj = load_isotropic_elasticity_matrix_toepliz(Mesh,Ep)
    slice = np.asarray(range(Mesh.NumberOfElts))
    C_new = C_obj[np.ix_(slice, slice)]
    common_test_for_all_toepliz_tests(C, C_new, expect_simmetric=True)

def test_toepliz_get_submatrix_same_dim():
    Mesh = CartesianMesh(0.45, 0.6, 39, 49)
    # old way
    C = load_isotropic_elasticity_matrix(Mesh, Ep)
    # new way
    C_obj = load_isotropic_elasticity_matrix_toepliz(Mesh,Ep)
    xslice = np.asarray([33, 55, 66])
    yslice = np.asarray([27, 12, 41])
    C_new_sliced = C_obj[np.ix_(xslice,yslice)]
    C_sliced = C[np.ix_(xslice, yslice)]
    common_test_for_all_toepliz_tests(C_sliced, C_new_sliced, expect_simmetric=False)

def test_toepliz_get_submatrix_different_dim():
    Mesh = CartesianMesh(0.45, 0.6, 29, 29)
    # old way
    C = load_isotropic_elasticity_matrix(Mesh, Ep)
    # new way
    C_obj = load_isotropic_elasticity_matrix_toepliz(Mesh,Ep)
    xslice = np.asarray([33,55,66])
    yslice = np.asarray([2,18,22,45])
    C_new_sliced = C_obj[np.ix_(xslice, yslice)]
    C_sliced = C[np.ix_(xslice, yslice)]
    common_test_for_all_toepliz_tests(C_sliced, C_new_sliced, expect_simmetric=False)


