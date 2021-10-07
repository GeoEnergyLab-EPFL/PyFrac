# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Andreas Möri on 03.10.21.
Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory, 2016-2021.
All rights reserved. See the LICENSE.TXT file for more details.
"""

import pytest

# local imports
from mesh import CartesianMesh
import numpy as np
from elasticity import load_isotropic_elasticity_matrix
from elasticity import load_isotropic_elasticity_matrix_toepliz

def evaluate_eikonal_equation_solution(num_solution, analy_solution, mesh):


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

def get_expected_error(mesh):
    # Here we need to get the expected error from the scheme.
    assert 0 == 0

###### TESTING ######

# common parameters
elNx = [41, 61, 61, 101, 101]            # evaluate for four different number of elements.
elNy = [41, 41, 61, 81, 101]
youngs_mod = 3.3e10                 # Young's modulus
Ep = youngs_mod / (1 - nu ** 2) # plain strain modulus

def test_propagation_from_the_origin():
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

def test_radial():
    # Mesh hx!=hy, nx!=ny
    Mesh = CartesianMesh(0.5, 0.6, 23, 27)
    # old way
    C = load_isotropic_elasticity_matrix(Mesh, Ep)
    # new way
    C_obj = load_isotropic_elasticity_matrix_toepliz(Mesh,Ep)
    slice = np.asarray(range(Mesh.NumberOfElts))
    C_new = C_obj[np.ix_(slice, slice)]
    common_test_for_all_toepliz_tests(C, C_new, expect_simmetric=True)

def test_elliptical():
    # Mesh hx=hy, nx!=ny
    Mesh = CartesianMesh(31*1/63, 1, 11, 23)
    # old way
    C = load_isotropic_elasticity_matrix(Mesh, Ep)
    # new way
    C_obj = load_isotropic_elasticity_matrix_toepliz(Mesh,Ep)
    slice = np.asarray(range(Mesh.NumberOfElts))
    C_new = C_obj[np.ix_(slice, slice)]
    common_test_for_all_toepliz_tests(C, C_new, expect_simmetric=True)

def test_square():
    # Mesh hx!=hy, nx=ny
    Mesh = CartesianMesh(1.4, 1.6, 19, 19)
    # old way
    C = load_isotropic_elasticity_matrix(Mesh, Ep)
    # new way
    C_obj = load_isotropic_elasticity_matrix_toepliz(Mesh,Ep)
    slice = np.asarray(range(Mesh.NumberOfElts))
    C_new = C_obj[np.ix_(slice, slice)]
    common_test_for_all_toepliz_tests(C, C_new, expect_simmetric=True)
