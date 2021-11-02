# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Carlo Peruzzo on 01.09.20.
Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory, 2016-2020.
All rights reserved. See the LICENSE.TXT file for more details.
"""

# local imports
from fracture.fracture_initialization import get_radial_survey_cells
from mesh.mesh import CartesianMesh
import numpy as np
from solid.elasticity_isotropic import load_isotropic_elasticity_matrix
from solid.elasticity_isotropic import load_isotropic_elasticity_matrix_toepliz

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

# def test_toepliz_get_submatrix_hy_eq_hx_and_nx_eq_ny():
#     # Mesh hx=hy, nx=ny
#     # creating mesh
#     print("Mesh hx=hy, nx=ny ...")
#     Mesh = CartesianMesh(0.6, 0.6, 19, 19)
#     # old way
#     C = load_isotropic_elasticity_matrix(Mesh, Ep, C_precision = np.float32)
#     # new way
#     C_obj = load_isotropic_elasticity_matrix_toepliz(Mesh, Ep, C_precision = np.float32)
#     slice = np.asarray(range(Mesh.NumberOfElts))
#     C_new = C_obj[np.ix_(slice, slice)]
#     common_test_for_all_toepliz_tests(C, C_new, expect_simmetric=True)
#
# def test_toepliz_get_submatrix_hy_noteq_hx_and_nx_noteq_ny():
#     # Mesh hx!=hy, nx!=ny
#     Mesh = CartesianMesh(0.5, 0.6, 23, 27)
#     # old way
#     C = load_isotropic_elasticity_matrix(Mesh, Ep, C_precision = np.float32)
#     # new way
#     C_obj = load_isotropic_elasticity_matrix_toepliz(Mesh, Ep, C_precision = np.float32)
#     slice = np.asarray(range(Mesh.NumberOfElts))
#     C_new = C_obj[np.ix_(slice, slice)]
#     common_test_for_all_toepliz_tests(C, C_new, expect_simmetric=True)
#
# def test_toepliz_get_submatrix_hy_eq_hx_and_nx_noteq_ny():
#     # Mesh hx=hy, nx!=ny
#     Mesh = CartesianMesh(31*1/63, 1, 11, 23)
#     # old way
#     C = load_isotropic_elasticity_matrix(Mesh, Ep, C_precision = np.float32)
#     # new way
#     C_obj = load_isotropic_elasticity_matrix_toepliz(Mesh, Ep, C_precision = np.float32)
#     slice = np.asarray(range(Mesh.NumberOfElts))
#     C_new = C_obj[np.ix_(slice, slice)]
#     common_test_for_all_toepliz_tests(C, C_new, expect_simmetric=True)
#
# def test_toepliz_get_submatrix_hy_noteq_hx_and_nx_eq_ny():
#     # Mesh hx!=hy, nx=ny
#     Mesh = CartesianMesh(1.4, 1.6, 19, 19)
#     # old way
#     C = load_isotropic_elasticity_matrix(Mesh, Ep, C_precision = np.float32)
#     # new way
#     C_obj = load_isotropic_elasticity_matrix_toepliz(Mesh, Ep, C_precision = np.float32)
#     slice = np.asarray(range(Mesh.NumberOfElts))
#     C_new = C_obj[np.ix_(slice, slice)]
#     common_test_for_all_toepliz_tests(C, C_new, expect_simmetric=True)
#
# def test_toepliz_get_submatrix_same_dim():
#     Mesh = CartesianMesh(0.45, 0.6, 39, 49)
#     # old way
#     C = load_isotropic_elasticity_matrix(Mesh, Ep, C_precision = np.float32)
#     # new way
#     C_obj = load_isotropic_elasticity_matrix_toepliz(Mesh, Ep, C_precision = np.float32)
#     xslice = np.asarray([33, 55, 66])
#     yslice = np.asarray([27, 12, 41])
#     C_new_sliced = C_obj[np.ix_(xslice,yslice)]
#     C_sliced = C[np.ix_(xslice, yslice)]
#     common_test_for_all_toepliz_tests(C_sliced, C_new_sliced, expect_simmetric=False)
#
# def test_toepliz_get_submatrix_different_dim():
#     Mesh = CartesianMesh(0.45, 0.6, 29, 29)
#     # old way
#     C = load_isotropic_elasticity_matrix(Mesh, Ep, C_precision = np.float32)
#     # new way
#     C_obj = load_isotropic_elasticity_matrix_toepliz(Mesh, Ep, C_precision = np.float32)
#     xslice = np.asarray([33,55,66])
#     yslice = np.asarray([2,18,22,45])
#     C_new_sliced = C_obj[np.ix_(xslice, yslice)]
#     C_sliced = C[np.ix_(xslice, yslice)]
#     common_test_for_all_toepliz_tests(C_sliced, C_new_sliced, expect_simmetric=False)
#
# def test_toepliz_dot_product():
#     #This will test the dot product made in parallel
#     Mesh = CartesianMesh(0.45, 0.6, 29, 29)
#     # old way
#     C = load_isotropic_elasticity_matrix(Mesh, Ep, C_precision=np.float64)
#     # new way
#     C_obj = load_isotropic_elasticity_matrix_toepliz(Mesh, Ep, C_precision=np.float64)
#
#     # array test
#     x = np.ones(C.shape[0],dtype=np.float64)
#     x1 = C.dot(x)
#     x2 = C_obj._matvec_fast(x)
#     diff = np.linalg.norm(x1 - x2)
#     assert(diff < 0.05)
#

def test_toepliz_bandedC():

    #This will test the dot product made in parallel
    Mesh = CartesianMesh(0.45, 0.6, 41, 51)
    #      subcells = [6, 7, 8, 11, 12, 13, 16, 17, 18] example with CartesianMesh(0.45, 0.6, 5, 5)
    r = 0.2
    surv_cells, surv_dist, inner_cells = get_radial_survey_cells(Mesh, r)
    # from utility import plot_as_matrix
    # K = np.zeros((Mesh.NumberOfElts,), )
    # K[inner_cells] = 1
    # plot_as_matrix(K, Mesh)
    subcells = inner_cells

    # old way
    C = load_isotropic_elasticity_matrix(Mesh, Ep, C_precision=np.float64)
    # new way
    C_obj = load_isotropic_elasticity_matrix_toepliz(Mesh, Ep, C_precision=np.float64)

    C9stencilC_full = C[np.ix_(subcells, subcells)]
    C9stencilC = C_obj._get9stencilC(subcells).toarray()

    #check the 0 elements
    res1 = 0.
    [left_elem, right_elem, bottom_elem, top_elem] = [0, 1, 2, 3]
    for i in range(len(subcells)):
        for j in range(len(subcells)):
            cell_i = subcells[i]
            cell_j = subcells[j]
            a = Mesh.NeiElements[cell_i, top_elem]
            b = Mesh.NeiElements[cell_i, right_elem]
            c = Mesh.NeiElements[cell_i, bottom_elem]
            d = Mesh.NeiElements[cell_i, left_elem]
            e = Mesh.NeiElements[d, top_elem]
            f = Mesh.NeiElements[b, top_elem]
            g = Mesh.NeiElements[b, bottom_elem]
            h = Mesh.NeiElements[d, bottom_elem]

            if cell_j not in [a, b, c, d, e, f, g, h, cell_i]:
                res1 = res1 + np.abs(C9stencilC[i, j])
    assert res1 < 10.e-4

    #check the nonzero elements
    res2 = 0.
    for i in range(len(subcells)):
        for j in range(len(subcells)):
            if C9stencilC[i,j] != 0.:
                res2 = res2 + np.abs(C9stencilC_full[i,j]-C9stencilC[i,j])/np.abs(C9stencilC_full[i,j])
    assert res2 < 10.e-2

    #matplotlib.pyplot.spy(C9stencilC)
