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
from fracture_initialization import get_radial_survey_cells
from level_set import SolveFMM
from FMM import fmm

def evaluate_eikonal_equation_solution(num_solution, analy_solution, mesh, test, tol):

    limit = tol * get_expected_error(mesh, test[1]) # what we would accept as error

    Error = abs(1 - num_solution / analy_solution)
    Error[analy_solution == 0] = abs(num_solution - analy_solution)[analy_solution == 0]
    assert max(Error) < limit, "The test " + test[0] + " has an error that is above 1% of the expected error!"

def get_expected_error(mesh, test_type):
    # Here we need to get the expected error from the scheme.
    if test_type == "o":
        return abs(1 - (2 + np.sqrt(2))/(2 * np.sqrt(2)))

###### TESTING ######

# common parameters
elNx = [41, 61, 61, 101, 101]            # evaluate for four different number of elements.
elNy = [41, 41, 61, 81, 101]
radius = 2
tolerance = 1.01

def test_propagation_from_the_origin():
    iter = 0
    for n in elNx:
        # creating mesh
        Mesh = CartesianMesh(10, 10, n, elNy[iter])

        analyticalSol = (Mesh.CenterCoor[::, 0] ** 2 + Mesh.CenterCoor[::, 1] ** 2) ** (1 / 2)

        ## --- "Old" method for a level set radially outward from the center point

        sgndDist = np.full((Mesh.NumberOfElts,), 1e50)
        sgndDist[Mesh.CenterElts] = 0.0
        EltRest = np.setdiff1d(np.arange(Mesh.NumberOfElts), Mesh.CenterElts)

        # fast marching to get level set
        SolveFMM(sgndDist,
                 Mesh.CenterElts,
                 Mesh.CenterElts,
                 Mesh,
                 EltRest,
                 [])

        teststr = "Origin old, Nx = " + str(n) + " , Ny = " + str(elNy[iter])
        evaluate_eikonal_equation_solution(sgndDist, analyticalSol, Mesh, [teststr, "o"], tolerance)

        ## --- "New" method

        fmmStruct = fmm(Mesh)

        fmmStruct.solveFMM(([0.0], Mesh.CenterElts), np.arange(Mesh.NumberOfElts), Mesh)

        teststr = "Origin new, Nx = " + str(n) + " , Ny = " + str(elNy[iter])
        evaluate_eikonal_equation_solution(fmmStruct.LS, analyticalSol, Mesh, [teststr, "o"], tolerance)

        iter += 1

# def test_radial():
#     iter = 0
#     for n in elNx:
#         # creating mesh
#         Mesh = CartesianMesh(10, 10, n, elNy[iter])
#
#         surv_cells, surv_dist, inner_cells = get_radial_survey_cells(Mesh, radius)
#
#         analyticalSol = (Mesh.CenterCoor[::, 0] ** 2 + Mesh.CenterCoor[::, 1] ** 2) ** (1 / 2) - radius
#
#         ## --- "Old" method for a level set radially outward from the center point
#
#         sgndDist = np.full((Mesh.NumberOfElts,), 1e50)
#         sgndDist[surv_cells] = -surv_dist
#         EltRest = np.setdiff1d(np.arange(Mesh.NumberOfElts), inner_cells)
#
#         # fast marching to get level set
#         SolveFMM(sgndDist,
#                  surv_cells,
#                  inner_cells,
#                  Mesh,
#                  EltRest,
#                  inner_cells)
#
#         teststr = "Radial old, Nx = " + str(n) + " , Ny = " + str(elNy[iter])
#         evaluate_eikonal_equation_solution(sgndDist, analyticalSol, Mesh, teststr, tolerance)
#
#         ## --- "New" method
#
#         fmmStruct = fmm(Mesh)
#
#         fmmStruct.solveFMM((-surv_dist, surv_cells),
#                            np.hstack((np.setdiff1d(np.arange(Mesh.NumberOfElts), inner_cells), surv_cells)), Mesh)
#
#         fmmStruct.solveFMM((surv_dist, surv_cells), inner_cells, Mesh)
#
#         phi = fmmStruct.LS
#         phi[inner_cells] = -phi[inner_cells]
#
#         teststr = "Radial new, Nx = " + str(n) + " , Ny = " + str(elNy[iter])
#         evaluate_eikonal_equation_solution(fmmStruct.LS, analyticalSol, Mesh, teststr, tolerance)
#
#         iter += 1

#
# def test_elliptical():
#     # Mesh hx=hy, nx!=ny
#     Mesh = CartesianMesh(31*1/63, 1, 11, 23)
#     # old way
#     C = load_isotropic_elasticity_matrix(Mesh, Ep)
#     # new way
#     C_obj = load_isotropic_elasticity_matrix_toepliz(Mesh,Ep)
#     slice = np.asarray(range(Mesh.NumberOfElts))
#     C_new = C_obj[np.ix_(slice, slice)]
#     common_test_for_all_toepliz_tests(C, C_new, expect_simmetric=True)
#
# def test_square():
#     # Mesh hx!=hy, nx=ny
#     Mesh = CartesianMesh(1.4, 1.6, 19, 19)
#     # old way
#     C = load_isotropic_elasticity_matrix(Mesh, Ep)
#     # new way
#     C_obj = load_isotropic_elasticity_matrix_toepliz(Mesh,Ep)
#     slice = np.asarray(range(Mesh.NumberOfElts))
#     C_new = C_obj[np.ix_(slice, slice)]
#     common_test_for_all_toepliz_tests(C, C_new, expect_simmetric=True)
