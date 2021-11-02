# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Andreas Möri on 03.10.21.
Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory, 2016-2021.
All rights reserved. See the LICENSE.TXT file for more details.
"""

# local imports
from mesh.mesh import CartesianMesh
import numpy as np
from fracture.fracture_initialization import get_radial_survey_cells, get_rectangular_survey_cells
from level_set.level_set import SolveFMM
from level_set.FMM import fmm

def evaluate_eikonal_equation_solution(num_solution, analy_solution, mesh, test, tol):

    # We get the limit for the corresponding evaluation.
    limit = tol * get_expected_error(mesh, test[1])

    # The elements at the boundary can pose problem (different evaluation between both approaches) so we only take inner
    # elements.
    inner_els = get_inner_elements(mesh)

    # We calculate the relative error
    Error = abs(1 - num_solution / analy_solution)
    Error[analy_solution == 0] = abs(num_solution - analy_solution)[analy_solution == 0]

    # Check if we are within the tolerance
    assert max(Error[inner_els]) < limit, "The test " + test[0] + \
                                          " has an error that is above" + str((tol-1)*100) + \
                                          "% of the expected error!"

def get_expected_error(mesh, test_type):
    # Here we need to get the expected error from the scheme.
    if test_type == "o":
        return abs(1 - (2 + np.sqrt(2))/(2 * np.sqrt(2)))
    else:
        return 1e-4

def get_inner_elements(mesh):
    els = np.setdiff1d(np.arange(mesh.NumberOfElts), np.arange(mesh.nx)) # eliminate bottom boundary
    els = np.setdiff1d(els, np.arange(mesh.ny)*mesh.nx)                 # eliminate left boundary
    els = np.setdiff1d(els, np.arange(mesh.ny)[1:]*mesh.nx - 1)         # eliminate right boundary
    return np.setdiff1d(els, np.arange(mesh.nx) + mesh.nx*(mesh.ny - 1)) # eliminate top boundary

###### TESTING ######

# common parameters
elNx = [41, 61, 61, 101, 101]            # evaluate for four different number of elements.
elNy = [41, 41, 61, 81, 101]
tolerance = 1.01

def test_propagation_from_the_origin():
    # Here we test if we solve a propagation from the origin exactly according to the error we can expect from the
    # scheme.
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

def test_radial():
    # We cannot really asses the expected error in this case. So we simply check if the new solution is equivalent
    # to the old solution.
    iter = 0
    for n in elNx:
        # creating mesh
        Mesh = CartesianMesh(10, 10, n, elNy[iter])

        radius = 7.5 * Mesh.hx

        surv_cells, surv_dist, inner_cells = get_radial_survey_cells(Mesh, radius)

        ## --- "Old" method for a level set radially outward from the center point

        sgndDist = np.full((Mesh.NumberOfElts,), 1e50)
        sgndDist[surv_cells] = -surv_dist
        EltRest = np.setdiff1d(np.arange(Mesh.NumberOfElts), inner_cells)

        # fast marching to get level set
        SolveFMM(sgndDist,
                 surv_cells,
                 inner_cells,
                 Mesh,
                 EltRest,
                 inner_cells)

        ## --- "New" method

        fmmStruct = fmm(Mesh)

        fmmStruct.solveFMM((-surv_dist, surv_cells),
                           np.hstack((np.setdiff1d(np.arange(Mesh.NumberOfElts), inner_cells), surv_cells)), Mesh)

        fmmStruct.solveFMM((surv_dist, surv_cells), inner_cells, Mesh)

        fmmStruct.LS[inner_cells] = -fmmStruct.LS[inner_cells]

        teststr = "Radial comp, Nx = " + str(n) + " , Ny = " + str(elNy[iter])
        evaluate_eikonal_equation_solution(fmmStruct.LS, sgndDist, Mesh, [teststr, "r"], tolerance)

        iter += 1


def test_rectangular():
    # We cannot really assess the expected error in this case. So we simply check if the new solution is equivalent
    # to the old solution.
    iter = 0
    for n in elNx:
        # creating mesh
        Mesh = CartesianMesh(10, 10, n, elNy[iter])

        length = 5.5 * Mesh.hx
        height = 3.5 * Mesh.hy

        surv_cells, surv_dist, inner_cells = get_rectangular_survey_cells(Mesh, length, height)

        ## --- "Old" method for a level set radially outward from the center point

        sgndDist = np.full((Mesh.NumberOfElts,), 1e50)
        sgndDist[surv_cells] = -surv_dist
        EltRest = np.setdiff1d(np.arange(Mesh.NumberOfElts), inner_cells)

        # fast marching to get level set
        SolveFMM(sgndDist,
                 surv_cells,
                 inner_cells,
                 Mesh,
                 EltRest,
                 inner_cells)

        ## --- "New" method

        fmmStruct = fmm(Mesh)

        fmmStruct.solveFMM((-surv_dist, surv_cells),
                           np.hstack((np.setdiff1d(np.arange(Mesh.NumberOfElts), inner_cells), surv_cells)), Mesh)

        fmmStruct.solveFMM((surv_dist, surv_cells), inner_cells, Mesh)

        fmmStruct.LS[inner_cells] = -fmmStruct.LS[inner_cells]

        teststr = "Rectangular comp, Nx = " + str(n) + " , Ny = " + str(elNy[iter])
        evaluate_eikonal_equation_solution(fmmStruct.LS, sgndDist, Mesh, [teststr, "s"], tolerance)

        iter += 1
