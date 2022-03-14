# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Carlo Peruzzo on Tue Dec 15 10:18:56 2021.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2022.
All rights reserved. See the LICENSE.TXT file for more details.
"""
# external imports
import time

from common_rect_and_radial_tests import *
from utilities.utility import plot_as_matrix
from solid.elasticity_isotropic import load_isotropic_elasticity_matrix_toepliz

# ----------------------------------------------
# ----------------------------------------------
# RUN
# ----------------------------------------------
# ----------------------------------------------

sim_info = {"nu": 0.25 , "youngs mod": 1e9} # Poissons'ratio and young's modulus
sim_info["Eprime"] = sim_info["youngs mod"] / (1 - sim_info["nu"] ** 2)  # plain strain modulus

# set the domain size
sim_info["domain x"] = [-100, 100]

# uniform opening
sim_info["wset"] = 1

## --- creating the mesh --- ##
Mesh = CartesianMesh([-18.79048192, 18.79048192], [-13.4217728, 848.25604096], 85, 3855)
EltTip = np.asarray(Mesh.get_Boundarylist())
EltRibbon = Mesh.get_Frontlist()
EltCrack = np.arange(Mesh.NumberOfElts)
FillF = np.ones(len(EltTip))

# setting the constant opening DD
w = np.full(len(EltCrack), sim_info["wset"])


## --- prepare the HMat parameters --- ##
HMATparam = [500, 10, 1e-4]

## --- load HMat --- ##
print(f" Loading the HMat")
tHmatLoad = - time.time()
C = load_isotropic_elasticity_matrix_toepliz(Mesh, sim_info["Eprime"], C_precision=np.float64,
                                             useHMATdot=True, nu=sim_info["nu"],
                                             HMATparam=HMATparam)
tHmatLoad = tHmatLoad + time.time()
print(f"     --> HMat loaded in {tHmatLoad} [s]")

## --- Do the multiplication --- ##
print(f" Solving for pressure")
tHmatDot = - time.time()
sol_p = C.HMAT._matvec(w)
tHmatDot = tHmatDot + time.time()
print(f"     --> HMat dot product solved in {tHmatDot} [s]")
print(f"     --> The compression ratio is {C.HMAT.HMATtract.getCompressionRatio()*100} [%]")

p_ana = C.get_normal_traction_at(Mesh.CenterCoor, Mesh.CenterCoor, w)


# some plots
rel_err_num = 100 * np.abs(sol_p - p_ana) / p_ana
#plot_as_matrix(rel_err_num, mesh=Mesh) # 2D plot
#plot_3d_scatter(sol_p, Mesh.CenterCoor[:, 0], Mesh.CenterCoor[:, 1]) # 3D plot
#plot_3d_scatter(p_ana, Mesh.CenterCoor[:, 0], Mesh.CenterCoor[:, 1])  # 3D plot

print(f"     --> The max rel error is {rel_err_num.max()} [%]")
print(f" Done ")