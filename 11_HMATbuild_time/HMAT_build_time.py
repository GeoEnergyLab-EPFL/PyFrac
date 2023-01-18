# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Carlo Peruzzo on Mon Dec 6 17:49:21 2021.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2019.
All rights reserved. See the LICENSE.TXT file for more details.
"""

# imports
import os
import numpy as np
import psutil
import time

# local imports
from mesh_obj.mesh import CartesianMesh
from solid.elasticity_isotropic import load_isotropic_elasticity_matrix_toepliz
from utilities.utility import setup_logging_to_console, append_new_line

# setting up the verbosity level of the log at console
setup_logging_to_console(verbosity_level='debug')

def get_statistics(HMATparam, Mesh, filename):

    # solid properties
    nu = 0.25
    youngs_mod = 3.3e10
    G = youngs_mod/(2.0*(1.0 + nu))
    Eprime = youngs_mod / (1 - nu**2)   # plain strain modulus
    K1c = 1e6                           # Fracture toughness


    # gives an object with many fields
    a = psutil.virtual_memory()
    used0 = a.used/1024/1024/1024
    print(used0)
    time_build = - time.time()
    C = load_isotropic_elasticity_matrix_toepliz(Mesh, Eprime, C_precision = np.float64, useHMATdot=True, nu=nu, HMATparam=HMATparam )
    time_build = time_build + time.time()
    a = psutil.virtual_memory()
    used1= a.used/1024/1024/1024
    memuse = used1-used0
    print(str(memuse))
    print(time_build)

    # compute matvec with toeplitz
    x = np.ones(Mesh.NumberOfElts)
    true_res = C._matvec(x)
    approx_res = C.HMAT.matvec(x)
    diff = np.abs((true_res - approx_res))
    CR = C.HMAT.HMATtract.getCompressionRatio()
    max_leaf_size, eta, eps_aca = HMATparam



    append_new_line(file_name,
                    str(Mesh.NumberOfElts) +','  # HMAT size
                    + str(time_build) +','   # HMAT time to build s
                    + str(diff.max()) + ','  # HMAT max diff with exact vector
                    + str(np.abs(true_res).max()) + ','  # max value of the exact vector
                    + str(memuse) + ','      # HMAT mem use Gb
                    + str(CR) + ','          # HMAT Compression Ratio
                    + str(Mesh.nx) + ','
                    + str(Mesh.ny) + ','
                    + str(max_leaf_size) + ','
                    + str(eta) + ','
                    + str(eps_aca) + ','
                    )




# <<< MAIN >>>
MYRANGE_max_leaf_size = [200, 300, 500, 1000]
MYRANGE_eta = [1, 2, 3, 4]
MYRANGE_eps_aca = [1.e-4, 1.e-6]
iter = 0
itertot = 4 * 4 * 2 * 9
for filename_ID in range(10):
    file_name = '/home/peruzzo/Desktop/test_HDOT_scale/HMAT_stat_'+str(filename_ID+1)+'.csv'
    for max_leaf_size in MYRANGE_max_leaf_size:
        for eta in MYRANGE_eta:
            for eps_aca in MYRANGE_eps_aca:
                HMATparam = [max_leaf_size, eta, eps_aca]
                for i in range(9):
                    # creating mesh
                    nx = 101
                    ny = (i+1) * nx
                    if ny % 2 == 0:
                        ny = ny+1
                    Lx = 20
                    hx = 2. * Lx / (nx - 1)
                    Ly = hx * (ny - 1) / 2.
                    Mesh = CartesianMesh(Lx, Ly, nx, ny)

                    # building HMAT
                    get_statistics(HMATparam, Mesh, file_name)

                    print('loop 1: '+str(100*iter/itertot)+' %')
                    iter = iter + 1

