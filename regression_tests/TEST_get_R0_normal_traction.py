# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Carlo Peruzzo on 01.09.20.
Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory, 2016-2020.
All rights reserved. See the LICENSE.TXT file for more details.
"""

# external imports
import numpy as np

# internal imports
from mesh_obj.mesh import CartesianMesh
from solid.elasticity_isotropic import load_isotropic_elasticity_matrix_toepliz

def test_get_traction_R0_kernel():
    nu = 0.4                            # Poisson's ratio
    youngs_mod = 3.3e10                 # Young's modulus
    Ep = youngs_mod / (1 - nu ** 2) # plain strain modulus

    Mesh = CartesianMesh(0.6, 0.6, 19, 19)
    C_obj = load_isotropic_elasticity_matrix_toepliz(Mesh, Ep, C_precision = np.float64, Kernel='R0')
    xy_obs = Mesh.CenterCoor
    xy_crack =  Mesh.CenterCoor
    w_crack = np.ones(Mesh.NumberOfElts)
    res1 = C_obj.get_normal_traction_at(xy_obs, xy_crack, w_crack)
    res2 = C_obj._matvec(w_crack)
    difference = np.max(np.abs(np.abs(res1)-np.abs(res2)))
    assert difference < 1.e-3