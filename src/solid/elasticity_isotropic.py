# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on Tue Dec 27 17:41:56 2016.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2020.
All rights reserved. See the LICENSE.TXT file for more details.
"""

# external imports
import numpy as np
from numba.typed import List

# internal imports
from solid.elasticity_toeplitz import elasticity_matrix_toepliz
from solid.elsticity_kernels.isotropic_R0_elem import get_toeplitzCoe_isotropic
from solid.elsticity_kernels.isotropic_R4_elem import get_toeplitzCoe_isotropic_R4

# -----------------------------------------------------------------------------------------------------------------------

class load_isotropic_elasticity_matrix_toepliz(elasticity_matrix_toepliz):

    def __init__(self, Mesh, Ep, C_precision=np.float64, useHMATdot=False, nu=None, HMATparam = None, Kernel='R4'):

        self.nu = nu
        self.Ep = Ep
        self.Kernel = Kernel

        if useHMATdot:
            if nu is None: SystemExit("please, provide the Poisson's ratio to get the full blocks")
            self.HMATcreationTime = []
            elas_prop_HMAT = [self.Ep * (1 - self.nu ** 2), self.nu] #  youngs_mod, nu
        else:
            elas_prop_HMAT = []

        matprop = [Ep]
        super().__init__(Mesh, matprop, elas_prop_HMAT, C_precision, useHMATdot, kerneltype='Isotropic', HMATparam = HMATparam)

    def reload_toepliz_Coe(self, Lx, Ly, nx, ny, hx, hy, mat_prop):
        typedList_mat_prop = List()
        [typedList_mat_prop.append(x) for x in mat_prop]
        if self.Kernel == 'R4':
            return get_toeplitzCoe_isotropic_R4(nx, ny, hx, hy, typedList_mat_prop, self.C_precision)
        elif self.Kernel == 'R0':
            return get_toeplitzCoe_isotropic(nx, ny, hx, hy, typedList_mat_prop, self.C_precision)
        else:
            SystemExit("Elastic kernel non supported: \n try: 'Kernel=R4' or 'Kernel=R0'")
