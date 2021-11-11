# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Carlo Peruzzo on 20.10.21.
Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory, 2016-2020.
All rights reserved. See the LICENSE.TXT file for more details.
"""

# External imports
from scipy.sparse.linalg import spilu
from scipy.sparse import csc_matrix

# Internal imports
from linear_solvers.preconditioners.preconditioner import Preconditioner

class EHL_iLU_Prec(Preconditioner):
  def __init__(self, A, drop_tol=1e-10, fill_factor=10):
    EHL_iLU = spilu(csc_matrix(A), drop_tol=drop_tol, fill_factor=fill_factor)

    # ---> to check the sparsity pattern
    # import matplotlib
    # matplotlib.pyplot.spy(EHL_iLU.L)

    # ---> to check eig
    # (A1, b1, interItr1, indices1) = sys_fun._getsys(xks[0, ::], interItr, *args)
    # ss, sss = np.linalg.eig(A1)

    super().__init__(EHL_iLU)
