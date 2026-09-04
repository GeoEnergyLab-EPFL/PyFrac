# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Carlo Peruzzo on 20.10.21.
Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory, 2016-2020.
All rights reserved. See the LICENSE.TXT file for more details.
"""

# External imports
from scipy.sparse.linalg import LinearOperator

class Preconditioner(LinearOperator):
  def __init__(self, prec_csc_matrix):
    self.dtype_ = float
    self.shape_ = prec_csc_matrix.shape
    self.prec_csc_matrix = prec_csc_matrix
    super().__init__(self.dtype_, self.shape_)

  def _matvec(self, v):
    """
    This function implements the dot product.
    :param v: vector expected to be of size unknowns_number_
    :return: prec_csc_matrix.v, where prec_csc_matrix is a precoditioner matrix
    """
    return self.prec_csc_matrix.solve(v)

  @property
  def _init_shape(self):
    return self.shape_

  def _init_dtype(self):
    return self.dtype_