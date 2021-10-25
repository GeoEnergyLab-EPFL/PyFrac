# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Carlo Peruzzo on 20.10.21.
Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory, 2016-2020.
All rights reserved. See the LICENSE.TXT file for more details.
"""

# external
from scipy.sparse.linalg import LinearOperator

class APrec(LinearOperator):
  def __init__(self, EHL_iLU):
    self.dtype_ = float
    self.shape_ = EHL_iLU.shape
    self.EHL_iLU = EHL_iLU
    super().__init__(self.dtype_, self.shape_)

  def _matvec(self, v):
    """
    This function implements the dot product.
    :param v: vector expected to be of size unknowns_number_
    :return: HMAT.v, where HMAT is a matrix obtained by selecting equations from either HMATtract or HMATdispl
    """
    return self.EHL_iLU.solve(v)

  @property
  def _init_shape(self):
    return self.shape_

  def _init_dtype(self):
    return self.dtype_