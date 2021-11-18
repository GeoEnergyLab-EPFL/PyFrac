# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Carlo Peruzzo on 20.10.21.
Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory, 2016-2020.
All rights reserved. See the LICENSE.TXT file for more details.
"""

# External import
import numpy as np
import time

# Internal import
from linear_solvers.linear_solver import Linear_solver

class Direct_linear_solver(Linear_solver):
  def __init__(self, sys_func):
    super().__init__(sys_func)
    self.interItr = None
    self.indices = None
    self.b = None
    self.call_ID = 0

  def solve(self, solk, interItr, *args):
    # assembling A and b
    A_creation = -time.time()
    A, self.b, self.interItr, self.indices = self.sys_func(solk, interItr, *args)
    self.A_creation = A_creation + time.time()

    self.call_ID += 1

    # solving A.x = b
    return np.linalg.solve(A, self.b)