# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Carlo Peruzzo on 20.10.21.
Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory, 2016-2020.
All rights reserved. See the LICENSE.TXT file for more details.
"""

# External import
import logging

# Internal import

class Linear_solver():
  def __init__(self, sys_func):
    self.log = logging.getLogger('PyFrac.LinearSolver')
    self.sys_func = sys_func

  def solve(self, solk, interItr, *args):
    raise NotImplementedError()