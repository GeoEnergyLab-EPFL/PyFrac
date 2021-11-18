# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Carlo Peruzzo on 20.10.21.
Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory, 2016-2020.
All rights reserved. See the LICENSE.TXT file for more details.
"""

# External imports
import numpy as np
from scipy.sparse.linalg import gmres
from scipy.sparse.linalg import bicgstab
import time

# Internal import
from linear_solvers.linear_solver import Linear_solver
from utilities.utility import append_new_line

class Iterative_linear_solver(Linear_solver):
  def __init__(self, sys_func, atol, maxiter, gmresRestart, prec_func = None, solver_type = 'bicgstab', rcmp_prec_before2ndIter = False):
    super().__init__(sys_func)
    self.prec_func = prec_func
    self.prec = None
    self.interItr = None
    self.b = None
    self.indices = None
    self.call_ID = 0
    self.cumulativeITERsSOLVER = 0
    self.rcmp_prec_before2ndIter = rcmp_prec_before2ndIter

    # iterative solver params
    self.solver_type = solver_type
    self.atol = atol
    self.maxiter = maxiter
    #       gmres specific params
    self.gmresRestart = gmresRestart

    if solver_type == 'gmres':
        self.solver_call = self.call_gmres
    elif solver_type == 'bicgstab':
        self.solver_call = self.call_bicgstab


  def solve(self, solk, interItr, *args):

      # to obtain the number of iteration and residual
      self.counter = iteration_counter(self.log)

      if self.prec_func is not None and (self.call_ID == 0 or self.rcmp_prec_before2ndIter and self.call_ID < 2):
          A_creation = -time.time()
          sys_size = len(solk)
          if sys_size < 3000:
            decay_tshold = 0.81
            fill_factor = 10
            #probability = 1.
          # elif sys_size < 8000:
          #   decay_tshold = 0.71
          #   fill_factor = 15
          #   #probability = 1.
          elif sys_size < 9000: #19000
               decay_tshold = 0.81
               fill_factor = 20
          #   #decay_tshold = 0.51
          #   decay_tshold = 0.71
          #   fill_factor = 20
            #probability = 0.5
          else:
              decay_tshold = 0.81
              fill_factor = 30
          # (A, self.b, self.interItr, self.indicies) = self.sys_fun._getsys(solk, interItr, *args)
          (A, self.b, self.interItr, self.indices) = self.sys_func._getsys_simplif(solk, interItr, *args, decay_tshold=decay_tshold, probability=1.)
          A_creation = A_creation + time.time()
          ILU_comp = -time.time()
          self.prec = self.prec_func(A, drop_tol=1.e-10, fill_factor=fill_factor)
          ILU_comp = ILU_comp + time.time()
          self.ILU_comp = ILU_comp
          self.A_creation = A_creation
      else:
          # to update the system A and RHS b
          (self.b, self.interItr, self.indices) = self.sys_func._update_sys(solk, interItr)

      x0 = self.residual_iter(solk, niter = 1)

      # solve the system
      sol_ = self.solver_call(self.counter, x0=x0)

      # check solution
      if sol_[1] > 0:
          self.log.warning("Iterative solver did NOT converge after " + str(sol_[1]) + " iterations!")
      elif sol_[1] == 0:
          self.log.debug(" --> iterative solver converged after " + str(self.counter.niter) + " iter. (system size: " + str(len(self.b))  + " )")
      # file_name = "/Users/carloperuzzo/Desktop/Pyfrac_formulation/_gmres_dev/_preconditioner/_data&performances/Gmres_with_parallel_Hdot/_data_Elast_stencil/gmres_iter_vs_size.txt"
      # append_new_line(file_name, str(len(b)) + ' ' + str(self.counter.niter))
      self.call_ID += 1
      self.cumulativeITERsSOLVER += self.counter.niter
      return sol_[0]

  def call_gmres(self, counter, x0 = None):
      sol_GMRES = gmres(self.sys_func,
                        self.b,
                        # x0=x0,
                        M=self.prec,
                        atol=self.atol,
                        tol=1.e-9,
                        maxiter=self.maxiter,
                        callback=counter,
                        restart=self.gmresRestart)
      return sol_GMRES

  def call_bicgstab(self, counter, x0 = None):
       sol_BCGSTAB = bicgstab(self.sys_func,
                              self.b,
                              x0=x0,
                              M=self.prec,
                              atol=self.atol,
                              tol=1.e-9,
                              maxiter=self.maxiter,
                              callback=counter)
       return sol_BCGSTAB

  def residual_iter(self, x0, niter = 5):
      for i in range(niter):
          rk = self.get_residual(x0)
          self.log.debug("Residual corr: " + str(np.linalg.norm(rk)))
          x0 = x0 + self.prec._matvec(rk)
      return x0

  def get_residual(self, xk):
      return self.b-self.sys_func._matvec(xk)

class iteration_counter(object):
    def __init__(self, log, disp=True):
        self._disp = disp
        self.niter = 0
        self.threshold = 100
        self.log = log
    def __call__(self, rk=None):
        self.niter += 1
        if self._disp:
            if self.niter == self.threshold:
                self.log.warning('Iterative solver has not converged in '+str(self.niter)+' iter, monitoring the residual')
            if self.niter > self.threshold:
                self.log.warning('iter %3i\trk = %s' % (self.niter, str(np.linalg.norm(rk))))