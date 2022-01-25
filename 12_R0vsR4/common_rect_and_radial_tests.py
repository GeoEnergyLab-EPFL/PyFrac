#!/usr/bin/env python3

# External imports
import numpy as np
from scipy.sparse.linalg import bicgstab
import logging
import matplotlib.pyplot as plt


# Internal imports
from mesh_obj.mesh import CartesianMesh
from utilities.utility import setup_logging_to_console
from linear_solvers.preconditioners.prec_back_subst_EHL import EHL_iLU_Prec
from linear_solvers.linear_iterative_solver import iteration_counter


# setting up the verbosity level of the log at console
setup_logging_to_console(verbosity_level='debug')
log = logging.getLogger('PyFrac.solve_width_pressure')


# ----------------------------------------------
def KI_2DPS_solution(p, H):
    # SIF of a plane strain crack of total height H
    return p * np.sqrt(np.pi * H / 2.)

def KI_radial_solution(p,R):
    return 2. * p * np.sqrt(R) / np.sqrt(np.pi)


def w_radial_solution(x,y,Young,nu,p,R):
    rr = x**2 + y**2
    if rr < R**2:
        return 8. * (1 - nu * nu) * p * np.sqrt(R**2 - rr) / (np.pi * Young)
    else:
        return 0.

def sig_zz_radial_solution(x,y,p,R):
    rr = np.sqrt(x**2 + y**2)
    rho = rr / R
    sig_zz = - (2. * p / (np.pi)) * (np.arcsin(1. / rho) - 1. / np.sqrt(rho * rho - 1.))
    return sig_zz


def Volume_radial_solution(Young,nu,p,R):
    return 16./3. * p * R**3 * (1 - nu * nu)/ (Young)


def wmax_plane_strain_solution(Young,nu,p,H):
    return 2. * H * p * (1 - nu * nu) / (Young)
# ----------------------------------------------

def plot_3d_scatter(zdata, xdata, ydata, zlabel = 'z', xlabel = 'x', ylabel = 'y'):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='viridis')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    return fig

# ----------------------------------------------

def get_solution(C, p, EltCrack, x0=None, TipCorr = None):
    """

    :param C: elasticity matrix
    :param p: pressure
    :param EltCrack: IDs of the elements in the crack
    :return: solution x of the system C.x = p
    """

    # prepare preconditioner
    Aprec = EHL_iLU_Prec(C._get9stencilC(EltCrack))
    counter = iteration_counter(log=log)  # to obtain the number of iteration and residual
    if TipCorr is not None:
        FillF, EltTip = TipCorr
        C._set_tipcorr(FillF, EltTip)
    C._set_domain_and_codomain_IDX(EltCrack, EltCrack, same_domain_and_codomain=True)


    sol_ = bicgstab(C, p, x0=x0, M=Aprec, atol=10.e-14, tol=1.e-9, maxiter=1000, callback=counter)

    if sol_[1] > 0:
        print("     --> iterative solver did NOT converge after " + str(sol_[1]) + " iterations!")
    elif sol_[1] == 0:
        print("     --> iterative solver converged after " + str(counter.niter) + " iter. ")
    return sol_[0]


# ----------------------------------------------
def get_mesh(sim_info, refinement_ID):

    ar = sim_info["aspect ratio"]
    lx = sim_info["domain x"][1] - sim_info["domain x"][0]
    ly = lx * ar
    d_x = sim_info["domain x"]
    d_y = [-ly / 2., ly / 2.]

    # define nx
    if refinement_ID < sim_info["n. of refinements x"]:
        nx = refinement_ID * sim_info["nx min"]
    else:
        nx = sim_info["nx min"]

    # define ny
    if refinement_ID < sim_info["n. of refinements y"]:
        ny = refinement_ID * sim_info["nx min"] * ar
    else:
        ny = sim_info["nx min"] * ar

    # define ny should be an odd number
    if ny % 2 == 0 : ny = ny + 1
    if nx % 2 == 0 : nx = nx + 1

    return CartesianMesh(d_x, d_y, nx, ny)