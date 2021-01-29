# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on Fri June 17 2020.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2021.
All rights reserved. See the LICENSE.TXT file for more details.
"""

# local imports
from mesh import CartesianMesh
from properties import MaterialProperties, FluidProperties, InjectionProperties, SimulationProperties
from fracture import Fracture
from controller import Controller
from fracture_initialization import Geometry, InitializationParameters
from utility import setup_logging_to_console

# imports
import numpy as np
from scipy import interpolate
import os


# setting up the verbosity level of the log at console
setup_logging_to_console(verbosity_level='info')

# creating mesh
Mesh = CartesianMesh(0.6, 0.6, 41, 41)

# solid properties
Eprime = 10e9                           # plain strain modulus
Kprime = 3.19e6
K_Ic = Kprime / (4 * np.sqrt(2 / np.pi))                              # fracture toughness

# material properties
Solid = MaterialProperties(Mesh,
                            Eprime,
                            K_Ic)

Q0 = 0.005
Injection = InjectionProperties(Q0, Mesh)

# fluid properties
Fluid = FluidProperties(viscosity=0.75,
                        rheology='HBF',
                        compressibility=0,
                        n=0.6, k=0.75, T0=10.)


# simulation properties
simulProp = SimulationProperties()
simulProp.finalTime = 38000                             # the time at which the simulation stops
simulProp.set_outputFolder("./Data/HB")                 # the disk address where the files are saved
simulProp.set_simulation_name('HB_Gauss_Chebyshev_comparison')  # setting simulation name
simulProp.saveG = True                                  # enable saving the coefficient G
simulProp.plotVar = ['w', 'G']                          # plot width of fracture
simulProp.saveEffVisc = True                            # enable saving of the effective viscosity
simulProp.relaxation_factor = 0.3                       # relax Anderson iteration
simulProp.maxSolverItrs = 200                           # set maximum number of Anderson iterations to 200
simulProp.collectPerfData = True                        # enable collect performance data
simulProp.tolFractFront = 3e-3                          # increasing fracture front iteration tolerance
simulProp.plotTSJump = 5                                # plotting after every five time steps
simulProp.tmStpPrefactor = 0.6                          # reducing time steps for better convergence
simulProp.Anderson_parameter = 10                       # saving last 10 solutions for better performance

# initializing the fracture width with the solution provided by  Madyarova & Detournay 2004 for power-law fluids.
w = np.zeros(Mesh.NumberOfElts)
xw = np.genfromtxt('width_n_05.csv', delimiter=',')     # loading dimensionless width profile for n = 0.5
t = 2e-2
n = Fluid.n
gamma = 0.699
Mprime = 2**(n + 1) * (2 * n + 1)**n / n**n * Fluid.k
Vel = 2 * (n + 1) / (n + 2) / 3 * gamma * (Eprime * Q0 ** (n + 2) / Mprime
        ) ** (1 / (3 * n + 6)) / t ** ((n + 4) / (3 * n + 6))
eps = (Mprime / Eprime / t**n) ** (1 / (n + 2))
L = (Eprime * Q0**(n + 2) * t**(2 * n + 2) / Mprime) ** (1 / (3 * n + 6))

# interpolating width on cell centers
f = interpolate.interp1d(gamma * L * xw[:, 0],
                         L * eps * xw[:, 1],
                         bounds_error=False,
                         fill_value='extrapolate')
w = f(Mesh.distCenter)
w[w < 0] = 0.
    
# initialization parameters
Fr_geometry = Geometry('radial', radius=gamma * L)
from elasticity import load_isotropic_elasticity_matrix
C = load_isotropic_elasticity_matrix(Mesh, Eprime)
init_param = InitializationParameters(Fr_geometry,
                                      regime='static',
                                      width=w,
                                      elasticity_matrix=C,
                                      tip_velocity=Vel)


# creating fracture object
Fr = Fracture(Mesh,
              init_param,
              Solid,
              Fluid,
              Injection,
              simulProp)


# create a Controller
controller = Controller(Fr,
                        Solid,
                        Fluid,
                        Injection,
                        simulProp)

# run the simulation 
controller.run()

####################
# plotting results #
####################

if not os.path.isfile('./batch_run.txt'): # We only visualize for runs of specific examples

    from visualization import *
    from postprocess_performance import *

    # loading simulation results
    Fr_list, properties = load_fractures(address="./Data/HB",
                                          sim_name='HB_Gauss_Chebyshev_comparison')
    time_srs_PyFrac = get_fracture_variable(Fr_list, 'time')


    # plot fracture radius
    plot_prop = PlotProperties(line_style='.', graph_scaling='loglog')
    Fig_r = plot_fracture_list(Fr_list,
                                variable='d_mean',
                                plot_prop=plot_prop)


    plot_prop = PlotProperties(line_style='.', graph_scaling='loglog')
    Fig_p = plot_fracture_list_at_point(Fr_list,
                                variable='pn',
                                plot_prop=plot_prop)

    plot_prop = PlotProperties(line_style='.', graph_scaling='loglog')
    Fig_w = plot_fracture_list_at_point(Fr_list,
                                variable='w',
                                plot_prop=plot_prop)


    # import json
    f = open('data_HB_n06.json',)
    data_gc = json.load(f)
    time = np.asarray(data_gc['time'])
    time_srs_PyFrac = np.asarray(time_srs_PyFrac)

    k = Fluid.k
    n = Fluid.n
    T0 = Fluid.T0
    Mprime = 2**(n + 1) * (2 * n + 1)**n / n**n * k
    L = (Eprime * Q0**(n + 2) * time_srs_PyFrac**(2*n + 2) / Mprime)**(1 / (3*n + 6))
    eps = (Mprime / Eprime / time_srs_PyFrac**n) ** (1 / (n + 2))
    ty = np.sqrt(Eprime) * Mprime ** (1 / n) * T0 ** -((2 + n) / 2 / n)

    ax_r = Fig_r.get_axes()[0]
    ax_p = Fig_p.get_axes()[0]
    ax_w = Fig_w.get_axes()[0]

    ax_p.loglog(time, np.asarray(data_gc['pres'])/1e6)
    p = Eprime * eps * 1.0
    ax_p.loglog(time_srs_PyFrac, p/1e6)

    ax_w.loglog(time, np.asarray(data_gc['opening'])*1e3)
    w = L * eps * 1.19
    ax_w.loglog(time_srs_PyFrac, w*1e3)

    ax_r.loglog(time, np.asarray(data_gc['length']))
    R = L * 0.7155
    ax_r.plot(time_srs_PyFrac, R, label='analytical Power law n=0.6')

    from postprocess_performance import *
    plt_prop_Pic = PlotProperties(graph_scaling='semilogx', line_style='.')
    plot_performance(address="./Data/HB",
                     sim_name='HB_Gauss_Chebyshev_comparison',
                     variable='Picard iterations',
                     plot_prop=plt_prop_Pic)
    plt.show()

