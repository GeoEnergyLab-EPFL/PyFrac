# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on Fri Dec 16 17:49:21 2017.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2019.
All rights reserved. See the LICENSE.TXT file for more details.
"""
from line_profiler import LineProfiler
profile = LineProfiler()
# imports
import os
import numpy as np
from line_profiler import LineProfiler
import psutil

# local imports
from mesh_obj.mesh import CartesianMesh
from solid.solid_prop import MaterialProperties
from solid.elasticity_isotropic import load_isotropic_elasticity_matrix_toepliz
from fluid.fluid_prop import FluidProperties
from properties import InjectionProperties, SimulationProperties
from fracture_obj.fracture import Fracture
from controller import Controller
from fracture_obj.fracture_initialization import Geometry, InitializationParameters
from utilities.utility import setup_logging_to_console
from utilities.postprocess_fracture import load_fractures

# setting up the verbosity level of the log at console
setup_logging_to_console(verbosity_level='debug')
# profile = LineProfiler()
# creating mesh
#Mesh = CartesianMesh(10, 10, 121, 121)
#Mesh = CartesianMesh(20, 20, 241, 241)
Mesh = CartesianMesh(20, 20, 501, 501)

# solid properties
nu = 0.25
G = 1.0
youngs_mod = 2.0 * G * (1.0 + nu)

#nu = 0.4                            # Poisson's ratio
#youngs_mod = 3.3e10                 # Young's modulus
Eprime = youngs_mod / (1 - nu**2)   # plain strain modulus
K1c = 1e6                           # Fracture toughness



# material properties
Solid = MaterialProperties(Mesh,
                           Eprime,
                           K1c)

# injection parameters
Q0 = 0.01  # injection rate
Injection = InjectionProperties(Q0, Mesh)

# fluid properties
viscosity = 0.001                         # mu' =0.001
Fluid = FluidProperties(viscosity=viscosity)

# simulation properties
simulProp = SimulationProperties()
simulProp.finalTime = 140.155421804482037                        # the time at which the simulation stops
simulProp.saveTSJump, simulProp.plotTSJump = 1, 1   # save and plot after every 5 time steps
simulProp.set_outputFolder("./Data/MtoK")   # the disk address where the files are saved
simulProp.plotVar = ['regime', 'w']
simulProp.EHL_GMRES = True
simulProp.solve_monolithic = False
simulProp.useBlockToeplizCompression = True
simulProp.frontAdvancing = 'implicit'
#simulProp.elastohydrSolver = 'implicit_Picard'
# simulProp.gmres_tol = 1.e-6

# initializing fracture
Fr_geometry = Geometry('radial',radius=1.598) #1

# gives an object with many fields
#a = psutil.virtual_memory()
#used0 = a.used/1024/1024/1024
#print(used0)
C = load_isotropic_elasticity_matrix_toepliz(Mesh, Eprime, C_precision = np.float64, useHMATdot=True, nu=nu)
#a = psutil.virtual_memory()
#used1= a.used/1024/1024/1024
#print(used1)
#exit()
init_param = InitializationParameters(Fr_geometry, regime='M', time=0.05, elasticity_matrix=C)
# init_param = InitializationParameters(Fr_geometry,
#                                       regime='static',
#                                       net_pressure=36.2e3,
#                                       elasticity_matrix=C,
#                                       time=0.698806)

# creating fracture object
Fr = Fracture(Mesh,
              init_param,
              Solid,
              Fluid,
              Injection,
              simulProp)

restart = False
if restart:
    from utilities.visualization import *

    Fr_list, properties = load_fractures(address="./Data/MtoK",
                                         step_size=1, load_all=True)  # load all fractures                                                # list of times
    Solid, Fluid, Injection, simulProp = properties
    #simulProp.finalTime = 24.155421804482037
    Fr = Fr_list[-1]



# create a Controller
controller = Controller(Fr,
                        Solid,
                        Fluid,
                        Injection,
                        simulProp,
                        C=C)

# run the simulation
controller.run()


####################
# plotting results #
####################
#
plotOUTPUT= False
if plotOUTPUT:
    if not os.path.isfile('./batch_run.txt'): # We only visualize for runs of specific examples

        from utilities.visualization import *

        # loading simulation results
        Fr_list, properties = load_fractures(address="./Data/MtoK")       # load all fractures
        time_srs = get_fracture_variable(Fr_list,                             # list of times
                                         variable='time')


        # solution taken from matlab code provided by Madyarova, 2003
        t = np.asarray([0.0501187, 0.0815387, 0.132656, 0.21582, 0.351119, 0.571239,
                        0.929355, 1.51198, 2.45985, 4.00196, 6.51083, 10.5925, 17.2331,
                        28.0367, 45.6132, 74.2087, 120.731, 196.418, 319.555, 519.887,
                        845.81, 1376.06, 2238.72, 3642.2, 5925.53, 9640.31, 15683.9, 25516.3,
                        41512.8, 67537.6, 109878., 178761., 290828., 473151., 769775.,
                        1.25235 * 1e6, 2.03747 * 1e6, 3.31478 * 1e6, 5.39285 * 1e6, 8.77368 * 1e6,
                        1.4274 * 1e7, 2.32225 * 1e7, 3.77809 * 1e7, 6.14662 * 1e7, 1.e8,
                        1.25893 * 1e8, 2.11349 * 1e8, 3.54813 * 1e8, 5.95662 * 1e8, 1.e9])

        r_analytical = np.asarray([0.976993, 1.21262, 1.505, 1.86785, 2.318, 2.87639, 3.56895, 4.42805,
                                   5.49363, 6.81433, 8.451, 10.478, 12.9904, 16.1016, 19.9513, 24.7158,
                                   30.602, 37.8899, 46.8844, 57.9802, 71.662, 88.5219, 109.324, 134.876,
                                   166.266, 204.75, 252.014, 309.977, 380.785, 467.467, 572.913,
                                   702.117, 859.517, 1051.05, 1284.4, 1567.95, 1914.05, 2334.17,
                                   2844.68, 3465.29, 4219.73, 5137.54, 6251.74, 7605.68, 9250.9, 10183.,
                                   12527.8, 15412.6, 18961.7, 23327.9])

        # plot fracture radius
        plot_prop = PlotProperties()
        plot_prop.lineStyle = '.'               # setting the linestyle to point
        plot_prop.graphScaling = 'loglog'       # setting to log log plot
        label = LabelProperties('d_mean')
        label.legend = 'radius'
        Fig_R = plot_fracture_list(Fr_list,
                                   variable='d_mean',
                                   plot_prop=plot_prop) # numerical radius


        ax_r = Fig_R.get_axes()[0]
        ax_r.loglog(t, r_analytical, 'g-', label='semi-anlytical radius')
        ax_r.legend()

        # plot analytical M-vertex solution for radius
        plt_prop = PlotProperties(line_color_anal='b')
        label = LabelProperties('d_mean')
        label.legend = 'M solution'
        Fig_R = plot_analytical_solution(regime='M',
                                         variable='d_mean',
                                         labels=label,
                                         mat_prop=properties[0],
                                         inj_prop=properties[2],
                                         fluid_prop=properties[1],
                                         time_srs=time_srs,
                                         plot_prop=plt_prop,
                                         fig=Fig_R)

        # plot analytical K-vertex solution for radius
        plt_prop = PlotProperties(line_color_anal='r')
        label = LabelProperties('d_mean')
        label.legend = 'K solution'
        Fig_R = plot_analytical_solution(regime='K',
                                         variable='d_mean',
                                         labels=label,
                                         mat_prop=properties[0],
                                         inj_prop=properties[2],
                                         fluid_prop=properties[1],
                                         time_srs=time_srs,
                                         plot_prop=plt_prop,
                                         fig=Fig_R)

        # # plot slice of width
        # time_slice = np.asarray([1, 10, 1e2, 1e8, 5e8, 1e9])
        # Fr_slice, properties = load_fractures(address="./Data/MtoK",
        #                                       time_srs=time_slice)       # load specific fractures
        # time_slice = get_fracture_variable(Fr_slice,
        #                                    variable='time')
        #
        # ext_pnts = np.empty((2, 2), dtype=np.float64)
        # Fig_WS_K = plot_fracture_list_slice(Fr_slice[3:],
        #                                   variable='w',
        #                                   projection='2D',
        #                                   plot_cell_center=True,
        #                                   extreme_points=ext_pnts)
        # # plot slice of width analytical
        # Fig_WS_K = plot_analytical_solution_slice('K',
        #                                         'w',
        #                                         Solid,
        #                                         Injection,
        #                                         time_srs=time_slice[3:],
        #                                         fluid_prop=Fluid,
        #                                         fig=Fig_WS_K,
        #                                         point1=ext_pnts[0],
        #                                         point2=ext_pnts[1])
        #
        # ext_pnts = np.empty((2, 2), dtype=np.float64)
        # Fig_WS_M = plot_fracture_list_slice(Fr_slice[:3],
        #                                   variable='w',
        #                                   projection='2D',
        #                                   plot_cell_center=True,
        #                                   extreme_points=ext_pnts)
        # # plot slice of width analytical
        # Fig_WS_M = plot_analytical_solution_slice('M',
        #                                         'w',
        #                                         Solid,
        #                                         Injection,
        #                                         time_srs=time_slice[:3],
        #                                         fluid_prop=Fluid,
        #                                         fig=Fig_WS_M,
        #                                         point1=ext_pnts[0],
        #                                         point2=ext_pnts[1])

        plt.show(block=True)