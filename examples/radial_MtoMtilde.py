# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Andreas Möri on Fri Feb 14 10:35:25 2017.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2019.
All rights reserved. See the LICENSE.TXT file for more details.
"""

# imports
import os

# local imports
from mesh import CartesianMesh
from properties import MaterialProperties, FluidProperties, InjectionProperties, SimulationProperties
from fracture import Fracture
from controller import Controller
from fracture_initialization import Geometry, InitializationParameters
from utility import setup_logging_to_console

# setting up the verbosity level of the log at console
setup_logging_to_console(verbosity_level='info')

# creating mesh
Mesh = CartesianMesh(50, 50, 41, 41)

# solid properties
nu = 0.4                            # Poisson's ratio
youngs_mod = 3.3e10                 # Young's modulus
Eprime = youngs_mod / (1 - nu**2)   # plain strain modulus
K1c = 0                             # zero toughness case
Cl = 0.5e-6                         # Carter's leak off coefficient

# material properties
Solid = MaterialProperties(Mesh,
                           Eprime,
                           K1c,
                           Carters_coef=Cl,
                           minimum_width=1e-12)

# injection parameters
Q0 = 0.01  # injection rate
Injection = InjectionProperties(Q0, Mesh)

# fluid properties
viscosity = 0.001 / 12  # mu' =0.001
Fluid = FluidProperties(viscosity=viscosity)

# simulation properties
simulProp = SimulationProperties()
simulProp.finalTime = 3e7                      # the time at which the simulation stops
simulProp.saveTSJump, simulProp.plotTSJump = 3, 5 # save and plot after every 5 time steps
simulProp.set_outputFolder("./Data/MtoMt_FO")     # the disk address where the files are saved
simulProp.plotVar = ['w', 'regime']

# initializing fracture
Fr_geometry = Geometry('radial')
init_param = InitializationParameters(Fr_geometry, regime='M', time=50)


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

    # loading simulation results
    Fr_list, properties = load_fractures(address="./Data/MtoMt_FO")       # load all fractures
    time_srs = get_fracture_variable(Fr_list,                             # list of times
                                     variable='time')

    # plotting efficiency
    plot_prop = PlotProperties(graph_scaling='loglog',
                               line_style='.')
    label = LabelProperties('efficiency')
    label.legend = 'fracturing efficiency'
    Fig_eff = plot_fracture_list(Fr_list,
                               variable='efficiency',
                               plot_prop=plot_prop,
                               labels=label)

    # solution taken from matlab code provided by Madyarova, 2003
    t = np.asarray([50., 120.51, 284.964, 725.383, 1733.2, 4270.2, 9711.62, 21815.4,
                    44260.6, 84765.4, 150509., 1.15035*1e6, 3.93354*1e6, 9.17576*1e6,
                    1.73278*1e7, 2.87601*1e7])
    eff_analytical = np.asarray([0.954627, 0.93687, 0.912921, 0.880517, 0.839712, 0.78455, 0.724188,
                                 0.651409, 0.58109, 0.513443, 0.452043, 0.254634, 0.167084, 0.119908,
                                 0.0934669, 0.0776936])
    ax_r = Fig_eff.get_axes()[0]
    ax_r.loglog(t, eff_analytical, 'b-', label='semi-anlytical efficiency')
    ax_r.legend()

    # plot fracture radius
    plot_prop = PlotProperties()
    plot_prop.lineStyle = '.'               # setting the linestyle to point
    plot_prop.graphScaling = 'loglog'       # setting to log log plot
    label = LabelProperties('d_mean')
    label.legend = 'radius'
    Fig_R = plot_fracture_list(Fr_list,
                               variable='d_mean',
                               plot_prop=plot_prop) # numerical radius

    # solution taken from matlab code provided by Madyarova, 2003
    r_analytical = np.asarray([27.1406, 39.7624, 57.5537, 85.6466, 123.165, 177.722, 247.294,
                               338.427, 441.007, 558.138, 680.608, 1302.06, 1857.74, 2351.25, 2791.48,
                               3190.64])

    ax_r = Fig_R.get_axes()[0]
    ax_r.loglog(t, r_analytical, 'b-', label='semi-anlytical radius')
    ax_r.legend()

    # plot analytical M-vertex solution for radius
    plt_prop = PlotProperties(line_color_anal='r')
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

    # plot analytical Mtilde-vertex solution for radius
    plt_prop = PlotProperties(line_color_anal='g')
    label = LabelProperties('d_mean')
    label.legend = 'Mt solution'
    Fig_R = plot_analytical_solution(regime='Mt',
                                     variable='d_mean',
                                     labels=label,
                                     mat_prop=properties[0],
                                     inj_prop=properties[2],
                                     fluid_prop=properties[1],
                                     time_srs=time_srs,
                                     plot_prop=plt_prop,
                                     fig=Fig_R)

    # plot slice of width
    time_slice = np.asarray([1e6, 7.5e6, 3e7])
    Fr_slice, properties = load_fractures(address="./Data/MtoMt_FO",
                                          time_srs=time_slice)       # load specific fractures
    time_slice = get_fracture_variable(Fr_slice,
                                       variable='time')

    ext_pnts = np.empty((2, 2), dtype=np.float64)
    Fig_WS = plot_fracture_list_slice(Fr_slice,
                                      variable='w',
                                      projection='2D',
                                      plot_cell_center=True,
                                      extreme_points=ext_pnts)
    # plot slice of width analytical
    Fig_WS = plot_analytical_solution_slice('Mt',
                                            'w',
                                            Solid,
                                            Injection,
                                            time_srs=time_slice,
                                            fluid_prop=Fluid,
                                            fig=Fig_WS,
                                            point1=ext_pnts[0],
                                            point2=ext_pnts[1])

    plt.show(block=True)