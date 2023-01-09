# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on Fri Dec 16 17:49:21 2017.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2019.
All rights reserved. See the LICENSE.TXT file for more details.
"""
# imports
import numpy as np
import os
from matplotlib import pyplot as plt

# local imports
from mesh_obj.mesh import CartesianMesh
from solid.solid_prop import MaterialProperties
from fluid.fluid_prop import FluidProperties
from properties import InjectionProperties, SimulationProperties
from fracture_obj.fracture import Fracture
from controller import Controller
from fracture_obj.fracture_initialization import Geometry, InitializationParameters
from utilities.utility import setup_logging_to_console

# setting up the verbosity level of the log at console
setup_logging_to_console(verbosity_level='debug')

"""
This files reproduces the "block 4 experiment" published in https://doi.org/10.1002/2016JB013183
"""

class custom_factory():
    def __init__(self, xlabel, ylabel):
        self.data = {'xlabel' : xlabel,
                     'ylabel': ylabel,
                     'xdata': [],
                     'p_in_crack': [],
                     'p_in_line': []} # max value of x that can be reached during the simulation

    def custom_plot(self, sim_prop, fig=None):
        # this method is mandatory
        if fig is None:
            fig = plt.figure()
            ax = fig.gca()
        else:
            ax = fig.get_axes()[0]

        ax.scatter(self.data['xdata'], self.data['p_in_line'], color='b')
        ax.scatter(self.data['xdata'], self.data['p_in_crack'], color='r',marker='*')

        ax.set_xlabel(self.data['xlabel'])
        ax.set_ylabel(self.data['ylabel'])
        #ax.set_yscale('log')
        #ax.set_xscale('log')
        return fig

    def postprocess_fracture(self, sim_prop,solid_prop,fluid_prop,injection_prop,  fr):
        # this method is mandatory
        self.data['xdata'].append(fr.time)
        self.data['p_in_line'].append(fr.pInjLine/1000000)
        ID= fr.mesh.locate_element(0.,0.)
        self.data['p_in_crack'].append(fr.pFluid[ID[0]] / 1000000) #
        fr.postprocess_info = self.data
        return fr


run = True
restart = False
if run:
    # creating mesh
    Mesh = CartesianMesh(0.003, 0.003, 41, 41)

    Eprime = 3.93e9
    K1c = 1.3e6
    sigma0 = 0

    # material properties
    Solid = MaterialProperties(Mesh,
                               Eprime,
                               K1c,
                               minimum_width=1e-10,
                               Carters_coef=0.,
                               Carters_t0=None)

    # injection parameters
    def source_location(x, y, hx, hy):
        """ This function is used to evaluate if a point is included in source, i.e. the fluid is injected at the given
            point.
        """
        # the condition
        return x**2 + y**2 < 0.0001**2

    Q0 = 0.0158e-6  # injection rate
    Injection = InjectionProperties(Q0,
                                    Mesh,
                                    source_loc_func=source_location,
                                    model_inj_line=True,
                                    il_compressibility=1.0474e-10,
                                    il_volume=1.e-3,
                                    perforation_friction=0.*1e1,
                                    initial_pressure=2.e6) #1.25104)

    # fluid properties
    viscosity = 60.
    Fluid = FluidProperties(viscosity=viscosity)

    # simulation properties
    simulProp = SimulationProperties()
    simulProp.customPlotsOnTheFly = True
    simulProp.finalTime = 600                               # the time at which the simulation stops
    #simulProp.plotTSJump = 10                               # save and plot after every 5 time steps
    simulProp.set_outputFolder("./Data/injection_line")     # the disk address where the files are saved
    simulProp.custom = custom_factory('time [s]', 'pressure [MPa]')
    simulProp.plotVar = ['ir', 'w', 'custom']
    #simulProp.plotVar = [ 'w', 'pf']
    simulProp.plotFigure = True
    simulProp.frontAdvancing = 'implicit'
    simulProp.set_simulation_name('block4_ts_0.2_r_1.45')
    simulProp.maxSolverItrs = 500
    simulProp.fixedTmStp = np.asarray([[0, 15., 200., 206, 223, 300, 600], [0.4, 1., 0.1, 0.03, 1,10,None]])
    simulProp.maxFrontItrs = 50
    #simulProp.projMethod = 'ILSA_orig'
    simulProp.Anderson_parameter = 20

    # starting simulation with a static radial fracture with radius 20cm and pressure of 1MPa
    Fr_geometry = Geometry('radial', radius=1.45e-3)

    # building elasticity matrix
    simulProp.useBlockToeplizCompression = True
    from solid.elasticity_isotropic import load_isotropic_elasticity_matrix_toepliz
    C = load_isotropic_elasticity_matrix_toepliz(Mesh, Eprime)

    init_param = InitializationParameters(Fr_geometry, regime='static', net_pressure=5e2, elasticity_matrix=C) #time=2.5

    # creating fracture object
    Fr = Fracture(Mesh,
                  init_param,
                  Solid,
                  Fluid,
                  Injection,
                  simulProp)

    Fr.pInjLine = Fr.pFluid[Mesh.CenterElts]

    if restart:
        from utilities.postprocess_fracture import load_fractures
        Fr_list, properties = load_fractures("./Data/injection_line",
                                             sim_name='block4_ts_0.2_r_1.45')
        Solid, Fluid, Injection, simulProp = properties
        simulProp.frontAdvancing = 'implicit' #'explicit'
        simulProp.set_outputFolder("./Data/injection_line_restarted")
        Fr = Fr_list[-1]

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
from utilities.visualization import *
from utilities.postprocess_fracture import load_fractures

results_folder = "./Data/injection_line_restarted"
results_folder = "./Data/injection_line"
Fr_list_complete, properties = load_fractures(results_folder,
                                     sim_name='block4_ts_0.2_r_1.45')
Solid, Fluid, Injection, simulProp = properties

time_srs = get_fracture_variable(Fr_list_complete,'time')

# plot fracture radius
plot_prop = PlotProperties()

Fr_list = []
for i in range(0,len(Fr_list_complete),10):
    Fr_list.append(Fr_list_complete[i])
Fig_R = plot_fracture_list(Fr_list,
                           variable='footprint',
                           plot_prop=plot_prop)
Fig_R = plot_fracture_list(Fr_list,
                           fig=Fig_R,
                           variable='mesh',
                           mat_properties=properties[0],
                           backGround_param='K1c',
                           plot_prop=plot_prop)

# plot width at center
Fig_w = plot_fracture_list_at_point(Fr_list,
                                    variable='w',
                                    plot_prop=plot_prop)
# plot analytical width at center
Fig_w = plot_analytical_solution_at_point('M',
                                          'w',
                                          Solid,
                                          Injection,
                                          fluid_prop=Fluid,
                                          time_srs=time_srs,
                                          fig=Fig_w)

Fig_w = plot_analytical_solution_at_point('K',
                                          'w',
                                          Solid,
                                          Injection,
                                          fluid_prop=Fluid,
                                          time_srs=time_srs,
                                          fig=Fig_w)

# plot pressure at center
Fig_pf = plot_fracture_list_at_point(Fr_list,
                                    variable='pn',
                                    plot_prop=plot_prop)
# plot analytical width at center
Fig_pf = plot_analytical_solution_at_point('M',
                                          'pn',
                                          Solid,
                                          Injection,
                                          fluid_prop=Fluid,
                                          time_srs=time_srs,
                                          fig=Fig_pf)

Fig_pf = plot_analytical_solution_at_point('K',
                                          'pn',
                                          Solid,
                                          Injection,
                                          fluid_prop=Fluid,
                                          time_srs=time_srs,
                                          fig=Fig_pf)


# animate_simulation_results(Fr_list, variable=['ir'])
plt_prop = PlotProperties(line_style='.-', line_color='k')
Fig_p = plot_fracture_list(Fr_list, variable='injection line pressure', plot_prop=plt_prop)
Fig_ir = plot_fracture_list(Fr_list, variable='tir', plot_prop=plt_prop)
Fig_ir = plot_variable_vs_time(time_srs, np.full(len(time_srs), 0.0158e-6), fig=Fig_ir)
Fig_r = plot_fracture_list(Fr_list, variable='d_mean', plot_prop=plt_prop)
Fig_v = plot_fracture_list(Fr_list, variable='volume', plot_prop=plt_prop)

plt.show(block=True)
