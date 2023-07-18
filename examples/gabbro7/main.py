"""
Created by Pedro Lima.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory".
All rights reserved. See the LICENSE.TXT file for more details.
"""

import numpy as np

from mesh_obj.mesh import CartesianMesh

from solid.solid_prop import MaterialProperties
from fluid.fluid_prop import FluidProperties
from properties import InjectionProperties, SimulationProperties
from fracture_obj.fracture import Fracture
from fracture_obj.fracture_initialization import Geometry, InitializationParameters
from controller import Controller
from utilities.utility import setup_logging_to_console

runQ = True
plotQ = True
dataPath = "./Data/gabbro7"


# setting up the verbosity level of the log at console
setup_logging_to_console(verbosity_level="debug")

if runQ:
    notchRadius_m = 10.5 * 10**-3
    mesh = CartesianMesh(Lx=notchRadius_m * 5, Ly=notchRadius_m * 5, nx=200, ny=200)

    poissonRatio = 0.29
    youngMod_GPa = 99.7
    youngModPlane_GPa = youngMod_GPa / (1 - poissonRatio**2)
    toughness_MPaSqrtMeters = 2.79  # Fracture toughness (MPa.m^1/2)
    cartersLeakoff = 0.0

    solid = MaterialProperties(
        Mesh=mesh,
        Eprime=youngModPlane_GPa,
        toughness=toughness_MPaSqrtMeters,
        Carters_coef=cartersLeakoff,
    )

    injectionRate_mlPerMin = 0.08
    injection = InjectionProperties(rate=injectionRate_mlPerMin, mesh=mesh)

    viscosity = 0.6
    fluid = FluidProperties(viscosity=viscosity)

    simulation = SimulationProperties()
    simulation.finalTime = 600
    # simulation.saveTSJump, simulation.plotTSJump = 1, 20
    simulation.set_outputFolder(dataPath)
    simulation.frontAdvancing = "explicit"
    simulation.projMethod = "LS_continousfront"

    initialState = InitializationParameters(Geometry("radial", radius=notchRadius_m))

    fracture = Fracture(
        mesh=mesh,
        init_param=initialState,
        solid=solid,
        fluid=fluid,
        injection=injection,
        simulProp=simulation,
    )

    controller = Controller(
        Fracture=fracture,
        Solid_prop=solid,
        Fluid_prop=fluid,
        Injection_prop=injection,
        Sim_prop=simulation,
    )

    controller.run()


# if plotQ:
#     from utilities.visualization import *
#     from utilities.postprocess_fracture import load_fractures

#     # loading simulation results
#     Fr_list, properties = load_fractures(dataPath)
#     solid, fluid, injection, simulation = properties
#     time_srs = get_fracture_variable(Fr_list, variable='time')                      # list of times

#     # plot fracture radius
#     plot_prop = PlotProperties()
#     plot_prop.lineStyle = '.'               # setting the line style to point
#     plot_prop.graphScaling = 'loglog'       # setting to log log plot
#     Fig_R = plot_fracture_list(Fr_list,
#                                variable='d_mean',
#                                plot_prop=plot_prop)
#     # plot analytical radius
#     Fig_R = plot_analytical_solution(regime='M',
#                                      variable='d_mean',
#                                      mat_prop=Solid,
#                                      inj_prop=Injection,
#                                      fluid_prop=Fluid,
#                                      time_srs=time_srs,
#                                      fig=Fig_R)
#     # plot analytical radius
#     Fig_R = plot_analytical_solution(regime='K',
#                                      variable='d_mean',
#                                      mat_prop=Solid,
#                                      inj_prop=Injection,
#                                      fluid_prop=Fluid,
#                                      time_srs=time_srs,
#                                      fig=Fig_R)
#     plt.show(block=True)
