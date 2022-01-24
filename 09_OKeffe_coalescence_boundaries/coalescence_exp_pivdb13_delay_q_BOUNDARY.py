# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on Fri June 16 17:49:21 2017.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2019. All rights reserved.
See the LICENSE.TXT file for more details.
"""

import numpy as np

# local imports
from mesh_obj.mesh import CartesianMesh
from solid.solid_prop import MaterialProperties
from fluid.fluid_prop import FluidProperties
from properties import InjectionProperties, SimulationProperties
from fracture_obj.fracture import Fracture
from controller import Controller
from fracture_obj.fracture_initialization import Geometry, InitializationParameters
from utilities.utility import setup_logging_to_console
from solid.elasticity_boundary_effect_get_traction import BoundaryEffect
from utilities.postprocess_fracture import load_fractures

# setting up the verbosity level of the log at console
setup_logging_to_console(verbosity_level='debug')

run = True
export_results = True

"_______________________________________________________"
# creating mesh
#mesh_discretiz_x=265
#mesh_discretiz_y=171
mesh_discretiz_x=161
mesh_discretiz_y=101
Mesh = CartesianMesh(0.025, 0.015, mesh_discretiz_x, mesh_discretiz_y)

# solid properties
nu = 0.48                            # Poisson's ratio
youngs_mod = 97000                   # Young's modulus (+/- 10) #kPa
Eprime = youngs_mod / (1 - nu ** 2)  # plain strain modulus
K_Ic = np.sqrt(2*5.2*Eprime)         # fracture toughness (+/- 1)
Cl = 0.0                             # Carter's leak off coefficient

Solid = MaterialProperties(Mesh,
                           Eprime,
                           K_Ic,
                           confining_stress=0.,
                           Carters_coef=Cl,
                           minimum_width=1e-9)

path =  '/home/carlo/BigWhamLink/BigWhamLink/Examples/StaticCrackBenchmarks/boundary_effect_mesh.json'
Boundary = BoundaryEffect(Mesh, Eprime, nu, path,
                          preconditioner=False,
                          lgmres=True,
                          maxLeafSizeTrK=300,
                          etaTrK=10,
                          maxLeafSizeDispK=6000,
                          etaDispL=50,
                          epsACA=0.001)
# Boundary = None


def source_location(x, y, hx, hy):
    """ This function is used to evaluate if a point is included in source, i.e. the fluid is injected at the given
        point.
    """
    # the condition
    if (abs(x + .02) < hx*0.5 and abs(y - 0) < hy*.5 ) or (abs(x - .02) < hx*0.5 and abs(y + 0) < hy*.5):

        return True

# injection parameters
Q0 = 5/1000/60/1000 #20mL/min  # injection rate
initialratesecondpoint=0
#ratesecondpoint=np.asarray([[2.833], [Q0/2]])
ratesecondpoint=np.asarray([[1.09865], [Q0/2]])
delayed_second_injpoint_loc=np.asarray([-0.02,0])
Injection = InjectionProperties(Q0, Mesh, source_loc_func=source_location,
                                initial_rate_delayed_second_injpoint=initialratesecondpoint,
                                rate_delayed_second_injpoint=ratesecondpoint,
                                delayed_second_injpoint_loc=delayed_second_injpoint_loc)

# fluid properties
viscosity = 0.44 #Pa.s
Fluid = FluidProperties(viscosity=viscosity)

# simulation properties
simulProp = SimulationProperties()
simulProp.finalTime = 25                     # the time at which the simulation stops
myfolder ="./Data/coalescence_exp_pivdb13_"+str(mesh_discretiz_x)+"x"+str(mesh_discretiz_y)+"_BOUNDARY"
simulProp.set_outputFolder(myfolder)     # the address of the output folder
#simulProp.set_solTimeSeries(np.asarray([6.0,6.4, 6.45, 6.5, 6.55, 6.6, 6.62, 6.8]))
##simulProp.set_solTimeSeries(np.concatenate((np.linspace(0.9, 2.00,100),np.linspace(2.833, 4.88,150),np.linspace(21.0, 23.0,30))))
simulProp.set_solTimeSeries(np.concatenate((np.linspace(0.69, 1.08,100),np.linspace(1.0833, 3.88,250),np.linspace(21.0, 23.0,30))))
simulProp.plotTSJump = 20
simulProp.timeStepLimit=.2
simulProp.saveToDisk=True
simulProp.saveFluidFluxAsVector=True
simulProp.saveFluidVelAsVector=True
#simulProp.plotVar = ['pf', 'ffvf']
#simulProp.plotVar = ['footprint']
simulProp.projMethod = 'LS_continousfront'
simulProp.frontAdvancing = 'implicit'
#simulProp.set_tipAsymptote('K')
simulProp.maxFrontItrs=35
simulProp.maxSolverItrs=240

if run:
    # initializing fracture
    from fracture_obj.fracture_initialization import get_radial_survey_cells
    #initRad1 = 0.000802
    initRad1 = 0.0016
    initRad2 = initRad1
    surv_cells_1, surv_cells_dist_1, inner_cells_1 = get_radial_survey_cells(Mesh, initRad1, center=[-0.02, 0])
    # surv_cells_dist = np.cos(Mesh.CenterCoor[surv_cells, 0]) + 2.5 - abs(Mesh.CenterCoor[surv_cells, 1])

    surv_cells_2, surv_cells_dist_2, inner_cells_2 = get_radial_survey_cells(Mesh, initRad2, center=[0.02, 0])
    surv_cells = np.concatenate((surv_cells_1, surv_cells_2))
    surv_cells_dist = np.concatenate((surv_cells_dist_1, surv_cells_dist_2))
    inner_cells = np.concatenate((inner_cells_1, inner_cells_2))

    Fr_geometry = Geometry(shape='level set',
                           survey_cells=surv_cells,
                           tip_distances=surv_cells_dist,
                           inner_cells=inner_cells)

    from solid.elasticity_kernels.isotropic_R0_elem import load_isotropic_elasticity_matrix

    C = load_isotropic_elasticity_matrix(Mesh, Eprime)

    # init_param = InitializationParameters(Fr_geometry,
    #                                       regime='static',
    #                                       net_pressure=9.e3,
    #                                       elasticity_matrix=C)
    init_param = InitializationParameters(Fr_geometry,
                                          regime='static',
                                          net_pressure=36.2e3,
                                          elasticity_matrix=C,
                                          time=0.698806)
                                          #time=0.984)

    # creating fracture object
    Fr = Fracture(Mesh,
                  init_param,
                  Solid,
                  Fluid,
                  Injection,
                  simulProp,
                  boundaryEffect = Boundary)


    # create a Controller
    controller = Controller(Fr,
                            Solid,
                            Fluid,
                            Injection,
                            simulProp,
                            boundaryEffect = Boundary)

    # run the simulation
    controller.run()

####################
# plotting results #
####################
write = True
if export_results   :
    myJsonName="Z:\exp_pivdb13_shared_folder\coalescence_expPivdb13_4mma2.json"
    myfolder="./Data/coalescence_exp_pivdb13_265x171"
    from utilities.visualization import *

    # loading five fractures from the simulation separated with equal time period

    # Fr_list, properties = load_fractures(address="./my_examples/Data/coalescence_exp3")
    # ,time_srs=np.linspace(5.5, 6.09,100))
    # time_srs=np.linspace(5.875, 7.28,40))

    # Fr_list, properties = load_fractures(address=myfolder, time_srs=np.linspace(5.751, 5.752, 600))
    # from utility import plot_as_matrix
    # selected = 0
    # K = np.zeros((Fr_list[selected].mesh.NumberOfElts,), )
    # fr = np.where(np.logical_and(Fr_list[selected].sgndDist>-1e50, Fr_list[selected].sgndDist<1e50 ))
    # K[fr] = Fr_list[selected].sgndDist[fr]
    # plot_as_matrix(K, Fr_list[selected].mesh)

    # +++
    # Fr_list, properties = load_fractures(address=myfolder ,time_srs=np.linspace(5., 8.0,600))
    Fr_list, properties = load_fractures(address=myfolder)
    Solid, Fluid, Injection, simulProp = properties
    time_srs = get_fracture_variable(Fr_list,
                                     variable='time')

    from utilities.postprocess_fracture import append_to_json_file

    # append_to_json_file(file_name, content, action, key=None, delete_existing_filename=False)

    if write:
        append_to_json_file(myJsonName, time_srs, 'append2keyASnewlist', key='time_srs_of_Fr_list',
                            delete_existing_filename=True)

        fracture_fronts = []
        numberof_fronts = []
        for fracture in Fr_list:
            fracture_fronts.append(np.ndarray.tolist(fracture.Ffront))
            numberof_fronts.append(fracture.number_of_fronts)
        append_to_json_file(myJsonName, fracture_fronts, 'append2keyASnewlist', key='Fr_list')
        append_to_json_file(myJsonName, numberof_fronts, 'append2keyASnewlist', key='Number_of_fronts')
        append_to_json_file(myJsonName,
                            [Fr_list[-1].mesh.Lx, Fr_list[-1].mesh.Ly, Fr_list[-1].mesh.nx, Fr_list[-1].mesh.ny],
                            'append2keyASnewlist', key='mesh_info')

    ########## TAKE A VERTICAL SECTION TO GET w AT THE MIDDLE ########
    # plot fracture radius
    plot_prop = PlotProperties()

    # Fig_R = plot_fracture_list(Fr_list,
    #                            variable='footprint',
    #                            plot_prop=plot_prop)
    # Fig_R = plot_fracture_list(Fr_list,
    #                            fig=Fig_R,
    #                            variable='mesh',
    #                            mat_properties=properties[0],
    #                            backGround_param='K1c',
    #                            plot_prop=plot_prop)

    # ++++
    # Fr_list, properties = load_fractures(address=myfolder ,time_srs=np.linspace(5., 8.0,600))
    Fr_list, properties = load_fractures(address=myfolder)
    ext_pnts = np.empty((2, 2), dtype=np.float64)
    fracture_list_slice = plot_fracture_list_slice(Fr_list,
                                                   variable='w',
                                                   projection='2D',
                                                   plot_cell_center=True,
                                                   extreme_points=ext_pnts,
                                                   orientation='vertical',
                                                   point1=[0., -0.008],
                                                   point2=[0., 0.008], export2Json=True)
    # +++
    # point1 = [0., -0.018],
    # point2 = [0., 0.018], export2Json = True)

    # fracture_list_slice[new_key] = fracture_list_slice[old_key]
    # del fracture_list_slice[old_key]

    if write:
        towrite = {'intersectionVslice': fracture_list_slice}
        append_to_json_file(myJsonName, towrite, 'extend_dictionary')

    ########## TAKE A HORIZONTAL SECTION TO GET w AT THE MIDDLE ########
    ext_pnts = np.empty((2, 2), dtype=np.float64)
    fracture_list_slice = plot_fracture_list_slice(Fr_list,
                                                   variable='w',
                                                   projection='2D',
                                                   plot_cell_center=True,
                                                   extreme_points=ext_pnts,
                                                   orientation='horizontal',
                                                   point1=[-0.025, 0.0],
                                                   point2=[0.025, 0.], export2Json=True)
    # +++
    # point1 = [-0.035, 0.0],
    # point2 = [0.035, 0.], export2Json = True)

    # fracture_list_slice[new_key] = fracture_list_slice[old_key]
    # del fracture_list_slice[old_key]
    if write:
        towrite = {'intersectionHslice': fracture_list_slice}
        append_to_json_file(myJsonName, towrite, 'extend_dictionary')

    ########## IMPORT THE COMPLETE SERIES OF FRACTURE FOOTPRINTS ########
    Fr_list_COMPLETE, properties = load_fractures(address=myfolder)
    Solid, Fluid, Injection, simulProp = properties
    time_srs_COMPLETE = get_fracture_variable(Fr_list_COMPLETE,
                                              variable='time')

    if write:
        fracture_fronts = []
        numberof_fronts = []
        for fracture in Fr_list_COMPLETE:
            fracture_fronts.append(np.ndarray.tolist(fracture.Ffront))
            numberof_fronts.append(fracture.number_of_fronts)

        complete_footprints = {'time_srs_of_Fr_list': time_srs_COMPLETE,
                               'Fr_list': fracture_fronts,
                               'Number_of_fronts': numberof_fronts
                               }
        towrite = {'complete_footrints': complete_footprints}
        append_to_json_file(myJsonName, towrite, 'extend_dictionary')

    # Fig_FP = plot_analytical_solution(regime='K',
    #                                  variable='footprint',
    #                                  mat_prop=Solid,
    #                                  inj_prop=Injection,
    #                                  fig=Fig_FP,
    #                                  time_srs=time_srs)

    # other
    ########## TAKE A HORIZONTAL SECTION TO GET w AT THE MIDDLE ########
    ext_pnts = np.empty((2, 2), dtype=np.float64)
    fracture_list_slice = plot_fracture_list_slice(Fr_list,
                                                   variable='pf',
                                                   projection='2D',
                                                   plot_cell_center=True,
                                                   extreme_points=ext_pnts,
                                                   orientation='horizontal',
                                                   point1=[-0.025, 0.0],
                                                   point2=[0.025, 0.], export2Json=True)
    ########## TAKE A HORIZONTAL SECTION TO GET w AT THE MIDDLE ########
    ext_pnts = np.empty((2, 2), dtype=np.float64)
    fracture_list_slice = plot_fracture_list_slice(Fr_list,
                                                   variable='pn',
                                                   projection='2D',
                                                   plot_cell_center=True,
                                                   extreme_points=ext_pnts,
                                                   orientation='horizontal',
                                                   point1=[-0.025, 0.0],
                                                   point2=[0.025, 0.], export2Json=True)

    print("DONE! in " + myJsonName)
    plt.show(block=True)
