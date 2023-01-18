# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Carlo Peruzzo 2020.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2019.
All rights reserved. See the LICENSE.TXT file for more details.
"""

# local imports
from mesh_obj import CartesianMesh
from properties import MaterialProperties, FluidProperties, InjectionProperties, SimulationProperties
from fracture_obj import Fracture
from controller import Controller
from fracture_initialization import Geometry, InitializationParameters
import numpy as np

run =True
export_results = False

"_______________________________________________________"
# creating mesh

mesh_discretiz_x=265
mesh_discretiz_y=171
Mesh = CartesianMesh(0.035, 0.015, mesh_discretiz_x, mesh_discretiz_y)

# solid properties
nu = 0.48                            # Poisson's ratio
youngs_mod = 97000                 # Young's modulus (+/- 10) #kPa
Eprime = youngs_mod / (1 - nu ** 2)  # plain strain modulus
K_Ic = np.sqrt(2*5.2*Eprime)         # fracture toughness (+/- 1)
Cl = 0.0                             # Carter's leak off coefficient

Solid = MaterialProperties(Mesh,
                           Eprime,
                           K_Ic,
                           minimum_width=1e-9)
hx=Mesh.hx
hy=Mesh.hy
def source_location(x, y,hx,hy):
    """ This function is used to evaluate if a point is included in source, i.e. the fluid is injected at the given
        point.
    """
    # the condition
    if (abs(x + .02) < Mesh.hx*0.5 and abs(y - 0) < Mesh.hy*0.5 ) or (abs(x - .02) < Mesh.hx*0.5 and abs(y + 0) < Mesh.hy*0.5):
        return True

# injection parameters
Q0 =  5/1000/60/1000 #20mL/min  # injection rate, sum of the injections in the single fracture
initialratesecondpoint=0
ratesecondpoint=np.asarray([[1.09865], [Q0/2]])
delayed_second_injpoint_loc=np.asarray([-0.02,0])
Injection = InjectionProperties(Q0, Mesh, source_loc_func=source_location,
                                initial_rate_delayed_second_injpoint=initialratesecondpoint,
                                rate_delayed_second_injpoint=ratesecondpoint,
                                delayed_second_injpoint_loc=delayed_second_injpoint_loc)

# fluid properties
#Fluid = FluidProperties(viscosity=0.01)
Fluid = FluidProperties(viscosity=0.)




# simulation properties
simulProp = SimulationProperties()
simulProp.finalTime = 23.0               # the time at which the simulation stops
#simulProp.timeStepLimit = 0.01

myfolder ="./Data/coal_VC_exp_pivdb13_"+str(mesh_discretiz_x)+"x"+str(mesh_discretiz_y)
simulProp.set_outputFolder(myfolder)          # the address of the output folder

simulProp.saveToDisk=True
#simulProp.plotVar = ['pf','w']
fixedtimesteps = np.array([[0.69,1.1,2.7,3.7,5,22],[0.0005,0.005,0.01,0.05,0.2,0.08]])
simulProp.fixedTmStp = fixedtimesteps
simulProp.projMethod = 'LS_continousfront'
simulProp.set_tipAsymptote('K')         # the tip asymptote is evaluated with the toughness dominated assumption
simulProp.set_volumeControl(True)       # use the inviscid fluid solver(toughness dominated), imposing volume balance
simulProp.doublefracture = True
simulProp.plotTSJump = 40
simulProp.saveTSJump=5
simulProp.maxFrontItrs=35
simulProp.meshExtensionFactor = 1.1
simulProp.set_mesh_extension_direction(['all'])
simulProp.useBlockToeplizCompression = True
simulProp.saveToDisk=False

if run:
    # initializing fracture
    from fracture_initialization import get_radial_survey_cells

    initRad1 = 0.000802
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

    from elasticity import load_isotropic_elasticity_matrix

    C = load_isotropic_elasticity_matrix(Mesh, Eprime)

    # Fr_geometry = Geometry('radial', radius=0.15)
    init_param = InitializationParameters(Fr_geometry,
                                          time=0.698806,
                                          regime='static',
                                          net_pressure=36.2e3,
                                          elasticity_matrix=C)

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
write = True
plot = False
slicing = False
if slicing: time_slicing=np.linspace(0., 23.,50000)
myJsonName = "./Data/coalescence_expivpdb13_4mma_VC.json"
myfolder ="./Data/coal_VC_exp_pivdb13_"+str(mesh_discretiz_x)+"x"+str(mesh_discretiz_y)
from postprocess_fracture import append_to_json_file

if export_results:
    from visualization import *


    # loading simulation results
    if slicing:
        Fr_list, properties = load_fractures(address=myfolder,
                                             time_srs=time_slicing)       # load all fractures
    else: Fr_list, properties = load_fractures(address=myfolder)

    time_srs = get_fracture_variable(Fr_list,                                       # list of times
                                     variable='time')

    if write:
        append_to_json_file(myJsonName, time_srs, 'append2keyASnewlist', key='time_srs_of_Fr_list',delete_existing_filename=True)


        fracture_fronts=[]
        numberof_fronts=[]
        for fracture in Fr_list:
            fracture_fronts.append(np.ndarray.tolist(fracture.Ffront))
            numberof_fronts.append(fracture.number_of_fronts)
        append_to_json_file(myJsonName, fracture_fronts, 'append2keyASnewlist', key='Fr_list')
        append_to_json_file(myJsonName, numberof_fronts, 'append2keyASnewlist', key='Number_of_fronts')
        append_to_json_file(myJsonName, [Fr_list[-1].mesh.Lx, Fr_list[-1].mesh.Ly, Fr_list[-1].mesh.nx, Fr_list[-1].mesh.ny], 'append2keyASnewlist', key='mesh_info')

    if plot:
        plot_prop = PlotProperties()

        # plot fracture radius
        plot_prop.lineStyle = '.'
        #plot_prop.graphScaling = 'loglog'
        Fig_R = plot_fracture_list(Fr_list,
                                   variable='footprint',
                                   plot_prop=plot_prop)

        Fig_R = plot_fracture_list(Fr_list,
                                   fig=Fig_R,
                                   variable='mesh',
                                   mat_properties=properties[0],
                                   plot_prop=plot_prop)

    if plot:
        ########## TAKE A VERTICAL SECTION TO GET w AT THE MIDDLE ########
        # plot fracture radius
        plot_prop = PlotProperties()

        Fig_R = plot_fracture_list(Fr_list,
                                   variable='footprint',
                                   plot_prop=plot_prop)
        Fig_R = plot_fracture_list(Fr_list,
                                   fig=Fig_R,
                                   variable='mesh',
                                   mat_properties=properties[0],
                                   backGround_param='K1c',
                                   plot_prop=plot_prop)

    # ++++
    # Fr_list, properties = load_fractures(address=myfolder ,time_srs=np.linspace(5., 8.0,600))
    if slicing:
        Fr_list, properties = load_fractures(address=myfolder,
                                             time_srs=time_slicing)       # load all fractures
    else: Fr_list, properties = load_fractures(address=myfolder)

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
    if slicing:
        Fr_list_COMPLETE, properties = load_fractures(address=myfolder,
                                             time_srs=time_slicing)       # load all fractures
    else: Fr_list_COMPLETE, properties = load_fractures(address=myfolder)

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
    if plot:
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
    if write: print("DONE! in "+myJsonName)
    plt.show(block=True)