# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on Fri June 16 17:49:21 2017.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2019. All rights reserved.
See the LICENSE.TXT file for more details.
"""

import numpy as np

# local imports
from mesh_obj import CartesianMesh
from properties import MaterialProperties, FluidProperties, InjectionProperties, SimulationProperties
from fracture_obj import Fracture
from controller import Controller
from fracture_initialization import Geometry, InitializationParameters
from utility import setup_logging_to_console

# setting up the verbosity level of the log at console
setup_logging_to_console(verbosity_level='debug')
run = True
export_results = False

"_______________________________________________________"
# creating mesh
#mesh_discretiz_x=237
#mesh_discretiz_y=129
mesh_discretiz_x=137
mesh_discretiz_y=89
#Mesh = CartesianMesh(0.039, 0.019, mesh_discretiz_x, mesh_discretiz_y)
Mesh = CartesianMesh(0.037, 0.013, mesh_discretiz_x, mesh_discretiz_y)
# solid properties
nu = 0.48                            # Poisson's ratio
youngs_mod = 97000                   # Young's modulus (+/- 10) #kPa
Eprime = youngs_mod / (1 - nu ** 2)  # plain strain modulus
K_Ic = np.sqrt(2*5.2*Eprime)         # fracture toughness (+/- 1)
Cl = 0.0                             # Carter's leak off coefficient


Solid = MaterialProperties(Mesh,
                           Eprime,
                           K_Ic,
                           minimum_width=1e-12,
                           boundary_eff=True,
                           PoissonRatio=0.48
                           )

sizehx =  Mesh.hx
sizehy =  Mesh.hy

def source_location(x, y, sizehx, sizehy):
    """ This function is used to evaluate if a point is included in source, i.e. the fluid is injected at the given
        point.
    """
    Rinj = 0.001
    if (abs(x + .022115) < sizehx * 0.5 + Rinj and abs(y - 0) < sizehy * 0.5  + Rinj ) or (abs(x - .022115) < sizehx*0.5 + Rinj and abs(y + 0) < sizehy*.5 + Rinj):
        return True

def delayed_second_injpoint_loc_func(x, y, sizehx, sizehy):
    Rinj = 0.001
    if (abs(x + .022115) < sizehx * 0.5 + Rinj and abs(y - 0) < sizehy * 0.5  + Rinj ):
        return True

# injection parameters
Q0 = 10/1000/60/1000 #20mL/min  # injection rate
initialratesecondpoint=0.
ratesecondpoint=np.asarray([[1.279], [Q0/2]])
#ratesecondpoint=np.asarray([[1.09865], [Q0/2]]) #short
#delayed_second_injpoint_loc=np.asarray([-0.02,0])
Injection = InjectionProperties(Q0, Mesh, source_loc_func = source_location,
                                initial_rate_delayed_second_injpoint = initialratesecondpoint,
                                rate_delayed_second_injpoint = ratesecondpoint,
                                delayed_second_injpoint_loc_func = delayed_second_injpoint_loc_func)

# fluid properties
viscosity = 0.44 #Pa.s
Fluid = FluidProperties(viscosity=viscosity)

# simulation properties
simulProp = SimulationProperties()
simulProp.finalTime = 100.0                     # the time at which the simulation stops
myfolder ="./Data/coalescence_exp_pivdb13_"+str(mesh_discretiz_x)+"x"+str(mesh_discretiz_y)+"_viscous_distance44p23mm_test2021"
simulProp.set_outputFolder(myfolder)     # the address of the output folder
#simulProp.set_solTimeSeries(np.asarray([6.0,6.4, 6.45, 6.5, 6.55, 6.6, 6.62, 6.8]))
                                                                                  #2.833, 4.88,150
simulProp.set_solTimeSeries(np.concatenate((np.linspace(0.279, 2.00,100),np.linspace(2, 18.00,100),np.linspace(18., 20.0,400),np.linspace(20., 40.0,100))))
#simulProp.set_solTimeSeries(np.concatenate((np.linspace(0.69, 1.08,100),np.linspace(1.0833, 2.0833,200),np.linspace(2.0833, 3.88,156),np.linspace(18.0, 21.0,40))))
simulProp.plotTSJump = 10
simulProp.timeStepLimit=1
simulProp.saveToDisk=False
simulProp.saveFluidFluxAsVector=True
simulProp.saveFluidVelAsVector=True
#simulProp.plotVar = ['pf', 'ffvf']
simulProp.plotVar = ['footprint']
simulProp.projMethod = 'LS_continousfront'
simulProp.frontAdvancing = 'implicit'
simulProp.useBlockToeplizCompression = True
simulProp.maxFrontItrs=35
simulProp.maxSolverItrs=240
simulProp.plotATsolTimeSeries=False


# setting up mesh extension options
simulProp.meshExtensionAllDir = True
simulProp.maxElementIn = 4000
simulProp.set_mesh_extension_factor(1.4)
simulProp.set_mesh_extension_direction(['all'])
simulProp.meshReductionPossible = False
simulProp.maxCellSize = 0.2

if run:
    # initializing fracture
    p=np.zeros((Mesh.NumberOfElts,), dtype=np.float64)
    from fracture_initialization import get_radial_survey_cells
    initRad2 = 0.00186
    initRad1 = 0.001622
    p1 = 25546.7
    p2 = 23861.8
    surv_cells_1, surv_cells_dist_1, inner_cells_1 = get_radial_survey_cells(Mesh, initRad1, center=[-0.022115, 0])
    Fr_geometry = Geometry(shape='level set',
                           survey_cells=surv_cells_1,
                           tip_distances=surv_cells_dist_1,
                           inner_cells=inner_cells_1)
    from fracture_initialization import generate_footprint
    EltCrack = generate_footprint(Mesh,
                                  surv_cells_1,
                                  inner_cells_1,
                                  surv_cells_dist_1,
                                  simulProp.projMethod)[2] #return the third result
    p[EltCrack] = np.full((EltCrack.size,), p1, dtype=np.float64)

    surv_cells_2, surv_cells_dist_2, inner_cells_2 = get_radial_survey_cells(Mesh, initRad2, center=[0.022115, 0])
    Fr_geometry = Geometry(shape='level set',
                           survey_cells=surv_cells_2,
                           tip_distances=surv_cells_dist_2,
                           inner_cells=inner_cells_2)
    from fracture_initialization import generate_footprint
    EltCrack = generate_footprint(Mesh,
                                  surv_cells_2,
                                  inner_cells_2,
                                  surv_cells_dist_2,
                                  simulProp.projMethod)[2] #return the third result
    p[EltCrack] = np.full((EltCrack.size,), p2, dtype=np.float64)

    surv_cells = np.concatenate((surv_cells_1, surv_cells_2))
    surv_cells_dist = np.concatenate((surv_cells_dist_1, surv_cells_dist_2))
    inner_cells = np.concatenate((inner_cells_1, inner_cells_2))

    Fr_geometry = Geometry(shape='level set',
                           survey_cells=surv_cells,
                           tip_distances=surv_cells_dist,
                           inner_cells=inner_cells)

    from elasticity import load_isotropic_elasticity_matrix
    C = load_isotropic_elasticity_matrix(Mesh, Eprime)


    init_param = InitializationParameters(Fr_geometry,
                                          regime='static',
                                          net_pressure=p,
                                          elasticity_matrix=C,
                                          time=0.279) #is the initiation time of the first fracture


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
write = False
if export_results   :
    myJsonName="./Data/coalescence_expPivdb13_4mma_delay_q_visc_NEWdist.json"
    myJsonNameVel = "./Data/coalescence_expPivdb13_4mma_delay_q_visc_NEWdist_VEL.json"
    myfolder='./Data/coalescence_exp_pivdb13_237x129_viscous_distance44p23mm'
    from visualization import *
    from postprocess_fracture import append_to_json_file

    print("\n1) loading results")
    # Fr_list, properties = load_fractures(address=myfolder ,time_srs=np.linspace(5., 8.0,600))
    Fr_list, properties = load_fractures(address=myfolder)
    Solid, Fluid, Injection, simulProp = properties
    time_srs = get_fracture_variable(Fr_list,variable='time')
    print(" <-- DONE\n")

    print("\n2) writing fronts")
    if write:
        append_to_json_file(myJsonName, time_srs,
                            'append2keyASnewlist',
                            key='time_srs_of_Fr_list',
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
    print(" <-- DONE\n")

    print("\n3) writing ux, qx")
    ########## TAKE A HORIZONTAL SECTION TO GET ux AT THE MIDDLE ########
    from postprocess_fracture_CP import get_velocity_slice
    ux_val, ux_times, ux_coord_x = get_velocity_slice(Solid, Fluid, Fr_list, [-0.03, 0.0], orientation='horizontal')
    if write:
        towrite = {'ux_horizontal_y0_value': ux_val,
                   'ux_horizontal_y0_time': ux_times,
                   'ux_horizontal_y0_coord': ux_coord_x}
        append_to_json_file(myJsonName, towrite, 'extend_dictionary')
    print(" <-- DONE\n")

    ########## TAKE A VERTICAL SECTION TO GET uy AT THE MIDDLE ########
    print("\n4) writing uy, qy")
    from postprocess_fracture_CP import get_velocity_slice
    uy_val, uy_times, uy_coord_y = get_velocity_slice(Solid, Fluid, Fr_list, [0.0, -0.015], vel_direction = 'uy', orientation='vertical')
    if write:
        towrite = {'uy_vertical_x0_value': uy_val,
                   'uy_vertical_x0_time': uy_times,
                   'uy_vertical_x0_coord': uy_coord_y}
        append_to_json_file(myJsonName, towrite, 'extend_dictionary')
    print(" <-- DONE\n")

    # get w(x,y,t) and pf(x,y,t)
    print("\n 5) get w(x,y,t) and  pf(x,y,t)... ")
    wofxyandt = []
    pofxyandt = []
    jump= True
    for frac in Fr_list:
        if not jump:
            wofxyandt.append(np.ndarray.tolist(frac.w))
            pofxyandt.append(np.ndarray.tolist(frac.pFluid))
        else:
            jump = False

    append_to_json_file(myJsonName, wofxyandt, 'append2keyASnewlist', key='w')
    append_to_json_file(myJsonName, pofxyandt, 'append2keyASnewlist', key='p')
    print(" <-- DONE\n")

    ########## GET THE VELOCITY, AND FLUX FIELDS ########
    print("\n6) process u(x,y), q(x,y)")
    vel_fields = []
    flux_fields = []
    xypoints = []
    indexremesh = []
    index = 0

    selected_times = [177, 184, 192, 200, 208, 216, 224,
                      253, 259, 265, 269, 275, 281, 285,
                      291, 295, 299, 303, 305, 309, 313, 317, 321]
    for fracture in Fr_list:
        index = index +1
        print("\nFR"+str(index))
        if index != 1 and index in selected_times:
            localVlist=[]
            for i in range(fracture.fluidVelocity_components.shape[1]):
                localElementList = np.ndarray.tolist(fracture.fluidVelocity_components[:,i])
                if fracture.fluidVelocity_components[:,i].max() != 0 and i in fracture.EltChannel:
                    localElementList.append(i)
                    localElementList.append(fracture.time)
                    localVlist.append(localElementList)
            vel_fields.append(localVlist)
            #flux_fields.append(np.ndarray.tolist(fracture.fluidFlux_components))
        elif index ==1 :
            xypoints.append(np.ndarray.tolist(fracture.mesh.CenterCoor))
            sizehx = fracture.mesh.hx
        if fracture.mesh.hx != sizehx:
            xypoints.append(np.ndarray.tolist(fracture.mesh.CenterCoor))
            sizehx = fracture.mesh.hx
            indexremesh.append(index)

    print(" <-- DONE\n")

    print(" Write element neighbors \n")
    append_to_json_file(myJsonName, np.ndarray.tolist(Fr_list[1].mesh.NeiElements), 'append2keyASnewlist', key='element_neighbors')

    print(" Write indexes \n")
    append_to_json_file(myJsonName, indexremesh, 'append2keyASnewlist', key='index_remesh')

    print(" Write xy points \n")
    append_to_json_file(myJsonName, xypoints, 'append2keyASnewlist', key='xypoints')

    append_to_json_file(myJsonName,
                        [Fr_list[1].mesh.Lx, Fr_list[1].mesh.Ly, Fr_list[1].mesh.nx, Fr_list[1].mesh.ny],
                        'append2keyASnewlist', key='first_mesh_info')

    print(" Write vel \n")
    append_to_json_file(myJsonNameVel, vel_fields, 'append2keyASnewlist', key='vel_list',
                            delete_existing_filename=True)
    #append_to_json_file(myJsonName, flux_fields, 'append2keyASnewlist', key='vel_list')
    print(" <-- DONE\n")
    n=1
    for i in indexremesh:
        n=n+1
        append_to_json_file(myJsonName,
                            [Fr_list[i].mesh.Lx, Fr_list[i].mesh.Ly, Fr_list[i].mesh.nx, Fr_list[i].mesh.ny],
                            'append2keyASnewlist', key='first_mesh_info_n_'+str(n))

    append_to_json_file(myJsonName,
                        [Fr_list[1].mesh.Lx, Fr_list[1].mesh.Ly, Fr_list[1].mesh.nx, Fr_list[1].mesh.ny],
                        'append2keyASnewlist', key='mesh_info')



    # ########## IMPORT THE COMPLETE SERIES OF FRACTURE FOOTPRINTS ########
    # Fr_list_COMPLETE, properties = load_fractures(address=myfolder)
    # Solid, Fluid, Injection, simulProp = properties
    # time_srs_COMPLETE = get_fracture_variable(Fr_list_COMPLETE,
    #                                           variable='time')
    #
    # if write:
    #     fracture_fronts = []
    #     numberof_fronts = []
    #     for fracture in Fr_list_COMPLETE:
    #         fracture_fronts.append(np.ndarray.tolist(fracture.Ffront))
    #         numberof_fronts.append(fracture.number_of_fronts)
    #
    #     complete_footprints = {'time_srs_of_Fr_list': time_srs_COMPLETE,
    #                            'Fr_list': fracture_fronts,
    #                            'Number_of_fronts': numberof_fronts
    #                            }
    #     towrite = {'complete_footrints': complete_footprints}
    #     append_to_json_file(myJsonName, towrite, 'extend_dictionary')


    print("DONE! in " + myJsonName)
    plt.show(block=True)