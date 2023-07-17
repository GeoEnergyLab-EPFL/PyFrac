# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on Fri June 16 17:49:21 2017.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2019.
All rights reserved. See the LICENSE.TXT file for more details.
"""

# imports
import os

# local imports
from mesh_obj import CartesianMesh
from properties import MaterialProperties, FluidProperties, InjectionProperties, SimulationProperties
from fracture_obj import Fracture
from controller import Controller
from fracture_initialization import Geometry, InitializationParameters
from utility import setup_logging_to_console
import numpy as np
import math
# setting up the verbosity level of the log at console
setup_logging_to_console(verbosity_level='debug')

myfolder = "./Data/toughness_layer_013"
stringvalue = "_013"
run= True
restart = False
if run == True:

    # creating mesh
    Mesh = CartesianMesh(0.31, 0.31, 181, 181, symmetric=True)

    # solid properties
    nu = 0.4                            # Poisson's ratio
    youngs_mod = 3.3e10                 # Young's modulus
    Eprime = youngs_mod / (1 - nu ** 2) # plain strain modulus


    def My_KIc_func(x, y):
        """ The function providing the fracture toughness"""
        K_Ic1 = 1.5e6  # fracture toughness everywhere
        K_Ic2 = K_Ic1*math.sqrt(7.)/2.  # fracture toughness inside inclusions
        R=0.01
        #and y > -0.30
        if abs(x) <= R and abs(y) >= 0.24:
           return K_Ic2
        else:
           return K_Ic1



    # material properties
    Solid = MaterialProperties(Mesh,
                               Eprime,
                               K1c_func=My_KIc_func,
                               minimum_width=1e-8)

    # injection parameters
    Q0 = 0.001  # injection rate
    Injection = InjectionProperties(Q0, Mesh)

    # fluid properties
    Fluid = FluidProperties(viscosity=0.)

    if not restart:
        # simulation properties
        simulProp = SimulationProperties()
        simulProp.bckColor = 'K1c'
        simulProp.finalTime =   0.01                          # the time at which the simulation stops
        simulProp.set_tipAsymptote('K')                     # tip asymptote is evaluated with the viscosity dominated assumption
        simulProp.frontAdvancing = 'implicit'               # to set explicit front tracking
        simulProp.set_outputFolder(myfolder) # the disk address where the files are saved
        simulProp.plotVar = ['footprint']
        simulProp.saveToDisk=True
        #simulProp.timeStepLimit = 0.00015
        simulProp.limitAdancementTo2cells=True
        # initialization parameters
        Fr_geometry = Geometry('radial', radius=0.25, center=[0,0])
        init_param = InitializationParameters(Fr_geometry, regime='K')
        simulProp.symmetric = True
        simulProp.limitAdancementTo2cells=False
        simulProp.timeStepLimit = None
        # setting up mesh extension options
        simulProp.meshExtensionAllDir = True
        simulProp.plotTSJump=20
        simulProp.set_mesh_extension_factor(1.1)
        simulProp.set_mesh_extension_direction(['all'])
        simulProp.meshReductionPossible = False
        simulProp.set_solTimeSeries(np.arange(0.005, 0.01, 0.00005))
        #simulProp.set_solTimeSeries([0.005639625889244824, 0.005675, 0.005745748221510353, 0.005887244664531058, 0.0060125000000000005, 0.006263010670937886, 0.00635, 0.006523978658124227, 0.006550314147964952, 0.006554119897198941, 0.006558580186132946, 0.0065675007640009575, 0.00658534191973698, 0.006621024231209025, 0.006687500000000001, 0.006820451537581953, 0.007025, 0.0073625, 0.007481982501379024])
        # creating fracture object
        Fr = Fracture(Mesh,
                      init_param,
                      Solid,
                      Fluid,
                      Injection,
                      simulProp)
    else:
        ################################################################################
        # the following lines are needed if you want to restart an existing simulation #
        ################################################################################
        from visualization import *
        Fr_list, properties = load_fractures(address=myfolder, step_size=10)       # load all fractures
        SolidWaste, FluidWaste, InjectionWaste, simulProp = properties
        Fr = Fr_list[-1]
        simulProp.limitAdancementTo2cells=False
        simulProp.timeStepLimit = None
        # setting up mesh extension options
        simulProp.meshExtensionAllDir = True
        simulProp.set_mesh_extension_factor(1.1)
        simulProp.set_mesh_extension_direction(['all'])
        simulProp.meshReductionPossible = False

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

plotfigures= True
if plotfigures:
    from visualization import *

    Fr_list, properties = load_fractures(address=myfolder)
    Solid, Fluid, Injection, simulProp = properties
                                         # ,time_srs=time_srs)
    time_srs = get_fracture_variable(Fr_list,
                                     variable='time')

    # plot footprint

    Fig_FP = plot_fracture_list(Fr_list,
                                variable='mesh',
                                projection='2D',
                                mat_properties=Solid,
                                backGround_param='K1c')
    Fig_FP = plot_fracture_list(Fr_list,
                                variable='footprint',
                                projection='2D',
                                fig=Fig_FP)


    plt.show(block=True)


####################################
# exporting to multiple json files #
####################################

export_results =True
# 1) mandatory
# 2) mandatory
# 3) write to json the coordinates of the points defining the fracture fronts at each time:
# 4) get the fracture opening w(t) versus time (t) at a point of coordinates myX and myY
# 5) get the fluid pressure p(t) versus time (t) at a point of coordinates myX and myY
# 6) get the fluid pressure p(t) versus time (t) at a point of coordinates myX and myY
# 7) get w(y) along a vertical line passing through mypoint for different times
# 8) get pf(x) along a horizontal line passing through mypoint for different times
# 9) get ux(x) along a horizontal line passing through mypoint for different times
# 10) get uy(y) along a vertical line passing through mypoint for different times
# 11) get w(x,y,t) and pf(x,y,t)
# 12) get fields of velocity and fluxes at selected times

to_export = [1,2,3]


if export_results:
    # 0) import some functions needed later
    if 1 in to_export:
        from visualization import *
        from postprocess_fracture import append_to_json_file


    # 1) decide the names of the Json files:
    if 1 in to_export:
        myJsonName_1 = "./Data/toughness_layer_export"+stringvalue+".json"           # I will export here most of the infos
        myJsonName_2 = "./Data/toughness_layer_VEL_as_vector"+stringvalue+".json"       # I will export here the velocity infos



    # 2) load the results:
    #
    # >>> Remember that you can select a subset of time steps <<<<
    # >>> otherwise you will export at all the time steps     <<<<
    #
    if 1 in to_export:
        print("\n 1) loading results")
        Fr_list, properties = load_fractures(address=myfolder) # or load_fractures(address=myfolder,time_srs=np.linspace(5., 8.0,600))
        Solid, Fluid, Injection, simulProp = properties
        print(" <-- DONE\n")


    # *) write to json the general informations
    if 2 in to_export:
        simul_info = {'Eprime': Solid.Eprime,
                      'max_KIc': Solid.K1c.max(),
                      'min_KIc': Solid.K1c.min(),
                      'max_Sigma0': Solid.SigmaO.max(),
                      'min_Sigma0': Solid.SigmaO.min(),
                      'viscosity': Fluid.viscosity,
                      'total_injection_rate': Injection.injectionRate.max(),
                      'sources_coordinates_lastFR': Fr_list[-1].mesh.CenterCoor[Injection.sourceElem].tolist()}
        append_to_json_file(myJsonName_1, simul_info, 'append2keyASnewlist', key='simul_info',delete_existing_filename=True) #be careful: delete_existing_filename=True only the first time you call "append_to_json_file"


    # 3) write to json the coordinates of the points defining the fracture fronts at each time:
    if 3 in to_export:
        print("\n 2) writing fronts")
        time_srs = get_fracture_variable(Fr_list,variable='time') # get the list of times corresponding to each fracture object
        append_to_json_file(myJsonName_1, time_srs, 'append2keyASnewlist', key='time_srs_of_Fr_list')
        fracture_fronts = []
        numberof_fronts = [] #there might be multiple fracture fronts in general
        mesh_info = [] # if you do not make remeshing or mesh extension you can export it only once
        index = 0
        for fracture in Fr_list:
            fracture_fronts.append(np.ndarray.tolist(fracture.Ffront))
            numberof_fronts.append(fracture.number_of_fronts)
            mesh_info.append([Fr_list[index].mesh.Lx, Fr_list[index].mesh.Ly, Fr_list[index].mesh.nx, Fr_list[index].mesh.ny])
            index = index + 1
        append_to_json_file(myJsonName_1, fracture_fronts, 'append2keyASnewlist', key='Fr_list')
        append_to_json_file(myJsonName_1, numberof_fronts, 'append2keyASnewlist', key='Number_of_fronts')
        append_to_json_file(myJsonName_1,mesh_info,'append2keyASnewlist', key='mesh_info')
        print(" <-- DONE\n")



    # 4) get the fracture opening w(t) versus time (t) at a point of coordinates myX and myY
    if 4 in to_export:
        print("\n 3) get w(t) at a point... ")
        my_X = 0.02 ; my_Y = 0.
        w_at_my_point, time_list_at_my_point = get_fracture_variable_at_point(Fr_list, variable='w', point=[my_X, my_Y])
        append_to_json_file(myJsonName_1, w_at_my_point, 'append2keyASnewlist', key='w_at_my_point')
        append_to_json_file(myJsonName_1, time_list_at_my_point, 'append2keyASnewlist', key='time_list_W_at_my_point')
        print(" <-- DONE\n")



    # 5) get the fluid pressure p(t) versus time (t) at a point of coordinates myX and myY
    if 5 in to_export:
        print("\n 4) get pf(t) at a point... ")
        my_X = 0.02 ; my_Y = 0.
        pf_at_my_point, time_list_at_my_point = get_fracture_variable_at_point(Fr_list, variable='pf', point=[my_X, my_Y])
        append_to_json_file(myJsonName_1, pf_at_my_point, 'append2keyASnewlist', key='pf_at_my_point_A')
        append_to_json_file(myJsonName_1, time_list_at_my_point, 'append2keyASnewlist', key='time_list_pf_at_my_point_A')
        print(" <-- DONE\n")


    # 6) get the fluid pressure p(t) versus time (t) at a point of coordinates myX and myY
    if 6 in to_export:
        print("\n 4) get pf(t) at a point... ")
        my_X = 0.00 ; my_Y = 0.00
        pf_at_my_point, time_list_at_my_point = get_fracture_variable_at_point(Fr_list, variable='pf', point=[my_X, my_Y])
        append_to_json_file(myJsonName_1, pf_at_my_point, 'append2keyASnewlist', key='pf_at_my_point_B')
        append_to_json_file(myJsonName_1, time_list_at_my_point, 'append2keyASnewlist', key='time_list_pf_at_my_point_B')
        print(" <-- DONE\n")


    # 7) get w(y) along a vertical line passing through mypoint for different times
    if 7 in to_export:
        print("\n 5) get w(y) with y passing through a specific point for different times... ")
        my_X = 0.; my_Y = 0.
        ext_pnts = np.empty((2, 2), dtype=np.float64)
        fracture_list_slice = plot_fracture_list_slice(Fr_list,
                                                       variable='w',
                                                       projection='2D',
                                                       plot_cell_center=True,
                                                       extreme_points=ext_pnts,
                                                       orientation='horizontal',
                                                       point1=[my_X , my_Y],
                                                       export2Json=True,
                                                       export2Json_assuming_no_remeshing=False)
        towrite = {'w_vert_slice_': fracture_list_slice}
        append_to_json_file(myJsonName_1, towrite, 'extend_dictionary')
        print(" <-- DONE\n")



    # 8) get pf(x) along a horizontal line passing through mypoint for different times
    if 8 in to_export:
        print("\n 6) get pf(x) with x passing through a specific point for different times... ")
        my_X = 0.; my_Y = 0.
        ext_pnts = np.empty((2, 2), dtype=np.float64)
        fracture_list_slice = plot_fracture_list_slice(Fr_list,
                                                       variable='pf',
                                                       projection='2D',
                                                       plot_cell_center=True,
                                                       extreme_points=ext_pnts,
                                                       orientation='horizontal',
                                                       point1=[my_X , my_Y],
                                                       export2Json=True,
                                                       export2Json_assuming_no_remeshing=False)
        towrite = {'pf_horiz_slice_': fracture_list_slice}
        append_to_json_file(myJsonName_1, towrite, 'extend_dictionary')
        print(" <-- DONE\n")



    # 9) get ux(x) along a horizontal line passing through mypoint for different times
    if 9 in to_export:
        print("\n 7) writing ux, qx")
        from postprocess_fracture import get_velocity_slice
        my_X = 0.01; my_Y = 0.
        ux_val, ux_times, ux_coord_x = get_velocity_slice(Solid, Fluid, Fr_list, [my_X, my_Y], orientation='horizontal')
        towrite = {'ux_horizontal_y0_value': ux_val,
                   'ux_horizontal_y0_time': ux_times,
                   'ux_horizontal_y0_coord': ux_coord_x}
        append_to_json_file(myJsonName_1, towrite, 'extend_dictionary')
        print(" <-- DONE\n")



    # 10) get uy(y) along a vertical line passing through mypoint for different times
    if 10 in to_export:
        print("\n 8) writing uy, qy")
        from postprocess_fracture import get_velocity_slice
        my_X = 0.01; my_Y = 0.
        uy_val, uy_times, uy_coord_y = get_velocity_slice(Solid, Fluid, Fr_list, [my_X, my_Y], vel_direction='uy',
                                                          orientation='vertical')
        towrite = {'uy_vertical_x0_value': uy_val,
                   'uy_vertical_x0_time': uy_times,
                   'uy_vertical_x0_coord': uy_coord_y}
        append_to_json_file(myJsonName_1, towrite, 'extend_dictionary')
        print(" <-- DONE\n")



    # 11) get w(x,y,t) and pf(x,y,t)
    if 11 in to_export:
        print("\n 9) get w(x,y,t) and  pf(x,y,t)... ")
        wofxyandt = []
        pofxyandt = []
        info = []
        jump = True #this is used to jump the first fracture
        for frac in Fr_list:
            if not jump:
                wofxyandt.append(np.ndarray.tolist(frac.w))
                pofxyandt.append(np.ndarray.tolist(frac.pFluid))
                info.append([frac.mesh.Lx,frac.mesh.Ly,frac.mesh.nx,frac.mesh.ny,frac.time])
            else:
                jump = False

        append_to_json_file(myJsonName_1, wofxyandt, 'append2keyASnewlist', key='w')
        append_to_json_file(myJsonName_1, pofxyandt, 'append2keyASnewlist', key='p')
        append_to_json_file(myJsonName_1, info, 'append2keyASnewlist', key='info_for_w_and_p')
        print(" <-- DONE\n")



    # 12) get fields of velocity and fluxes at selected times
    if 12 in to_export:
        print("\n 10) process v(x,y), q(x,y)")
            # NOTE: saving only the non-zero-entries
        selected_times = range(len(Fr_list))
        vel_times = []   # list of the selected times
        vel_fields = []  # list with non zero velocity components (for each selected time)
        flux_fields = [] # list with non zero flux components (for each selected time)
        mesh_info = []   # list with the mesh info for each time
        index = 0

        for fracture in Fr_list: # loop on the fractures and take the data only from the selected indexes
            if index != 0 and index in selected_times:
                localVlist = []
                for i in range(fracture.fluidVelocity_components.shape[1]):
                    localElementList = np.ndarray.tolist(fracture.fluidVelocity_components[:, i])
                    if fracture.fluidVelocity_components[:, i].max() != 0 and i in fracture.EltChannel:
                        localElementList.append(i)
                        localVlist.append(localElementList)
                vel_fields.append(localVlist)
                vel_times.append(fracture.time)
                mesh_info.append([Fr_list[index].mesh.Lx, Fr_list[index].mesh.Ly, Fr_list[index].mesh.nx, Fr_list[index].mesh.ny])
                # flux_fields.append(np.ndarray.tolist(fracture.fluidFlux_components))
            index = index + 1

        append_to_json_file(myJsonName_2, vel_fields, 'append2keyASnewlist', key='vel_list', delete_existing_filename=True)
        append_to_json_file(myJsonName_2, vel_times, 'append2keyASnewlist', key='vel_times')
        append_to_json_file(myJsonName_2, mesh_info, 'append2keyASnewlist', key='mesh_info')
        # append_to_json_file(myJsonName_1, flux_fields, 'append2keyASnewlist', key='flux_list')
        print(" <-- DONE\n")

    print("DONE! in " + myJsonName_1)
    print("DONE! in " + myJsonName_2)
    plt.show(block=True)