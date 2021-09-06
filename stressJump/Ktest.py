# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on Fri March 13 2018.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2019.
All rights reserved. See the LICENSE.TXT file for more details.
"""

from utility import setup_logging_to_console# setting up the verbosity level of the log at console
setup_logging_to_console(verbosity_level='debug')

import numpy as np
import matplotlib.pyplot as plt
# local imports
from mesh import CartesianMesh
from properties import MaterialProperties, FluidProperties, InjectionProperties, SimulationProperties
from fracture import Fracture
from controller import Controller
from fracture_initialization import Geometry, InitializationParameters

def getStressVertSlice(mesh, solid):
    nei=mesh.NeiElements
    el = 0
    go = True
    el_list = [el]
    # find slice cells
    while go:
        left, right, bottom, top = nei[el]
        if top != el:
            el_list.append(top)
            el = top
        else:
            go = False

    # find sigma values
    mySigma = solid.SigmaO[el_list]
    myYcoords = mesh.CenterCoor[el_list , 1]

    # plot the pressure vs time
    xlabel = 'y [m]'
    ylabel = 'sigma0 [MPa]'
    fig, ax = plt.subplots()
    ax.scatter(myYcoords, mySigma, color='k')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.show()

    return None

def getFrBounds(Fr):
    ymax = 0.
    ymin = 0.

    for segment in Fr.Ffront:
        x1, y1, x2, y2 = segment
        if y1 > ymax:
            ymax = y1
        if y1 < ymin:
            ymin = y1
        if y2 > ymax:
            ymax = y2
        if y2 < ymin:
            ymin = y2
    return ymax, ymin

#################################

run = True
#run = False

plot = True
export_results = False

simulation_name = 'Ktest'
myfolder = './Data/'+simulation_name

if run:
    # creating mesh
    Mesh = CartesianMesh(102, 102, 101, 101)

    # solid properties
    Eprime = 2.6042e10
    K1C = 3e7
    Cl = 0

    def sigmaO_func(x, y):
        """ The function providing the confining stress"""
        if y > 33:
            return 5.45e6
        elif y < -33:
            return 5.45e6
        else:
            return 2.2e6

    Solid = MaterialProperties(Mesh,
                               Eprime,
                               K1C,
                               Cl,
                               confining_stress_func=sigmaO_func,
                               minimum_width=1.e-8)

    #getStressVertSlice(Mesh, Solid)

    # injection parameters
    Q0 = 5.e-2
    Injection = InjectionProperties(Q0, Mesh)

    # fluid properties
    Fluid = FluidProperties(viscosity=0.0000001)

    # simulation properties
    simulProp = SimulationProperties()
    simulProp.bckColor = 'confining stress'           # the parameter according to which the background is color coded
    simulProp.frontAdvancing = 'implicit'
    simulProp.set_outputFolder(myfolder)
    #simulProp.set_solTimeSeries(np.asarray([22., 60., 144., 376., 665.]))
    #fixeddeltat=np.array([[3400],[50]])
    #simulProp.fixedTmStp=fixeddeltat
    simulProp.finalTime = 200000  # the time at which the simulation stops
    simulProp.projMethod = 'LS_continousfront'
    simulProp.plotTSJump = 1
    simulProp.plotVar = ['footprint', 'custom']
    #simulProp.plotVar = ['w', 'regime']
    simulProp.enableGPU = True
    simulProp.meshExtension = False
    simulProp.enableRemeshing = False
    simulProp.useBlockToeplizCompression = True  # this is to use less memory on your laptop

    ## for custom use ##
    simulProp.customPlotsOnTheFly = True
    simulProp.LHyst__ = []
    simulProp.tHyst__ = []

    # initializing fracture
    Fr_geometry = Geometry('radial', radius=23)
    init_param = InitializationParameters(Fr_geometry, regime='K')

    # creating fracture object
    Fr = Fracture(Mesh,
                  init_param,
                  Solid,
                  Fluid,
                  Injection,
                  simulProp)

    ymax, ymin = getFrBounds(Fr)
    print('crack bounds: '+str(ymax)+' '+str(ymin))

    # reload
    # from visualization import *
    # Fr_list, properties = load_fractures(address=myfolder, step_size=10)       # load all fractures                                                # list of times
    # Solid, Fluid, Injection, simulProp = properties
    # Fr = Fr_list[-1]

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

if plot:
    from visualization import *

    # loading simulation results
    Fr_list, properties = load_fractures(address=myfolder,step_size=10,load_all=True)       # load all fractures
    time_srs = get_fracture_variable(Fr_list, variable='time')                                                  # list of times
    Solid, Fluid, Injection, simulProp = properties

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

    plt.show(block=True)


####################################
# exporting to multiple json files #
####################################


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
        myJsonName_1 = "./Data/TJ_"+simulation_name+"_export.json"           # I will export here most of the infos
        myJsonName_2 = "./Data/TJ_VEL_as_vector"+simulation_name+"_export.json"        # I will export here the velocity infos




    # 2) load the results:
    #
    # >>> Remember that you can select a subset of time steps <<<<
    # >>> otherwise you will export at all the time steps     <<<<
    #
    if 1 in to_export:
        print("\n 1) loading results")
        Fr_list, properties = load_fractures(address=myfolder,load_all=True) # or load_fractures(address=myfolder,time_srs=np.linspace(5., 8.0,600))
        Solid, Fluid, Injection, simulProp = properties
        print(" <-- DONE\n")

    # *) write to json the general informations
    if 2 in to_export:
        time_srs = get_fracture_variable(Fr_list, variable='time')
        time_srs = np.asarray(time_srs)
        timetouch = None
        curvature_of_t = []
        for fr in Fr_list:
            # ---- 1 ----
            # get ribbon element
            EltRibbon = fr.EltRibbon
            # get center coord at ribbon
            EltRibbonCoor = fr.mesh.CenterCoor[EltRibbon]
            EltRibbon_Kvalues = []
            # get KIc at ribbon
            for cell in EltRibbonCoor:
                EltRibbon_Kvalues.append(My_KIc_func(cell[0], cell[1]))
            # if in all the ribbon KIc is constant we did not touch the heterogeneity
            EltRibbon_Kvalues = np.asarray(EltRibbon_Kvalues)
            if EltRibbon_Kvalues.max() != EltRibbon_Kvalues.min() and timetouch is None:
                timetouch = fr.time

            # ---- 2 ----
            # compute the max curvature in the front for the give time step
            curvature = []
            for idx in range(fr.Ffront.shape[0]):
                x0 = fr.Ffront[idx, 2]
                x1 = fr.Ffront[idx, 0]
                x2 = fr.Ffront[(idx + 1) % fr.Ffront.shape[0], 2]
                y0 = fr.Ffront[idx, 3]
                y1 = fr.Ffront[idx, 1]
                y2 = fr.Ffront[(idx + 1) % fr.Ffront.shape[0], 3]
                d = np.sqrt((-x1 + x2) ** 2 + (-y1 + y2) ** 2)
                h = pointtolinedistance(x0, x1, x2, y0, y1, y2)
                curvature.append(h / d)
            curvature = np.asarray(curvature).max()
            curvature_of_t.append(curvature)
        curvature_of_t = np.asarray(curvature_of_t)
        maxCurvPos = np.where(curvature_of_t == curvature_of_t.max())[0]
        timeMaxcurv = time_srs[maxCurvPos.min()]

        simul_info = {'Eprime': Solid.Eprime,
                      'max_KIc': Solid.K1c.max(),
                      'min_KIc': Solid.K1c.min(),
                      'max_Sigma0': Solid.SigmaO.max(),
                      'min_Sigma0': Solid.SigmaO.min(),
                      'viscosity': Fluid.viscosity,
                      'total_injection_rate': Injection.injectionRate.max(),
                      'sources_coordinates_lastFR': Fr_list[-1].mesh.CenterCoor[Injection.sourceElem].tolist(),
                      't_max': time_srs.max(),
                      't_min': time_srs.min(),
                      't_touching_interface': timetouch,
                      'max_curvature_h/d': curvature_of_t.max(),
                      'max_curvature_time': timeMaxcurv}
        append_to_json_file(myJsonName_1, simul_info, 'append2keyASnewlist', key='simul_info',
                            delete_existing_filename=True)  # be careful: delete_existing_filename=True only the first time you call "append_to_json_file"

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

