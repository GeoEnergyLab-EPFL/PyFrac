# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on Fri March 13 2018.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2019.
All rights reserved. See the LICENSE.TXT file for more details.
"""
from src.level_set.continuous_front_reconstruction import pointtolinedistance
from src.utilities.utility import setup_logging_to_console# setting up the verbosity level of the log at console

from src.solid.elasticity_isotropic import load_isotropic_elasticity_matrix_toepliz

setup_logging_to_console(verbosity_level='debug')

import numpy as np
import matplotlib.pyplot as plt

# local imports
from src.mesh_obj.mesh import CartesianMesh
from src.properties import InjectionProperties, SimulationProperties
from src.solid.solid_prop import MaterialProperties
from src.fluid.fluid_prop import FluidProperties
from src.fracture_obj.fracture import Fracture
from src.controller import Controller
from src.fracture_obj.fracture_initialization import Geometry, InitializationParameters
from src.utilities.postprocess_fracture import load_fractures
import src.utilities.postprocess_fracture

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


class stress_func_factory:
    def __init__(self, sigmaHigh, sigmaLow, ymax_layer):
        self.sigmaHigh = sigmaHigh
        self.sigmaLow = sigmaLow
        self.ymax_layer = ymax_layer


    def __call__(self, x, y):
            """ The function providing the confining stress"""
            if np.abs(y) > self.ymax_layer:
                return self.sigmaHigh  # 2.20005e6
            else:
                return self.sigmaLow


# --------------------------------------------------------------
def get_fracture_sizes(Fr):
    # Now we are at a given time step.
    # This function returns the coordinates of the smallest rectangle containing the fracture footprint

    x_min_temp = 0.
    x_max_temp = 0.
    y_min_temp = 0.
    y_max_temp = 0.

    # loop over the segments defining the fracture front
    for i in range(Fr.Ffront.shape[0]):
        segment = Fr.Ffront[i]

        # to find the x_max at this segment:
        if segment[0] > x_max_temp:
            x_max_temp = segment[0]
        if segment[2] > x_max_temp:
            x_max_temp = segment[2]

        # to find the n_min at this segment:
        if segment[0] < x_min_temp:
            x_min_temp = segment[0]
        if segment[2] < x_min_temp:
            x_min_temp = segment[2]

        # to find the y_max at this segment:
        if segment[1] > y_max_temp:
            y_max_temp = segment[1]
        if segment[3] > y_max_temp:
            y_max_temp = segment[3]

        # to find the y_min at this segment:
        if segment[1] < y_min_temp:
            y_min_temp = segment[1]
        if segment[3] < y_min_temp:
            y_min_temp = segment[3]

    return x_min_temp, x_max_temp, y_min_temp, y_max_temp


class custom_factory():
    def __init__(self, r_0, xlabel, ylabel):
        self.data = {'xlabel' : xlabel,
                     'ylabel': ylabel,
                     'xdata': [],
                     'ydata': [],
                     'H/2': r_0} # max value of x that can be reached during the simulation

    def custom_plot(self, sim_prop, fig=None):
        # this method is mandatory
        if fig is None:
            fig = plt.figure()
            ax = fig.gca()
        else:
            ax = fig.get_axes()[0]

        ax.scatter(self.data['xdata'], self.data['ydata'], color='k')
        ax.set_xlabel(self.data['xlabel'])
        ax.set_ylabel(self.data['ylabel'])
        ax.set_yscale('log')
        ax.set_xscale('log')
        return fig

    def postprocess_fracture(self, sim_prop, fr):
        # this method is mandatory
        x_min_n, x_max_n, y_min_n, y_max_n = get_fracture_sizes(fr)
        self.data['xdata'].append(fr.time)
        self.data['ydata'].append(y_max_n / x_max_n)
        fr.postprocess_info = self.data
        return fr



class  my_time_step_prefactor():
    def __init__(self, N, H):
        self.N = N
        self.H = H


    def __call__(self, t, fracture_obj, estimated_ts):
            """ The function providing the prefactor"""
            min_ts_indx = np.where(estimated_ts==np.min(estimated_ts))[0]
            if min_ts_indx == 0:
                # find the highest point in y direction
                a, b, c, yup = get_fracture_sizes( fracture_obj)

                # find fracture height in y direction above the reservoir
                yup_rel = (yup - self.H/2.)
                nofCellsAboveRes = yup_rel /fracture_obj.mesh.hy

                # find prefactor
                prefactor = np.maximum(nofCellsAboveRes/self.N, 1)

                return prefactor
            else:
                return 1

#################################

run = True
reload_from_radial = False
reload_from_other = True
plot = False
export_results = False

simulation_name = 'C05_hmat'
myfolder = './Data/'+simulation_name

if run:
    # creating mesh
    #Mesh = CartesianMesh(182, 182, 91, 91)
    Mesh = CartesianMesh(131000, 131000, 131, 131)

    nu = 0.4  # Poisson's ratio
    youngs_mod = 3.3e10  # Young's modulus
    Eprime = youngs_mod / (1 - nu ** 2)  # plain strain modulus
    K1C = 5.e5  # Fracture toughness
    Cl = 0.

    def sigmaO_func(x, y):
        """ The function providing the confining stress"""
        if np.abs(y) > 27000.:
            return 2.201e6 #2.20005e6
        else:
            return 2.2e6

    sigmaHigh = 2.201e6

    Solid = MaterialProperties(Mesh,
                               Eprime,
                               K1C,
                               Cl,
                               confining_stress_func=sigmaO_func,
                               minimum_width=1.e-8)

    #getStressVertSlice(Mesh, Solid)

    # injection parameters
    Q0 = 0.0001  # injection rate
    Injection = InjectionProperties(Q0, Mesh)

    # fluid properties
    viscosity = 0.001  # mu' =0.001
    Fluid = FluidProperties(viscosity=viscosity)
    from src.utilities.postprocess_fracture import load_fractures
    # simulation properties
    simulProp = SimulationProperties()
    simulProp.bckColor = 'confining stress'           # the parameter according to which the background is color coded
    simulProp.frontAdvancing = 'implicit'
    simulProp.set_outputFolder(myfolder)
    simulProp.EHL_iter_lin_solve = True
    simulProp.maxFrontItrs = 150
    #simulProp.set_solTimeSeries(np.asarray([22., 60., 144., 376., 665.]))
    #fixeddeltat=np.array([[3400],[50]])
    #simulProp.fixedTmStp=fixeddeltat
    simulProp.finalTime = 1e15  # the time at which the simulation stops
    simulProp.projMethod = 'LS_continousfront'
    simulProp.plotTSJump = 10
    simulProp.customPlotsOnTheFly = True
    simulProp.LHyst__ = []
    simulProp.tHyst__ = []
    simulProp.custom = custom_factory(27000., 'y/(0.5 H)', 'x/(0.5 H)')
    simulProp.plotVar = ['footprint','regime','custom']
    simulProp.tolFractFront = 0.0001
    simulProp.EHL_iter_lin_solve = True
    simulProp.gmres_Restart = 1000
    simulProp.gmres_maxiter = 1000
    simulProp.frontAdvancing = 'implicit'

    # setting up mesh extension options
    simulProp.meshExtension = True
    simulProp.meshExtensionAllDir = True
    simulProp.set_mesh_extension_direction(['all'])
    simulProp.set_mesh_extension_factor(1.25)
    simulProp.maxFrontItrs = 50

    # initializing fracture
    Fr_geometry = Geometry('radial', radius=25000.)
    init_param = InitializationParameters(Fr_geometry, regime='K')

    # # creating fracture object
    Fr = Fracture(Mesh,
                   init_param,
                   Solid,
                   Fluid,
                   Injection,
                   simulProp)

    ymax, ymin = getFrBounds(Fr)
    print('crack bounds: '+str(ymax)+' '+str(ymin))

    #reload
    if reload_from_radial and not reload_from_other:
        from src.utilities.visualization import *
        Fr_list, properties = load_fractures(address=myfolder, step_size=1)       # load all fractures                                                # list of times
        Solid, Fluid, Injection, simulProp = properties


        # setting up mesh extension options
        simulProp.meshExtension = True
        simulProp.meshExtensionAllDir = True
        simulProp.set_mesh_extension_direction(['all'])
        simulProp.set_mesh_extension_factor(1.5)
        simulProp.frontAdvancing = 'implicit'
        simulProp.maxFrontItrs = 50


        Fr = Fr_list[-1]
        Fr_old = copy.deepcopy(Fr_list[-1])
        # find limits
        x_min_temp, x_max_temp, y_min_temp, y_max_temp = get_fracture_sizes(Fr)

        # find cells names where ymax and ymin are
        ID_ymax = Fr.mesh.locate_element(0., y_max_temp)
        ID_ymin = Fr.mesh.locate_element(0., y_min_temp)

        # find cells IDs above the cell where we have ymax and below the one where we have ymin
        ID_ymax_top = Fr.mesh.NeiElements[ID_ymax, 3]
        ID_ymin_bottom = Fr.mesh.NeiElements[ID_ymin, 2]
        yup = Fr.mesh.CenterCoor[ID_ymax_top[0]][1] - Fr.mesh.hy/2
        ybottom = Fr.mesh.CenterCoor[ID_ymin_bottom[0]][1] + Fr.mesh.hy/2

        # ged hy_new and hx_new
        N_payzone = 13
        hx_new = (yup - ybottom) / N_payzone
        hy_new = hx_new

        # get new Lx and new Ly
        nx_new = 301; ny_new = nx_new
        Lx_new = hx_new * (nx_new - 1) / 2.
        Ly_new = Lx_new

        # create a new mesh
        Mesh_new = CartesianMesh(Lx_new, Ly_new, nx_new, ny_new)

        # se the new elasticity matrix
        max_leaf_size = 600
        eta = 8
        eps_aca = 0.0001
        HMATparam = [max_leaf_size, eta, eps_aca]
        C = load_isotropic_elasticity_matrix_toepliz(Mesh_new, Eprime, C_precision=np.float64,
                                                     useHMATdot=True, nu=nu,
                                                     HMATparam=HMATparam)


        # overwriting the stress function

        Solid.SigmaOFunc = stress_func_factory(sigmaHigh, Solid.SigmaO.min(), yup)
        Fr.EltTipBefore = Fr.EltTip

        # project the fracture on a new mesh
        Solid_new, Fr_new = Fr.project_solution_to_a_new_mesh(C, Mesh_new, Solid, Fluid, Injection, simulProp, Fr.Ffront)

    elif reload_from_other and not reload_from_radial:
        from src.utilities.visualization import *

        Fr_list, properties = load_fractures(address=myfolder,
                                             step_size=1)  # load all fractures                                                # list of times
        Solid, Fluid, Injection, simulProp = properties
        Fr = Fr_list[-1]
        # mesh_dict = copy.deepcopy(Fr.mesh)
        # Fr.mesh = CartesianMesh(mesh_dict['domain Limits'][[2,3]].tolist(),mesh_dict['domain Limits'][[0,1]].tolist(),mesh_dict['nx'],mesh_dict['ny'])
        # setting up mesh extension options
        simulProp.meshExtension = True
        simulProp.meshExtensionAllDir = True
        simulProp.set_mesh_extension_direction(['all'])
        simulProp.set_mesh_extension_factor(1.25)
        simulProp.frontAdvancing = 'predictor-corrector'
        simulProp.maxFrontItrs = 50
        simulProp.tmStpPrefactor = 2.
        simulProp.custom = custom_factory(27000., 'y/(0.5 H)', 'x/(0.5 H)')
        #N = 19; H=27000. #Solid.SigmaOFunc.ymax_layer
        #simulProp.get_time_step_prefactor = my_time_step_prefactor(N,H)
        simulProp.force_time_schedule = False
        simulProp.set_solTimeSeries(np.array([0.]))
        #simulProp.set_solTimeSeries(np.logspace(np.log10(Fr.time),np.log10(Fr.time)+6,120))
        Fr_new = Fr
        Solid_new = Solid

        # se the new elasticity matrix
        """
        leaf size maybe 500 or 750, epsilon plug it to 10^-4 (not higher than that) eta either 10 or 5 both should give similar results but 5 will take longer to calculate.
        """
        #max_leaf_size = 600
        #eta = 8
        eps_aca = 0.0001
        #chenged around 1 million elements
        max_leaf_size = 650
        eta = 8

        HMATparam = [max_leaf_size, eta, eps_aca]
        C = load_isotropic_elasticity_matrix_toepliz(Fr.mesh, Eprime, C_precision=np.float64,
                                                     useHMATdot=True, nu=nu,
                                                     HMATparam=HMATparam)

    elif reload_from_other and reload_from_radial:
        msg = "\n You can either reload from a radial sim or reload from a previous non radial sim."
        print(msg)
        SystemExit(msg)
    else:
        Fr_new = Fr
        Solid_new = Solid

        # se the new elasticity matrix
        max_leaf_size = 15000
        eta = 8
        eps_aca = 0.0001
        HMATparam = [max_leaf_size, eta, eps_aca]
        C = load_isotropic_elasticity_matrix_toepliz(Mesh, Eprime, C_precision=np.float64,
                                                     useHMATdot=True, nu=nu,
                                                     HMATparam=HMATparam)

    # from src.utilities.utility import plot_as_matrix
    # plot_as_matrix(Fr_new.w, Fr_new.mesh)
    # plot_as_matrix(Fr_new.pNet, Fr_new.mesh)

    # create a Controller
    controller = Controller(Fr_new,
                            Solid_new,
                            Fluid,
                            Injection,
                            simulProp,
                            C = C)

    # run the simulation
    controller.run()


####################
# plotting results #
####################

if plot:
    from utilities.visualization import *

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

to_export = [1,2,3,4,5]

if export_results:


    # 0) import some functions needed later
    if 1 in to_export:
        from utilities.visualization import *
        from utilities.postprocess_fracture import append_to_json_file, load_fractures

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
        my_X = 0.0 ; my_Y = 0.0
        w_at_my_point, time_list_at_my_point = get_fracture_variable_at_point(Fr_list, variable='w', point=[my_X, my_Y])
        append_to_json_file(myJsonName_1, w_at_my_point, 'append2keyASnewlist', key='w_at_my_point')
        append_to_json_file(myJsonName_1, time_list_at_my_point, 'append2keyASnewlist', key='time_list_W_at_my_point')
        print(" <-- DONE\n")



    # 5) get the fluid pressure p(t) versus time (t) at a point of coordinates myX and myY
    if 5 in to_export:
        print("\n 4) get pn(t) at a point... ")
        my_X = 0.0 ; my_Y = 0.
        pf_at_my_point, time_list_at_my_point = get_fracture_variable_at_point(Fr_list, variable='pn', point=[my_X, my_Y])
        append_to_json_file(myJsonName_1, pf_at_my_point, 'append2keyASnewlist', key='pn_at_my_point_A')
        append_to_json_file(myJsonName_1, time_list_at_my_point, 'append2keyASnewlist', key='time_list_pn_at_my_point_A')
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
        from utilities.postprocess_fracture import get_velocity_slice
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
        from utilities.postprocess_fracture import get_velocity_slice
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

