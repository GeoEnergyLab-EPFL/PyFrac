# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Carlo Peruzzo on Fri Apr 17 23:16:25 2020.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2020.
All rights reserved. See the LICENSE.TXT file for more details.
"""


# post processing function
def get_info(Fr_list_A):  # get L(t) and x_max(t) and p(t)
    double_L_A = [];
    x_max_A = [];
    p_A = [];
    w_A=[];
    time_simul_A = [];
    center_indx = (Fr_list_A[0]).mesh.locate_element( 0., 0.)
    for frac_sol in Fr_list_A:
        # we are at a give time step now,
        # I am getting double_L_A, x_max_A
        x_min_temp = 0.
        x_max_temp = 0.
        y_min_temp = 0.
        y_max_temp = 0.
        for i in range(frac_sol.Ffront.shape[0]):
            segment = frac_sol.Ffront[i]
            # to find the x_max at this time:
            if segment[0] > x_max_temp:
                x_max_temp = segment[0]
            if segment[2] > x_max_temp:
                x_max_temp = segment[2]
            # to find the n_min at this time:
            if segment[0] < x_min_temp:
                x_min_temp = segment[0]
            if segment[2] < x_min_temp:
                x_min_temp = segment[2]
            # to find the y_max at this time:
            if segment[1] > y_max_temp:
                y_max_temp = segment[1]
            if segment[3] > y_max_temp:
                y_max_temp = segment[3]
            # to find the y_min at this time:
            if segment[1] < y_min_temp:
                y_min_temp = segment[1]
            if segment[3] < y_min_temp:
                y_min_temp = segment[3]

        double_L_A.append(y_max_temp - y_min_temp)
        x_max_A.append(x_max_temp - x_min_temp)

        p_A.append(frac_sol.pFluid.max() / 1.e6)
        w_A.append(frac_sol.w.max())
        time_simul_A.append(frac_sol.time)
    return double_L_A, x_max_A, p_A,w_A,  time_simul_A

# imports
import os
import time
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
from utilities.postprocess_fracture import load_fractures
from solid.elasticity_isotropic import load_isotropic_elasticity_matrix_toepliz

# setting up the verbosity level of the log at console
setup_logging_to_console(verbosity_level='debug')

########## OPTIONS #########
run = False
run_dir =  "./"
restart = False

# postprocessing
plot_B = False
output_fol_B = run_dir
output_fol  = "./"
ftPntJUMP = 5
plot_slices = False
############################

if run:
    # creating mesh
    Mesh = CartesianMesh(0.7, 0.7, 91, 91)

    # solid properties
    nu = 0.4  # Poisson's ratio
    youngs_mod = 3.3e9  # Young's modulus
    Eprime = youngs_mod / (1 - nu ** 2)  # plain strain modulus

    def smoothing(K1, K2, r, delta, x):
        # instead of having -10/10, take the MESHNAME.Ly/Lx (if mesh square)
        #### LINEAR - DIRAC DELTA ####
        x = np.abs(x)
        if  x < r-delta :
            return K1
        elif x >= r-delta and x<r :
            K12 = K1 + (K2-K1)*0.01
            a = (K12 - K1) / (delta)
            b = K1 - a * (r - delta)
            return a * x + b
        elif x >= r:
            return K2
        else:
            print("ERROR")


    def K1c_func(x,y, alpha):
        """ The function providing the toughness"""
        K_Ic = 0.4e6  # fracture toughness
        r = 1.48
        delta = 0.0005
        return smoothing(K_Ic, 10.*K_Ic, r, delta, x)


    # ---- plot Kic_max vs time ---
    # import matplotlib.pyplot as plt
    # xlabel = 'x [m]'
    # ylabel = 'KIc [kPa.m^(1/2)]'
    # fig, ax = plt.subplots()
    # x = []
    # y = []
    # aa = 0.00001
    # for i in range(int((1.484-1.46)/aa)):
    #     x.append(1.46+i*aa)
    #     y.append(K1c_func(x[i],0.))
    # ax.scatter(x, y, color='k')
    # ax.set_xlabel(xlabel)
    # ax.set_ylabel(ylabel)
    # plt.show()

    def sigmaO_func(x, y):
        return 0

    Solid = MaterialProperties(Mesh,
                              Eprime,
                              K1c_func=K1c_func,
                              confining_stress_func = sigmaO_func,
                              minimum_width=0.)

    # injection parameters
    Q0 = 0.001
    Injection = InjectionProperties(Q0, Mesh)

    # fluid properties
    Fluid = FluidProperties(viscosity=0.1)

    # simulation properties
    simulProp = SimulationProperties()
    simulProp.finalTime = 105.12
    simulProp.tmStpPrefactor = 0.8
    simulProp.saveToDisk = True
    simulProp.tolFractFront = 0.0001
    simulProp.plotTSJump = 1
    simulProp.set_volumeControl(False)
    simulProp.bckColor = 'K1c'
    simulProp.set_outputFolder(run_dir)
    #simulProp.plotVar = ['ffvf', 'custom', 'regime']
    simulProp.plotVar = ['custom', 'regime','footprint']
    simulProp.frontAdvancing = 'implicit'
    simulProp.projMethod = 'LS_continousfront'
    simulProp.customPlotsOnTheFly = True
    simulProp.useBlockToeplizCompression = True
    simulProp.LHyst__ = []
    simulProp.tHyst__ = []
    simulProp.saveFluidFluxAsVector = False

    # setting up mesh extension options
    simulProp.meshExtensionAllDir = False
    simulProp.set_mesh_extension_factor(1.5)
    simulProp.set_mesh_extension_direction(['vertical'])
    simulProp.meshReductionPossible = False
    simulProp.simID = 'K1/K2=10.' # do not use _

    simulProp.EHL_GMRES = True
    simulProp.solve_monolithic = False
    simulProp.gmres_Restart = 1000
    simulProp.gmres_maxiter = 1000


    # initialization parameters
    Fr_geometry = Geometry('radial', radius=0.2)

    C = load_isotropic_elasticity_matrix_toepliz(Mesh, Eprime, C_precision=np.float64, useHMATdot=False, nu=nu)

    init_param = InitializationParameters(Fr_geometry, regime='M', elasticity_matrix=C)

    # creating fracture object
    Fr = Fracture(Mesh,
                  init_param,
                  Solid,
                  Fluid,
                  Injection,
                  simulProp)

    ################################################################################
    # the following lines are needed if you want to restart an existing simulation #
    ################################################################################
    if restart:
        from utilities.visualization import *
        Fr_list, properties = load_fractures(address=run_dir, step_size=100)
        Solid, Fluid, Injection, simulProp = properties
        Fr = Fr_list[-1]
        simulProp.saveFluidFluxAsVector = False
        simulProp.plotVar = ['custom', 'regime']
        Solid = MaterialProperties(Fr.mesh,
                                  Eprime,
                                  K1c_func=K1c_func,
                                  confining_stress_func = sigmaO_func,
                                  confining_stress=0.,
                                  minimum_width=0.)

        C = load_isotropic_elasticity_matrix_toepliz(Fr.mesh, Eprime, C_precision=np.float64, useHMATdot=False, nu=nu)
    ############################################################################


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


if not os.path.isfile('./batch_run.txt'):  # We only visualize for runs of specific examples

    from utilities.visualization import *

    # loading simulation results A
    Fr_list_A, properties_A = load_fractures(address=output_fol, load_all=True)  # load all fractures
    time_srs_A = get_fracture_variable(Fr_list_A, variable='time')  # list of times
    Solid_A, Fluid_A, Injection_A, simulProp_A = properties_A
    double_L_A, x_max_A, p_A, w_A, time_simul_A = get_info(Fr_list_A)
    if plot_B:
        # loading simulation results B
        Fr_list_B, properties_B = load_fractures(address=output_fol_B, load_all=True)  # load all fractures
        time_srs_B = get_fracture_variable(Fr_list_B, variable='time')  # list of times
        Solid_B, Fluid_B, Injection_B, simulProp_B = properties_B
        double_L_B, x_max_B, p_B, w_B, time_simul_B = get_info(Fr_list_B)

    # plot fracture radius
    my_list = []
    mytime = 195
    for i in np.arange(0,len(Fr_list_A),ftPntJUMP):
        if Fr_list_A[i].time < mytime:
            my_list.append(Fr_list_A[i])
    if plot_B:
        for i in np.arange(0, len(Fr_list_B), ftPntJUMP):
            if Fr_list_B[i].time < mytime:
                my_list.append(Fr_list_B[i])
    plot_prop = PlotProperties()
    Fig_R = plot_fracture_list(my_list,
                               variable='footprint',
                               plot_prop=plot_prop)
    Fig_R = plot_fracture_list(my_list,
                               fig=Fig_R,
                               variable='mesh',
                               mat_properties=properties_A[0],
                               backGround_param='K1c',
                               plot_prop=plot_prop)
    plt.show()
    #
    # # plot fracture radius
    # plot_prop = PlotProperties()
    # plot_prop.lineStyle = '.'               # setting the linestyle to point
    # plot_prop.graphScaling = 'loglog'       # setting to log log plot
    # Fig_R = plot_fracture_list(Fr_list,
    #                            variable='d_mean',
    #                            plot_prop=plot_prop)
    #
    # # plot analytical radius
    # Fig_R = plot_analytical_solution(regime='K',
    #                                  variable='d_mean',
    #                                  mat_prop=Solid,
    #                                  inj_prop=Injection,
    #                                  fluid_prop=Fluid,
    #                                  time_srs=time_srs,
    #                                  fig=Fig_R)
    #
    # # # plot width at center
    # Fig_w = plot_fracture_list_at_point(Fr_list,
    #                                     variable='w',
    #                                     plot_prop=plot_prop)
    # # # plot analytical width at center
    # Fig_w = plot_analytical_solution_at_point('K',
    #                                           'w',
    #                                           Solid,
    #                                           Injection,
    #                                           fluid_prop=Fluid,
    #                                           time_srs=time_srs,
    #                                           fig=Fig_w)
    #
    #
    # Fig_pf = plot_fracture_list_at_point(Fr_list,
    #                                     variable='pn',
    #                                     plot_prop=plot_prop)
    #
    # Fig_pf = plot_analytical_solution_at_point('K',
    #                                           'pn',
    #                                           Solid,
    #                                           Injection,
    #                                           fluid_prop=Fluid,
    #                                           time_srs=time_srs,
    #                                           fig=Fig_pf)
    #
    #
    # # plot slice
    # #ext_pnts = np.empty((2, 2), dtype=np.float64)
    # my_X = 0.
    # my_Y = 0.
    # ext_pnts = np.empty((2, 2), dtype=np.float64)
    # Fig_WS = plot_fracture_list_slice(Fr_list,
    #                                   variable='w',
    #                                   projection='2D',
    #                                   plot_cell_center=True,
    #                                   extreme_points=ext_pnts,
    #                                   orientation='horizontal',
    #                                   point1=[my_X, my_Y]
    #                                   )

    #################################





    #print("\n get w(x) with x passing through a specific point for different times... ")
    # my_X = 0.
    # my_Y = 0.
    # ext_pnts = np.empty((2, 2), dtype=np.float64)
    # fracture_list_slice_A = plot_fracture_list_slice(Fr_list_A,
    #                                                variable='w',
    #                                                projection='2D',
    #                                                plot_cell_center=True,
    #                                                extreme_points=ext_pnts,
    #                                                orientation='horizontal',
    #                                                point1=[my_X, my_Y],
    #                                                export2Json=True,
    #                                                export2Json_assuming_no_remeshing=True)
    # loading simulation results B
    # Fr_list_B, properties_B = load_fractures(address="./Data/sim_B",step_size=1)                  # load all fractures
    # time_srs_B = get_fracture_variable(Fr_list_A, variable='time')                                                 # list of times
    # Solid_B, Fluid_B, Injection_B, simulProp_B = properties_B
    # double_L_B, x_max_B, p_B, time_simul_B = get_info(Fr_list_B)
    # # plot slice
    # print("\n get w(x) with x passing through a specific point for different times... ")
    # my_X = 0.
    # my_Y = 0.
    # ext_pnts = np.empty((2, 2), dtype=np.float64)
    # fracture_list_slice_B = plot_fracture_list_slice(Fr_list_B,
    #                                                variable='w',
    #                                                projection='2D',
    #                                                plot_cell_center=True,
    #                                                extreme_points=ext_pnts,
    #                                                orientation='horizontal',
    #                                                point1=[my_X, my_Y],
    #                                                export2Json=True,
    #                                                export2Json_assuming_no_remeshing=True)

    #### BUILDING OUR PLOTS ###
    import numpy as np
    import matplotlib.pyplot as plt
    ######################################################
    # plot the pressure vs time
    ######################################################
    xlabel = 'time [s]'
    ylabel = 'Pressure [MPa]'
    fig, ax = plt.subplots()
    ax.scatter(time_simul_A, p_A, color='k')
    p_ana = []
    for i in range(len(time_simul_A)):
        #p_ana.append(0.3279)
        p_ana.append(2*0.5/np.sqrt(np.pi*2*1.48))

    p_rad = []
    for i in range(len(time_simul_A)):
        p_rad.append(((np.pi ** 3. * (0.5e6)**6 /(12*Solid_A.Eprime*Injection_A.injectionRate.max()*time_simul_A[i]))**(1/5))/1.e6)
    ax.plot(time_simul_A, p_ana, color='g')
    ax.plot(time_simul_A, p_rad, color='g')
    if plot_B: ax.scatter(time_simul_B, p_B, color='r')
    # p_scaling = []
    # for i in range(len(time_simul_A)):
    #     p_scaling.append(0.05/time_simul_A[i])
    # ax.plot(time_simul_A, p_scaling, color='b')

    if plot_B: ax.scatter(time_simul_B, p_B, color='g')
    p_scaling = []
    for i in range(len(time_simul_A)):
        p_scaling.append(0.2660 * time_simul_A[i] ** (1 / 5))
    ax.plot(time_simul_A, p_scaling, color='g')

    sl= []

    K_Ic = 0.5e6
    H=2*1.48
    p_limit = 1.46*K_Ic/np.sqrt(np.pi*H/2)/1.e6
    for i in range(len(time_simul_A)):
        sl.append(p_limit)
    ax.plot(time_simul_A, sl, color='r')


    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_yscale('log')
    ax.set_xscale('log')

    ######################################################
    # plot 2L vs time
    ######################################################
    xlabel = 'time [s]'
    ylabel = '2L [m]'
    fig, ax = plt.subplots()
    H=1.48*2
    doubleL_scaling = []
    for i in range(len(time_simul_A)):
        doubleL_scaling.append(9.5/H*time_simul_A[i]**(2/3))
        double_L_A[i] = double_L_A[i]/H
    ax.scatter(time_simul_A, double_L_A, color='k')
    # ax.plot(time_simul_A, doubleL_scaling, color='g')
    # if plot_B:
    #     for i in range(len(time_simul_B)):
    #         double_L_B[i] = double_L_B[i]/H
    #     ax.scatter(time_simul_B, double_L_B, color='r')
    # doubleL_radial = []
    # for i in range(len(time_simul_A)):
    #     doubleL_radial.append(6/H*time_simul_A[i]**(2/5))
    # ax.plot(time_simul_A, doubleL_radial, color='b')
    doubleL_scaling2 = []
    for i in range(len(time_simul_A)):
        doubleL_scaling2.append(1.5/H*time_simul_A[i]**(4/5))
    ax.plot(time_simul_A, doubleL_scaling2, color='b')
    doubleL_pkn = []
    # for i in range(len(time_simul_A)):
    #     doubleL_pkn.append(9/H*time_simul_A[i])
    # ax.plot(time_simul_A, doubleL_pkn, color='r')

    # ax.scatter(time_simul_B, p_B, color='g')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_yscale('log')
    ax.set_xscale('log')

    ######################################################
    # plot x_max vs time
    ######################################################
    xlabel = 'time [s]'
    ylabel = 'x max [m]'
    fig, ax = plt.subplots()
    ax.scatter(time_simul_A, x_max_A, color='k')
    if plot_B: ax.scatter(time_simul_B, x_max_B, color='r')
    p_const = []
    for i in range(len(time_simul_A)):
        p_const.append(1.6)
    ax.plot(time_simul_A, p_const, color='g')
    # ax.scatter(time_simul_B, p_B, color='g')
    p_scaling = []
    for i in range(len(time_simul_A)):
        p_scaling.append(3.5*time_simul_A[i]**(2/15))
    ax.plot(time_simul_A, p_scaling, color='g')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_yscale('log')
    ax.set_xscale('log')

    ######################################################
    # plot w_cemter vs time
    ######################################################
    xlabel = 'time [s]'
    ylabel = 'w center [m]'
    fig, ax = plt.subplots()
    ax.scatter(time_simul_A, w_A, color='k')
    if plot_B: ax.scatter(time_simul_B, w_B, color='r')
    w_scaling = []
    w_radial = []
    # for i in range(len(time_simul_A)):
    #     w_scaling.append(0.0000660*time_simul_A[i]**(1/3))
    # ax.plot(time_simul_A, w_scaling, color='g')
    for i in range(len(time_simul_A)):
        w_radial.append(0.000350*time_simul_A[i]**(1/5))
    ax.plot(time_simul_A, w_radial, color='b')
    # ax.scatter(time_simul_B, p_B, color='g')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_yscale('log')
    ax.set_xscale('log')

    ######################################################
    # --> relative error of the volume VS injected volume
    ######################################################
    V = []
    for fr in Fr_list_A:
        V.append((np.sum(fr.w)*fr.mesh.hx*fr.mesh.hy-fr.time*Injection_A.injectionRate.max())/fr.time*Injection_A.injectionRate.max())
    xlabel = 'time [s]'
    ylabel = 'volume res [m]'
    fig, ax = plt.subplots()
    ax.scatter(time_simul_A, V, color='k')
    w_scaling = []
    w_radial = []
    # ax.scatter(time_simul_B, p_B, color='g')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xscale('log')
    plt.show()

    if plot_slices:
        # --> plot slice
        print("\n get w(x) with x passing through a specific point for different times... ")
        my_X = 0.
        my_Y = 0.
        ext_pnts = np.empty((2, 2), dtype=np.float64)
        fracture_list_slice_B = plot_fracture_list_slice(my_list,
                                                       variable='w',
                                                       projection='2D',
                                                       plot_cell_center=True,
                                                       extreme_points=ext_pnts,
                                                       orientation='vertical',
                                                       point1=[my_X, my_Y],
                                                       export2Json=False)

        # --> plot slice
        print("\n get w(x) with x passing through a specific point for different times... ")
        my_X = 0.
        my_Y = 0.
        ext_pnts = np.empty((2, 2), dtype=np.float64)
        fracture_list_slice_B = plot_fracture_list_slice(my_list,
                                                       variable='w',
                                                       projection='2D',
                                                       plot_cell_center=True,
                                                       extreme_points=ext_pnts,
                                                       orientation='horizontal',
                                                       point1=[my_X, my_Y],
                                                       export2Json=False)
    plt.show(block=True)