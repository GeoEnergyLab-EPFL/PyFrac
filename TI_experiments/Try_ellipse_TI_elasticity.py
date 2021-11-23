# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Fatima-Ezzahra Moukhtari on Thu Aug 21, 2019.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory",
2016-2019. All rights reserved. See the LICENSE.TXT file for more details.
"""

# imports
import numpy as np

# local imports
from solid.elasticity_Transv_Isotropic import TI_plain_strain_modulus
from mesh_obj.mesh import CartesianMesh
from solid.solid_prop import MaterialProperties
from fluid.fluid_prop import FluidProperties
from properties import InjectionProperties, SimulationProperties
from fracture_obj.fracture import Fracture
from controller import Controller
from fracture_obj.fracture_initialization import Geometry, InitializationParameters
from utilities.utility import setup_logging_to_console
from utilities.postprocess_fracture import load_fractures

# setting up the verbosity level of the log at console
setup_logging_to_console(verbosity_level='debug')

# creating mesh
#Mesh = CartesianMesh(0.1, 0.1, 161, 161)
Mesh = CartesianMesh([-0.12,0.12], [-0.12, 0.02], 161, 161)

sigma_o = 0.5e6  # confining stress [Pa]

# solid properties
Cij = np.zeros((6, 6), dtype=float)
Cij[0, 0] = 114.6
Cij[0, 1] = 28.46
Cij[5, 5] = 0.5*(Cij[0, 0]-Cij[0, 1])
Cij[0, 2] = 47.49
Cij[2, 2] = 75.49
Cij[3, 3] = 35.86
Cij[1, 1] = Cij[0, 0]
Cij[1, 0] = Cij[0, 1]
Cij[2, 0] = Cij[0, 2]
Cij[2, 1] = Cij[0, 2]
Cij[1, 2] = Cij[0, 2]
Cij[4, 4] = Cij[3, 3]
Cij = Cij * 1e9

Eprime = TI_plain_strain_modulus(np.pi/2, Cij) # plain strain modulus

# the function below will make the fracture propagate in the form of an ellipse (see Zia and Lecampion, 2018)
# def K1c_func(alpha):
#     """ function giving the dependence of fracture toughness on propagation direction alpha"""
#
#     K1c_3 = 3.5e6                        # fracture toughness along y-axis
#     K1c_1 = 2.5e6                        # fracture toughness along x-axis
#
#     Eprime_ratio = TI_plain_strain_modulus(alpha, Cij) / TI_plain_strain_modulus(np.pi/2, Cij)
#     Eprime_ratio13 = TI_plain_strain_modulus(0, Cij) / TI_plain_strain_modulus(np.pi / 2, Cij)
#     gamma = (Eprime_ratio13 * K1c_3/K1c_1)**2  # aspect ratio
#     beta = np.arctan(np.tan(alpha) / gamma)
#
#     return K1c_3 * Eprime_ratio * ((np.sin(beta))**2 + (np.cos(beta)/gamma)**2)**0.25

def K1c_angle_func(alpha):
    """ function giving the dependence of fracture toughness on propagation direction alpha"""

    if alpha > np.pi/2. and alpha <= np.pi:
        alpha = np.pi-alpha
    elif alpha > np.pi and alpha <= 3*np.pi/2:
        alpha = alpha - np.pi
    elif alpha > 3*np.pi/2 and alpha <= 2*np.pi:
        alpha = 2*np.pi - alpha


    K1c_3 = 3.5e6                        # fracture toughness along y-axis
    K1c_1 = 2.5e6                        # fracture toughness along x-axis

    Eprime_ratio = TI_plain_strain_modulus(alpha, Cij) / TI_plain_strain_modulus(np.pi/2, Cij)
    Eprime_ratio13 = TI_plain_strain_modulus(0, Cij) / TI_plain_strain_modulus(np.pi / 2, Cij)
    gamma = (Eprime_ratio13 * K1c_3/K1c_1)**2  # aspect ratio
    beta = np.arctan(np.tan(alpha) / gamma)

    return K1c_3 * Eprime_ratio * ((np.sin(beta))**2 + (np.cos(beta)/gamma)**2)**0.25

def smoothing(K2, Ylim, delta, y, alpha):
    # instead of having -10/10, take the MESHNAME.Ly/Lx (if mesh square)
    #### LINEAR - DIRAC DELTA ####
    # x = np.abs(x)
    if  y < Ylim-delta :
        return K1c_angle_func(alpha)
    elif y >= Ylim-delta and y<Ylim :
        K12 = K1c_angle_func(alpha) + (K2-K1c_angle_func(alpha))*0.1
        a = (K12 - K1c_angle_func(alpha)) / (delta)
        b = K1c_angle_func(alpha) - a * (Ylim - delta)
        return a * y + b
    elif y >= Ylim:
        return K2
    else:
        print("ERROR")

def K1c_func(x,y,alpha):
    """ The function providing the toughness"""
    ylim = 0.015
    delta = 0.001
    K2 = 2.E7
    res = smoothing(K2, ylim, delta, y, alpha)
    return res

# materila properties
Solid = MaterialProperties(Mesh,
                           Eprime,
                           confining_stress=sigma_o,
                           anisotropic_K1c=True,
                           toughness=K1c_func(0,0,np.pi/2),
                           K1c_func=K1c_func,
                           TI_elasticity=True,
                           Cij=Cij)

# injection parameters
# Original averaged rates - too many data points
# times when the injection rate changes [s]
# Note: For injection histories the first entry always needs to be 0.0!
# t_change = [0, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 27, 32, 40, 50, 65, 80, 100, 125, 150, 175, 200]
# Qo = [2.641E-10, 4.40482E-10, 7.30516E-10, 1.43088E-09, 2.45744E-09, 3.41019E-09, 4.36517E-09, 6.21312E-09, 7.89502E-09,
#       7.15628E-09, 5.49541E-09, 4.22421E-09, 3.65004E-09, 3.25138E-09, 2.86383E-09, 2.34831E-09, 1.87419E-09, 1.75704E-09,
#       1.33455E-09, 1.13618E-09, 1.03664E-09, 9.78541E-10]

# Qo = 6.7E-10
# Estimated influx rate from only downstream U
# t_change = [0, 13, 16, 19, 22, 31, 53, 78, 133]
# Qo = [6E-10, 3.40687E-09, 5.68692E-09, 7.52565E-09, 4.68708E-09, 3.36481E-09, 2.59028E-09, 1.76568E-09, 1.1239E-09]
t_change = [0, 13, 23, 73, 123, 173]
Qo = [6E-10, 2E-09, 2.7E-09, 2.05E-09, 1.5E-09, 1.1E-09]
# If use this set, add 8 seconds (and then account for 2 seconds of pressurization) to the start time in comparison
# with Inversion prediction

# Injection rates in the time windows defined in t_change [m^3/s]
# Note: For a continuous injection use "Qo = your_injection_rate". For injection histories the number of entries in Qo
# must match with the number of entries in t_change. First entry of Qo is injection for t between the first and second
# entry of t_change and so on.

# Qo = 6.7e-10  # injection rate

if type(Qo) == list:
    inj = np.asarray([t_change,
                      Qo])
else:
    inj = Qo

Injection = InjectionProperties(inj, Mesh)

# fixed_times = [11, 21, 32, 42, 52, 62, 72, 82, 92, 103, 113, 123, 133, 143, 153, 163, 174, 184, 194, 204, 214,
#                224, 235, 245, 255]
fixed_times = [5, 15, 25, 36, 46, 56, 66, 76, 86, 96, 107, 117, 127, 137, 147, 157, 167, 178, 188, 198, 208]
# The fracture will automatically be saved at the times fixed within fixed_times [s]. If you leave it empty "fixed_times
# = []" the default scheme is applied.

# fluid properties
Fluid = FluidProperties(viscosity=0.11)

# aspect ratio of the elliptical fracture
gamma  = (K1c_func(0,0,np.pi/2) / K1c_func(0,0,0) * TI_plain_strain_modulus(    # gamma = (Kc3/Kc1*E1/E3)**2
            0, Cij)/TI_plain_strain_modulus(np.pi/2, Cij))**2

# simulation properties
simulProp = SimulationProperties()
simulProp.finalTime = 208            # the time at which the simulation stops
if len(fixed_times) != 0:
    simulProp.set_solTimeSeries(np.asarray(fixed_times))
simulProp.tolFractFront = 1e-3          # 4e-3         # increase tolerance for the anisotropic case
simulProp.maxFrontItrs  = 25
simulProp.tmStp_prefactor = 1.
simulProp.frontAdvancing = "implicit"   # set the front advancing scheme to implicit
simulProp.set_outputFolder("./Data/TI_elasticity_ellipse")  # setting the output folder
simulProp.set_simulation_name('TI_ellasticy_benchmark')     # setting the simulation name
simulProp.TI_KernelExecPath = '../TI_Kernel/build/'         # path to the executable that calculates TI kernel
simulProp.remeshFactor = 2            # the factor by which the domain is expanded
#simulProp.projMethod = 'ILSA_orig'    # comment out this line so the default is continuous front scheme
simulProp.set_tipAsymptote('U1')
simulProp.useBlockToeplizCompression = True
# simulProp.fixedTmStp = np.asarray([[0.0001, 0.5],[1., 0.6]])
# simulProp.tmStpPrefactor = 1.
# initialization parameters
# Fr_geometry = Geometry('elliptical',
#                        minor_axis=1,
#                        gamma=gamma)
# init_param = InitializationParameters(Fr_geometry, regime='E_E')
# init_param = InitializationParameters(Fr_geometry, regime='E_E')

########################################################################################################################
r_init = 0.01    # 0.00885
# initial radius of the fracture [m]
Fr_geometry = Geometry(shape='radial',
                       radius=r_init)
# init_param = InitializationParameters(Fr_geometry, regime='K')
init_param = InitializationParameters(Fr_geometry, regime='static',
                                      net_pressure=1e4,
                                      width=1.e-6,
                                      prescribe_w_and_pnet=True,
                                      )
########################################################################################################################


# #  creating fracture object
Fr = Fracture(Mesh,
              init_param,
              Solid,
              Fluid,
              Injection,
              simulProp)
# restart = True
# if restart == True:
#     run_dir = ".\\Data\\TI_ellasticy_benchmark"
#     from visualization import *
#     Fr_list, properties = load_fractures(address=run_dir, step_size=100, sim_name='TI_ellasticy_benchmark')
#     Solid, Fluid, Injection, simulProp = properties
#     Fr = Fr_list[-1]
#     Solid = MaterialProperties(Fr.mesh,
#                                Eprime,
#                                confining_stress=sigma_o,
#                                anisotropic_K1c=True,
#                                toughness=K1c_func(np.pi / 2),
#                                K1c_func=K1c_func,
#                                TI_elasticity=True,
#                                Cij=Cij)
# create a Controller
controller = Controller(Fr,
                        Solid,
                        Fluid,
                        Injection,
                        simulProp)

#run the simulation
controller.run()

####################
# plotting results #
####################
#
# ########################################################################################################################
# # -- loading simulation results -- #
# from visualization import *
# save_folder = "C:/research/EPFL_projects/Code related/PyFrac-updated_cpp_binding/PyFrac-cpp_binding/examples/Data/TI_elasticity_ellipse"
# sim_name = 'TI_ellasticy_benchmark'
# Fr_list, properties = load_fractures(address=save_folder, sim_name=sim_name,  step_size=1)  # load all fractures
# time_srs = get_fracture_variable(Fr_list, variable='time')  # list of times
# Solid, Fluid, Injection, simulProp = properties
#
# # -- FIGURE 1) plot fracture footprint VS time -- #
# # plot the solution every "tsJump" time steps
# tsJump = 10
# sol_at_selected_times = []
# for i in np.arange(0, len(Fr_list), tsJump):
#     sol_at_selected_times.append(Fr_list[i])
# plot_prop = PlotProperties()
# Fig_R = plot_fracture_list(sol_at_selected_times,
#                             variable='footprint',
#                             plot_prop=plot_prop)
# Fig_R = plot_fracture_list(sol_at_selected_times,
#                             fig=Fig_R,
#                             variable='mesh',
#                             mat_properties=properties[0],
#                             plot_prop=plot_prop)
#
# # -- FIGURE 2) plot fracture radius VS time -- #
# # plot the solution every "tsJump" time steps
# tsJump = 3
# sol_at_selected_times = []
# for i in np.arange(0, len(Fr_list), tsJump):
#     sol_at_selected_times.append(Fr_list[i])
# plot_prop = PlotProperties()
# plot_prop.lineStyle = '.'  # setting the linestyle to point
# plot_prop.graphScaling = 'loglog'  # setting to log log plot
# Fig_R = plot_fracture_list(sol_at_selected_times,
#                            variable='d_mean',
#                            plot_prop=plot_prop)
#
# # -- FIGURES 3) and 4) plot fracture opening VS time at a given point (my_X, my_Y)-- #
# # plot the solution every "tsJump" time steps
# tsJump = 3
# my_X = 0.0
# my_Y = 0.0
# sol_at_selected_times = []
# for i in np.arange(0, len(Fr_list), tsJump):
#     sol_at_selected_times.append(Fr_list[i])
#
# Fig_w = plot_fracture_list_at_point(sol_at_selected_times,
#                                     variable='w',
#                                     point=[my_X, my_Y],
#                                     plot_prop=plot_prop)
#
# # -- FIGURES 5) and 6) plot net fluid pressure VS time at a given point (my_X, my_Y)-- #
# # plot the solution every "tsJump" time steps
# tsJump = 3
# my_X = 0.0
# my_Y = 0.0
#
# sol_at_selected_times = []
# for i in np.arange(0, len(Fr_list), tsJump):
#     sol_at_selected_times.append(Fr_list[i])
#
# Fig_pf = plot_fracture_list_at_point(sol_at_selected_times,
#                                      variable='pn',
#                                      point=[my_X, my_Y],
#                                      plot_prop=plot_prop)
#
# # -------------- exporting to json file -------------- #
#
# from visualization import *
# from postprocess_fracture import append_to_json_file
#
# # 1) export general information to json
# # 2) export to json the coordinates of the points defining the fracture front at each time
# # 3) export the fracture opening w(t) versus time (t) at a point of coordinates myX and myY
# # 4) export the fluid pressure p(t) versus time (t) at a point of coordinates myX and myY
# # 5) export w(y) along a vertical line passing through mypoint for different times
# # 6) export pf(x) along a horizontal line passing through mypoint for different times
# # 7) export w(x,y,t) and pf(x,y,t)
#
# to_export = [1,2,3,4,5,6,7]
#
# myJsonName_1 = "./Data/Pyfrac_"+sim_name+"_export.json"
#
# # load the results:
# print("\n 1) loading results")
# Fr_list, properties = load_fractures(address=save_folder, sim_name=sim_name, load_all=True)
# # or Fr_list, properties = load_fractures(address=save_folder, sim_name=sim_name, time_srs=np.linspace(initial_time, final_time, steps))
# Solid, Fluid, Injection, simulProp = properties
# print(" <-- DONE\n")
#
# # 1) export general information to json
# if 1 in to_export:
#     print("\n 2) writing general info")
#     time_srs = get_fracture_variable(Fr_list, variable='time')
#     time_srs = np.asarray(time_srs)
#
#     simul_info = {'Eprime': Solid.Eprime,
#                   'max_KIc': Solid.K1c.max(),
#                   'min_KIc': Solid.K1c.min(),
#                   'max_Sigma0': Solid.SigmaO.max(),
#                   'min_Sigma0': Solid.SigmaO.min(),
#                   'viscosity': Fluid.viscosity,
#                   'total_injection_rate': Injection.injectionRate.max(),
#                   'sources_coordinates_lastFR': Fr_list[-1].mesh.CenterCoor[Injection.sourceElem].tolist(),
#                   't_max': time_srs.max(),
#                   't_min': time_srs.min()}
#     append_to_json_file(myJsonName_1, simul_info, 'append2keyASnewlist', key='simul_info',
#                         delete_existing_filename=True)  # be careful: delete_existing_filename=True only the first time you call "append_to_json_file"
#
# # 2) export the coordinates of the points defining the fracture front at each time:
# if 2 in to_export:
#     print("\n 2) writing fronts")
#     time_srs = get_fracture_variable(Fr_list,variable='time') # get the list of times corresponding to each fracture object
#     append_to_json_file(myJsonName_1, time_srs, 'append2keyASnewlist', key='time_srs_of_Fr_list')
#     fracture_fronts = []
#     numberof_fronts = [] #there might be multiple fracture fronts in general
#     mesh_info = [] # if you do not make remeshing or mesh extension you can export it only once
#     index = 0
#     for fracture in Fr_list:
#         fracture_fronts.append(np.ndarray.tolist(fracture.Ffront))
#         numberof_fronts.append(fracture.number_of_fronts)
#         mesh_info.append([Fr_list[index].mesh.Lx, Fr_list[index].mesh.Ly, Fr_list[index].mesh.nx, Fr_list[index].mesh.ny])
#         index = index + 1
#     append_to_json_file(myJsonName_1, fracture_fronts, 'append2keyASnewlist', key='Fr_list')
#     append_to_json_file(myJsonName_1, numberof_fronts, 'append2keyASnewlist', key='Number_of_fronts')
#     append_to_json_file(myJsonName_1,mesh_info,'append2keyASnewlist', key='mesh_info')
#     print(" <-- DONE\n")
#
# # 3) export the fracture opening w(t) versus time (t) at a point of coordinates myX and myY
# if 3 in to_export:
#     print("\n 3) get w(t) at a point... ")
#     my_X = 0.; my_Y = 0.
#     w_at_my_point, time_list_at_my_point = get_fracture_variable_at_point(Fr_list, variable='w', point=[my_X, my_Y])
#     append_to_json_file(myJsonName_1, w_at_my_point, 'append2keyASnewlist', key='w_at_my_point')
#     append_to_json_file(myJsonName_1, time_list_at_my_point, 'append2keyASnewlist', key='time_list_W_at_my_point')
#     print(" <-- DONE\n")
#
# # 4) export the fluid pressure p(t) versus time (t) at a point of coordinates myX and myY
# if 4 in to_export:
#     print("\n 4) get pf(t) at a point... ")
#     my_X = 0.; my_Y = 0.
#     pf_at_my_point, time_list_at_my_point = get_fracture_variable_at_point(Fr_list, variable='pf', point=[my_X, my_Y])
#     append_to_json_file(myJsonName_1, pf_at_my_point, 'append2keyASnewlist', key='pf_at_my_point_A')
#     append_to_json_file(myJsonName_1, time_list_at_my_point, 'append2keyASnewlist', key='time_list_pf_at_my_point_A')
#     print(" <-- DONE\n")
#
# # 5) export w(y) along a vertical line passing through mypoint for different times
# if 5 in to_export:
#     print("\n 5) get w(y) with y passing through a specific point for different times... ")
#     my_X = 0.; my_Y = 0.
#     ext_pnts = np.empty((2, 2), dtype=np.float64)
#     fracture_list_slice = plot_fracture_list_slice(Fr_list,
#                                                    variable='w',
#                                                    projection='2D',
#                                                    plot_cell_center=True,
#                                                    extreme_points=ext_pnts,
#                                                    orientation='horizontal',
#                                                    point1=[my_X , my_Y],
#                                                    export2Json=True,
#                                                    export2Json_assuming_no_remeshing=False)
#     towrite = {'w_vert_slice_': fracture_list_slice}
#     append_to_json_file(myJsonName_1, towrite, 'extend_dictionary')
#     print(" <-- DONE\n")
#
# # 6) export pf(x) along a horizontal line passing through mypoint for different times
# if 6 in to_export:
#     print("\n 6) get pf(x) with x passing through a specific point for different times... ")
#     my_X = 0.; my_Y = 0.
#     ext_pnts = np.empty((2, 2), dtype=np.float64)
#     fracture_list_slice = plot_fracture_list_slice(Fr_list,
#                                                    variable='pf',
#                                                    projection='2D',
#                                                    plot_cell_center=True,
#                                                    extreme_points=ext_pnts,
#                                                    orientation='horizontal',
#                                                    point1=[my_X , my_Y],
#                                                    export2Json=True,
#                                                    export2Json_assuming_no_remeshing=False)
#     towrite = {'pf_horiz_slice_': fracture_list_slice}
#     append_to_json_file(myJsonName_1, towrite, 'extend_dictionary')
#     print(" <-- DONE\n")
#
# # 7) export w(x,y,t) and pf(x,y,t)
# if 7 in to_export:
#     print("\n 7) get w(x,y,t) and  pf(x,y,t)... ")
#     wofxyandt = []
#     pofxyandt = []
#     info = []
#     jump = True #this is used to jump the first fracture
#     for frac in Fr_list:
#         if not jump:
#             wofxyandt.append(np.ndarray.tolist(frac.w))
#             pofxyandt.append(np.ndarray.tolist(frac.pFluid))
#             info.append([frac.mesh.Lx,frac.mesh.Ly,frac.mesh.nx,frac.mesh.ny,frac.time])
#         else:
#             jump = False
#
#     append_to_json_file(myJsonName_1, wofxyandt, 'append2keyASnewlist', key='w')
#     append_to_json_file(myJsonName_1, pofxyandt, 'append2keyASnewlist', key='p')
#     append_to_json_file(myJsonName_1, info, 'append2keyASnewlist', key='info_for_w_and_p')
#     print(" <-- DONE\n")
#
# print("DONE! in " + myJsonName_1)
#
#
# ########################################################################################################################
#
#
#
# if not os.path.isfile('./batch_run.txt'): # We only visualize for runs of specific examples
#
#     from visualization import *
#
#     # loading simulation results
#     time_srs = np.geomspace(0.7, 1000, 8)
#     Fr_list, properties = load_fractures(address='./data/TI_elasticity_ellipse',
#                                                 sim_name='TI_ellasticy_benchmark',
#                                                 time_srs=time_srs)
#     time_srs = get_fracture_variable(Fr_list,
#                                      variable='time')
#
#     Fig_FP = plot_fracture_list(Fr_list,
#                                 variable='mesh',
#                                 projection='2D')
#     Fig_FP = plot_fracture_list(Fr_list,
#                                 variable='footprint',
#                                 projection='2D',
#                                 fig=Fig_FP)
#     Fig_FP = plot_analytical_solution('E_E',
#                                       'footprint',
#                                       Solid,
#                                       Injection,
#                                       fluid_prop=Fluid,
#                                       fig=Fig_FP,
#                                       projection='2D',
#                                       time_srs=time_srs,
#                                       gamma=gamma)
#
#     ext_pnts = np.empty((2, 2), dtype=np.float64)
#     Fig_w_slice = plot_fracture_list_slice(Fr_list,
#                                            variable='width',
#                                            plot_cell_center=True,
#                                            orientation='horizontal',
#                                            extreme_points=ext_pnts)
#     Fig_w_slice = plot_analytical_solution_slice('E_E',
#                                                  variable='width',
#                                                  mat_prop=Solid,
#                                                  inj_prop=Injection,
#                                                  fluid_prop=Fluid,
#                                                  fig=Fig_w_slice,
#                                                  point1=ext_pnts[0],
#                                                  point2=ext_pnts[1],
#                                                  time_srs=time_srs,
#                                                  gamma=gamma)
#
#     # plotting slice
#     Fr_list, properties = load_fractures(address='./data/TI_elasticity_ellipse',
#                                                 sim_name='TI_ellasticy_benchmark')
#     time_srs = get_fracture_variable(Fr_list,
#                                      variable='time')
#     plot_prop = PlotProperties(line_style='.',
#                                graph_scaling='loglog')
#
#     labels = LabelProperties('d_min', 'wm', '1D')
#     labels.figLabel = 'Minor axis length'
#     Fig_len_a = plot_fracture_list(Fr_list,
#                                  variable='d_min',
#                                  plot_prop=plot_prop,
#                                  labels=labels)
#     Fig_len_a = plot_analytical_solution('E_E',
#                                        'd_min',
#                                        Solid,
#                                        Injection,
#                                        fluid_prop=Fluid,
#                                        fig=Fig_len_a,
#                                        time_srs=time_srs,
#                                        gamma=gamma,
#                                        labels=labels)
#
#     labels.figLabel = 'Major axis length'
#     Fig_len_b = plot_fracture_list(Fr_list,
#                                  variable='d_max',
#                                  plot_prop=plot_prop,
#                                  labels=labels)
#     Fig_len_b = plot_analytical_solution('E_E',
#                                        'd_max',
#                                        Solid,
#                                        Injection,
#                                        fluid_prop=Fluid,
#                                        fig=Fig_len_b,
#                                        time_srs=time_srs,
#                                        gamma=gamma,
#                                        labels=labels)
#
#     plt.show(block=True)