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
    notchRadius_m = 10.5e-3
    mesh = CartesianMesh(Lx=notchRadius_m * 5, Ly=notchRadius_m * 5, nx=41, ny=41)

    nu = poissonRatio = 0.29
    youngMod_Pa = 99.7e9
    youngModPlane_Pa = youngMod_Pa / (1 - poissonRatio**2)
    toughness_PaSqrtMeters = 2.79e6  # Fracture toughness (Pa.m^1/2)
    cartersLeakoff_m3perSqrtS = 0.0
    confinementStress_Pa = 5e6

    solid = MaterialProperties(
        Mesh=mesh,
        Eprime=youngModPlane_Pa,
        toughness=toughness_PaSqrtMeters,
        Carters_coef=cartersLeakoff_m3perSqrtS,
        minimum_width=1e-6,
        confining_stress=confinementStress_Pa,
    )

    # Note: For a constant injection use "Qo = your_injection_rate". For injection histories the number of entries in Qo
    # must match with the number of entries in t_change. First entry of Qo is injection for t between the first and second
    # entry of t_change and so on.
    t_change = [0, 13, 23, 73, 123, 173]
    # Qo = [6e-10, 2e-09, 2.7e-09, 2.05e-09, 1.5e-09, 1.1e-09]
    Qo = injectionRate_m3PerSec = 0.08 * 1e-6 / 60
    if type(Qo) == list:
        injectionRate_m3PerSec = np.asarray([t_change, Qo])

    injection = InjectionProperties(rate=injectionRate_m3PerSec, mesh=mesh)

    viscosity = 0.6
    fluid = FluidProperties(viscosity=viscosity, compressibility=0.0)

    simulation = SimulationProperties()
    simulation.finalTime = args.finaltime  # Seconds
    # simulation.saveTSJump, simulation.plotTSJump = 1, 20
    simulation.set_outputFolder(dataPath)
    # simulation.frontAdvancing = "explicit"
    simulation.projMethod = "LS_continousfront"

    # starting gfrom a prexisting fracture with net pressure specified and opening from elasticity
    from solid.elasticity_isotropic import load_isotropic_elasticity_matrix_toepliz

    Eprime = youngModPlane_Pa

    elasticityMatrix = load_isotropic_elasticity_matrix_toepliz(
        mesh,
        youngModPlane_Pa,
        C_precision=np.float64,
        useHMATdot=False,
        nu=poissonRatio,
    )

    Fr_geometry = Geometry("radial", radius=notchRadius_m)
    # init_param = InitializationParameters(Fr_geometry,
    #                                     regime='static',
    #                                     net_pressure=1e3,
    #                                     elasticity_matrix=C)
    # ---------------------------------

    # starting gfrom a prexisting fracture with min opening specified
    # initialState = InitializationParameters(Geometry("radial", radius=notchRadius_m))
    initialWidth_m = 1e-8
    initialNetPressure_Pa = 1e-5
    initialState = InitializationParameters(
        Fr_geometry,
        regime="static",
        net_pressure=initialNetPressure_Pa,
        width=initialWidth_m,
        elasticity_matrix=elasticityMatrix,
    )

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


if plotQ:
    from utilities.visualization import *
    from utilities.postprocess_fracture import load_fractures

    # loading simulation results
    Fr_list, properties = load_fractures(dataPath)
    solid, fluid, injection, simulation = properties
    time_srs = get_fracture_variable(Fr_list, variable='time')                      # list of times

    # plot fracture radius
    plot_prop = PlotProperties()
    plot_prop.lineStyle = '.'               # setting the line style to point
    # plot_prop.graphScaling = 'loglog'       # setting to log log plot
    Fig_R = plot_fracture_list_at_point(Fr_list,
                               variable='pf',
                               plot_prop=plot_prop)
    Fig_R = plot_fracture_list_at_point(Fr_list,
                               variable='pn',
                               plot_prop=plot_prop)
    # Fig_R = plot_fracture_list_at_point(Fr_list,
    #                            variable='ir',
    #                            plot_prop=plot_prop)
    # Fig_R = plot_fracture_list_at_point(Fr_list,
    #                            variable='tir',
    #                            plot_prop=plot_prop)
    # plot analytical radius
    # Fig_R = plot_analytical_solution(regime='M',
    #                                  variable='d_mean',
    #                                  mat_prop=solid,
    #                                  inj_prop=injection,
    #                                  fluid_prop=fluid,
    #                                  time_srs=time_srs,
    #                                  fig=Fig_R)
    # plot analytical radius
    # Fig_R = plot_analytical_solution(regime='K',
    #                                  variable='d_mean',
    #                                  mat_prop=solid,
    #                                  inj_prop=injection,
    #                                  fluid_prop=fluid,
    #                                  time_srs=time_srs,
    #                                  fig=Fig_R)
    plt.show(block=True)


# -------------- exporting to json file -------------- #

# from visualization import *
# from postprocess_fracture import append_to_json_file

# # 1) export general information to json
# # 2) export to json the coordinates of the points defining the fracture front at each time
# # 3) export the fracture opening w(t) versus time (t) at a point of coordinates myX and myY
# # 4) export the fluid pressure p(t) versus time (t) at a point of coordinates myX and myY
# # 5) export w(y) along a vertical line passing through mypoint for different times
# # 6) export pf(x) along a horizontal line passing through mypoint for different times
# # 7) export w(x,y,t) and pf(x,y,t)

# to_export = [1,2,3,4,5,6,7]

# if export_results_to_json:

#     # decide the names of the Json files:
#     myJsonName_1 = "./Data/Pyfrac_"+sim_name+"_export.json"

#     # load the results:
#     print("\n 1) loading results")
#     Fr_list, properties = load_fractures(address=save_folder, sim_name=sim_name, load_all=True)
#     # or Fr_list, properties = load_fractures(address=save_folder, sim_name=sim_name, time_srs=np.linspace(initial_time, final_time, steps))
#     Solid, Fluid, Injection, simulProp = properties
#     print(" <-- DONE\n")

#     # 1) export general information to json
#     if 1 in to_export:
#         print("\n 2) writing general info")
#         time_srs = get_fracture_variable(Fr_list, variable='time')
#         time_srs = np.asarray(time_srs)

#         simul_info = {'Eprime': Solid.Eprime,
#                       'max_KIc': Solid.K1c.max(),
#                       'min_KIc': Solid.K1c.min(),
#                       'max_Sigma0': Solid.SigmaO.max(),
#                       'min_Sigma0': Solid.SigmaO.min(),
#                       'viscosity': Fluid.viscosity,
#                       'total_injection_rate': Injection.injectionRate.max(),
#                       'sources_coordinates_lastFR': Fr_list[-1].mesh.CenterCoor[Injection.sourceElem].tolist(),
#                       't_max': time_srs.max(),
#                       't_min': time_srs.min()}
#         append_to_json_file(myJsonName_1, simul_info, 'append2keyASnewlist', key='simul_info',
#                             delete_existing_filename=True)  # be careful: delete_existing_filename=True only the first time you call "append_to_json_file"

#     # 2) export the coordinates of the points defining the fracture front at each time:
#     if 2 in to_export:
#         print("\n 2) writing fronts")
#         time_srs = get_fracture_variable(Fr_list,variable='time') # get the list of times corresponding to each fracture object
#         append_to_json_file(myJsonName_1, time_srs, 'append2keyASnewlist', key='time_srs_of_Fr_list')
#         fracture_fronts = []
#         numberof_fronts = [] #there might be multiple fracture fronts in general
#         mesh_info = [] # if you do not make remeshing or mesh extension you can export it only once
#         index = 0
#         for fracture in Fr_list:
#             fracture_fronts.append(np.ndarray.tolist(fracture.Ffront))
#             numberof_fronts.append(fracture.number_of_fronts)
#             mesh_info.append([Fr_list[index].mesh.Lx, Fr_list[index].mesh.Ly, Fr_list[index].mesh.nx, Fr_list[index].mesh.ny])
#             index = index + 1
#         append_to_json_file(myJsonName_1, fracture_fronts, 'append2keyASnewlist', key='Fr_list')
#         append_to_json_file(myJsonName_1, numberof_fronts, 'append2keyASnewlist', key='Number_of_fronts')
#         append_to_json_file(myJsonName_1,mesh_info,'append2keyASnewlist', key='mesh_info')
#         print(" <-- DONE\n")

#     # 3) export the fracture opening w(t) versus time (t) at a point of coordinates myX and myY
#     if 3 in to_export:
#         print("\n 3) get w(t) at a point... ")
#         my_X = 0.02 ; my_Y = 0.
#         w_at_my_point, time_list_at_my_point = get_fracture_variable_at_point(Fr_list, variable='w', point=[my_X, my_Y])
#         append_to_json_file(myJsonName_1, w_at_my_point, 'append2keyASnewlist', key='w_at_my_point')
#         append_to_json_file(myJsonName_1, time_list_at_my_point, 'append2keyASnewlist', key='time_list_W_at_my_point')
#         print(" <-- DONE\n")



#     # 4) export the fluid pressure p(t) versus time (t) at a point of coordinates myX and myY
#     if 4 in to_export:
#         print("\n 4) get pf(t) at a point... ")
#         my_X = 0.02 ; my_Y = 0.
#         pf_at_my_point, time_list_at_my_point = get_fracture_variable_at_point(Fr_list, variable='pf', point=[my_X, my_Y])
#         append_to_json_file(myJsonName_1, pf_at_my_point, 'append2keyASnewlist', key='pf_at_my_point_A')
#         append_to_json_file(myJsonName_1, time_list_at_my_point, 'append2keyASnewlist', key='time_list_pf_at_my_point_A')
#         print(" <-- DONE\n")


#     # 5) export w(y) along a vertical line passing through mypoint for different times
#     if 5 in to_export:
#         print("\n 5) get w(y) with y passing through a specific point for different times... ")
#         my_X = 0.; my_Y = 0.
#         ext_pnts = np.empty((2, 2), dtype=np.float64)
#         fracture_list_slice = plot_fracture_list_slice(Fr_list,
#                                                        variable='w',
#                                                        projection='2D',
#                                                        plot_cell_center=True,
#                                                        extreme_points=ext_pnts,
#                                                        orientation='horizontal',
#                                                        point1=[my_X , my_Y],
#                                                        export2Json=True,
#                                                        export2Json_assuming_no_remeshing=False)
#         towrite = {'w_vert_slice_': fracture_list_slice}
#         append_to_json_file(myJsonName_1, towrite, 'extend_dictionary')
#         print(" <-- DONE\n")



#     # 6) export pf(x) along a horizontal line passing through mypoint for different times
#     if 6 in to_export:
#         print("\n 6) get pf(x) with x passing through a specific point for different times... ")
#         my_X = 0.; my_Y = 0.
#         ext_pnts = np.empty((2, 2), dtype=np.float64)
#         fracture_list_slice = plot_fracture_list_slice(Fr_list,
#                                                        variable='pf',
#                                                        projection='2D',
#                                                        plot_cell_center=True,
#                                                        extreme_points=ext_pnts,
#                                                        orientation='horizontal',
#                                                        point1=[my_X , my_Y],
#                                                        export2Json=True,
#                                                        export2Json_assuming_no_remeshing=False)
#         towrite = {'pf_horiz_slice_': fracture_list_slice}
#         append_to_json_file(myJsonName_1, towrite, 'extend_dictionary')
#         print(" <-- DONE\n")



#     # 7) export w(x,y,t) and pf(x,y,t)
#     if 7 in to_export:
#         print("\n 7) get w(x,y,t) and  pf(x,y,t)... ")
#         wofxyandt = []
#         pofxyandt = []
#         info = []
#         jump = True #this is used to jump the first fracture
#         for frac in Fr_list:
#             if not jump:
#                 wofxyandt.append(np.ndarray.tolist(frac.w))
#                 pofxyandt.append(np.ndarray.tolist(frac.pFluid))
#                 info.append([frac.mesh.Lx,frac.mesh.Ly,frac.mesh.nx,frac.mesh.ny,frac.time])
#             else:
#                 jump = False

#         append_to_json_file(myJsonName_1, wofxyandt, 'append2keyASnewlist', key='w')
#         append_to_json_file(myJsonName_1, pofxyandt, 'append2keyASnewlist', key='p')
#         append_to_json_file(myJsonName_1, info, 'append2keyASnewlist', key='info_for_w_and_p')
#         print(" <-- DONE\n")

#     print("DONE! in " + myJsonName_1)
