# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Carlo Peruzzo 2020.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2019. All rights reserved.
See the LICENSE.TXT file for more details.
"""

# local imports
from mesh_obj import CartesianMesh
from properties import MaterialProperties, FluidProperties, InjectionProperties, SimulationProperties
from fracture_obj import Fracture
from controller import Controller
from fracture_initialization import Geometry, InitializationParameters
import numpy as np

run = False
export_results = True

"_______________________________________________________"
# creating mesh
#mesh_discretiz_x=229 # coarse
#mesh_discretiz_y=121 #coarse
mesh_discretiz_x=237 # coarse
mesh_discretiz_y=129 #coarse


Mesh = CartesianMesh(0.039, 0.019, mesh_discretiz_x, mesh_discretiz_y)


# solid properties
nu = 0.48                            # Poisson's ratio
youngs_mod = 125000                  # Young's modulus (+/- 10) #kPa
Eprime = youngs_mod / (1 - nu ** 2)  # plain strain modulus
K_Ic = np.sqrt(2*4.4*Eprime)         # fracture toughness (+/- 1)
Cl = 0.0                             # Carter's leak off coefficient

Solid = MaterialProperties(Mesh,
                           Eprime,
                           K_Ic,
                           Carters_coef=Cl,
                           minimum_width=1e-11)

def source_location(x, y):
    """
    This function is used to evaluate if a point is included in source, i.e. the fluid is injected at the given
    point.
    """
    # the condition
    if (abs(x + .02) < Mesh.hx*0.5 and abs(y - 0) < Mesh.hy*0.5 ) or (abs(x - .02) < Mesh.hx*0.5 and abs(y + 0) < Mesh.hy*0.5):
        return True

def rate_delayed_inj_pt_func(time):
    """
    This function provides the rate vs time at the delayed injection point
    """
    Q0 =  20/1000/60/1000
    tini= 2.6
    if time < tini:
        return 0
    # elif time >tini and time <tini+0.2:
    #    return (time-tini)/0.2*Q0/2
    else:
       return Q0/2

# injection parameters
Q0 =  20/1000/60/1000 #20mL/min  # injection rate total
#initialratesecondpoint=0
#ratesecondpoint=np.asarray([[2.1], [Q0/2]])
delayed_second_injpoint_loc=np.asarray([-0.02,0])
Injection = InjectionProperties(Q0, Mesh, source_loc_func=source_location,
                                #initial_rate_delayed_second_injpoint=initialratesecondpoint,
                                #rate_delayed_second_injpoint=ratesecondpoint,
                                delayed_second_injpoint_loc=delayed_second_injpoint_loc,
                                rate_delayed_inj_pt_func=rate_delayed_inj_pt_func)
# fluid properties
viscosity = 1.13 #Pa.s
Fluid = FluidProperties(viscosity=viscosity)

# simulation properties
simulProp = SimulationProperties()
simulProp.finalTime = 9.8
myfolder ="./Data/coalescence_exp_db12_"+str(mesh_discretiz_x)+"x"+str(mesh_discretiz_y)
simulProp.set_outputFolder(myfolder)          # the address of the output folder
#simulProp.set_solTimeSeries(np.linspace(5.0, 8.0,600))
#simulProp.set_solTimeSeries(np.concatenate((np.linspace(2.9, 2.914,20),np.linspace(2.914, 2.925,20),np.linspace(2.926, 3.30,20),np.linspace(3.40, 5.90,30),np.linspace(6.0, 8.0,25))))
simulProp.set_solTimeSeries(np.concatenate((np.linspace(2.6, 3.2,1000),np.linspace(4.3, 6.0,20),np.linspace(6.3, 8.6,220),np.linspace(8.61, 9.8,70))))
simulProp.maxFrontItrs=35
simulProp.maxSolverItrs=240
simulProp.plotTSJump = 50
simulProp.maximum_steps = 10000

simulProp.timeStepLimit=.2
#simulProp.set_volumeControl(True)
simulProp.saveToDisk=True
simulProp.saveFluidFluxAsVector=True
simulProp.saveFluidVelAsVector=True
simulProp.plotATsolTimeSeries=False
simulProp.plotVar = ['pf']
simulProp.projMethod = 'LS_continousfront'
simulProp.frontAdvancing = 'implicit'
#fixedtimestep=np.array([[2.6,4], [0.0005, 0.005]])
#simulProp.fixedTmStp=fixedtimestep
if run :
    # initializing fracture
    p=np.zeros((Mesh.NumberOfElts,), dtype=np.float64)
    from fracture_initialization import get_radial_survey_cells
    initRad2 = 0.00317542
    #initRad1 = 0.00317542
    initRad1 = 0.000807414
    p1 = 1900
    p2 = 19000
    surv_cells_1, surv_cells_dist_1, inner_cells_1 = get_radial_survey_cells(Mesh, initRad1, inj_point=[-0.02, 0])
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

    surv_cells_2, surv_cells_dist_2, inner_cells_2 = get_radial_survey_cells(Mesh, initRad2, inj_point=[0.02, 0])
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
                                          time=0.89848) #is the initiation time of the first fracture

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
    # from visualization import *
    # Fr_list, properties = load_fractures(address=myfolder)
    # Solid, Fluid, Injection, simulProp = properties
    # plot_prop = PlotProperties()
    # Fig_R = plot_fracture_list(Fr_list[-1],
    #                            variable='footprint',
    #                            plot_prop=plot_prop)

####################
# plotting results #
####################
from visualization import *
from postprocess_fracture import append_to_json_file

# DECIDE THE NAME OF THE OUTPUT JSON FILE
myJsonName = "./Data/coalescence_expdb12_4mma_visc1_237x129.json"

# DECIDE THE NAME OF THE FOLDER
# myfolder="./Data/coalescence_exp_db12_239x121"
# myfolder="./Data/coalescence_exp_db12_249x161"
myfolder = "./Data/coalescence_exp_db12_237x129"

# DECIDE IF EXPORTING TO JSON FILE
write = True

# DECIDE IF PLOTTING TO JSON FILE
plot = False

# DECIDE IF SLICING THE TIME
slicing=False
if slicing:
    time_slicing=np.linspace(0., 9.8,400)
#--------------------------------------
if export_results:
    ########## IMPORT THE COMPLETE SERIES OF FRACTURE FOOTPRINTS ########
    print("\n 1) Loading the fractures ... ")
    Fr_list, properties = load_fractures(address=myfolder)
    Solid, Fluid, Injection, simulProp = properties
    time_srs_COMPLETE = get_fracture_variable(Fr_list,variable='time')

    print("\n 1-bis) Slicing the complete series for time reasons - Loading the fractures ... ")
    slicing_factor=10 #take a fracture every <<slicing_factor>> if t<t_given
    t_given=4.3#s
    print("\n       take a fracture every" + str(slicing_factor) + " if t < " + str(t_given) + " s ... ")
    reduced_time_srs=[]
    for i in range(len(time_srs_COMPLETE)):
        if time_srs_COMPLETE[i]<t_given:
            if i % 10 == 0:
                reduced_time_srs.append(time_srs_COMPLETE[i])
        else:
            reduced_time_srs.append(time_srs_COMPLETE[i])

    print("\n 1-bis) Slicing the complete series for time reasons - Reloading the fractures ... ")
    Fr_list, properties = load_fractures(address=myfolder, time_srs=reduced_time_srs)
    Solid, Fluid, Injection, simulProp = properties
    time_srs_COMPLETE = get_fracture_variable(Fr_list,variable='time')

    if write:
        fracture_fronts = []
        numberof_fronts = []
        for fracture in Fr_list:
            fracture_fronts.append(np.ndarray.tolist(fracture.Ffront))
            numberof_fronts.append(fracture.number_of_fronts)

        complete_footprints = {'time_srs_of_Fr_list': time_srs_COMPLETE,
                               'Fr_list': fracture_fronts,
                               'Number_of_fronts': numberof_fronts
                               }
        print("\n 2) writing the complete footpronts serie to json ... ")
        append_to_json_file(myJsonName, complete_footprints, 'append2keyASnewlist', key='complete_footrints', delete_existing_filename=True)
        append_to_json_file(myJsonName,
                            [Fr_list[-1].mesh.Lx, Fr_list[-1].mesh.Ly, Fr_list[-1].mesh.nx, Fr_list[-1].mesh.ny],
                            'append2keyASnewlist', key='mesh_info')

    # get w at a point
    print("\n 3) get w(t) at a point (fracture left)... ")
    wATcenterL, timewATcenter = get_fracture_variable_at_point(Fr_list, variable='w', point=[-0.02, 0])
    append_to_json_file(myJsonName, wATcenterL, 'append2keyASnewlist', key='wATcenterL')
    append_to_json_file(myJsonName, timewATcenter, 'append2keyASnewlist', key='twATcenter')
    print("\n 4) get w(t) at a point (fracture right)... ")
    wATcenterR, timewATcenter = get_fracture_variable_at_point(Fr_list, variable='w', point=[+0.02, 0])
    append_to_json_file(myJsonName, wATcenterR, 'append2keyASnewlist', key='wATcenterR')

    # get pressure at a point
    # pnATcenter, time_not_used = get_fracture_variable_at_point(Fr_list, variable='pn', point=[0, 0])
    # append_to_json_file(myJsonName, pnATcenter, 'append2keyASnewlist', key='pnATcenter')

    if slicing:
        print("\n 5) loading again the fractures but slicing... ")
        Fr_list, properties = load_fractures(address=myfolder, time_srs=time_slicing)
        Solid, Fluid, Injection, simulProp = properties
        time_srs = get_fracture_variable(Fr_list,variable='time')

        if write:
            fracture_fronts=[]
            numberof_fronts=[]
            for fracture in Fr_list:
                fracture_fronts.append(np.ndarray.tolist(fracture.Ffront))
                numberof_fronts.append(fracture.number_of_fronts)

            selected_footprints = {'time_srs_of_Fr_list': time_srs,
                                   'Fr_list': fracture_fronts,
                                   'Number_of_fronts': numberof_fronts
                                   }
            print("\n 6) writing the sliced fractures... ")
            towrite = {'selected_footrints': selected_footprints}
            append_to_json_file(myJsonName, towrite, 'extend_dictionary')

    if plot:
        # plot fracture radius
        print("\n 7) plotting the fracture footprint... ")
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

    ########## TAKE A VERTICAL SECTION (along y axis) TO GET w AT THE MIDDLE ########
    ext_pnts = np.empty((2, 2), dtype=np.float64)
    print("\n 8) getting the slice along the y axis to get w(y)... ")
    fracture_list_slice = plot_fracture_list_slice(Fr_list,
                                      variable='w',
                                      projection='2D',
                                      plot_cell_center=True,
                                      extreme_points=ext_pnts,
                                      orientation='vertical',
                                      point1=[-0.0007,-0.008],
                                      point2=[-0.0007,0.008],export2Json=True)
                                        #+++
                                        # point1 = [0., -0.018],
                                        # point2 = [0., 0.018], export2Json = True)

    #fracture_list_slice[new_key] = fracture_list_slice[old_key]
    #del fracture_list_slice[old_key]

    if write:
        print("\n 9) writing the slice along the y axis to get w(y)... ")
        fracture_list_slice['Number_of_fronts']=numberof_fronts
        towrite =  {'intersectionVslice':fracture_list_slice}
        append_to_json_file(myJsonName, towrite, 'extend_dictionary')

########## TAKE A HORIZONTAL SECTION  (along x axis)  TO GET w AT THE MIDDLE ########
    ext_pnts = np.empty((2, 2), dtype=np.float64)
    print("\n 10) getting the slice along the x axis to get w(x)... ")
    fracture_list_slice = plot_fracture_list_slice(Fr_list,
                                      variable='w',
                                      projection='2D',
                                      plot_cell_center=True,
                                      extreme_points=ext_pnts,
                                      orientation='horizontal',
                                      point1=[-0.025,0.0],
                                      point2=[0.025,0.],export2Json=True)
                                    #+++
                                    # point1 = [-0.035, 0.0],
                                    # point2 = [0.035, 0.], export2Json = True)

    #fracture_list_slice[new_key] = fracture_list_slice[old_key]
    #del fracture_list_slice[old_key]
    if write:
        print("\n 11) writing the slice along the x axis to get w(x)... ")
        fracture_list_slice['Number_of_fronts']=numberof_fronts
        towrite =  {'intersectionHslice':fracture_list_slice}
        append_to_json_file(myJsonName, towrite, 'extend_dictionary')

    if plot:
        ########## PLOT THE FLUID PRESSURE AT A SECTION ALONG THE x axis ########
        print("\n 12) plot the fluid pressure along the x axis... ")
        ext_pnts = np.empty((2, 2), dtype=np.float64)
        fracture_list_slice = plot_fracture_list_slice(Fr_list,
                                                       variable='pf',
                                                       projection='2D',
                                                       plot_cell_center=True,
                                                       extreme_points=ext_pnts,
                                                       orientation='horizontal',
                                                       point1=[-0.025, 0.0],
                                                       point2=[0.025, 0.], export2Json=True)

    print("\n DONE! in "+myJsonName)
    plt.show(block=True)
