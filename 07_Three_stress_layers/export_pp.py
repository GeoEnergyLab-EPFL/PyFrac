# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Carlo Peruzzo on Fri Apr 17 23:16:25 2020.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2020.
All rights reserved. See the LICENSE.TXT file for more details.
"""
from utilities.visualization import *
from utilities.postprocess_fracture import append_to_json_file, load_fractures
from solid.solid_prop import MaterialProperties

####### simulation dependent piece of code #####
### to be used every time you have done remeshing
remeshing = False
def sigmaO_func(x, y):
    """ The function providing the confining stress"""
    if np.abs(y) > 47.:
        return 3.2e6
    else:
        return 2.2e6
################################################

def getFrBounds(Fr):
    ymax = 0.
    ymin = 0.

    xmax = 0.
    xmin = 0.

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

        if x1 > xmax:
            xmax = x1
        if x1 < xmin:
            xmin = x1
        if x2 > xmax:
            xmax = x2
        if x2 < xmin:
            xmin = x2
        L = (xmax - xmin)/2.
        H = (ymax - ymin) / 2.
    return L, H


def getdimK(t,Eprime,Q,muPrime,Kprime):
    return Kprime * (t**2 / (muPrime**5 * Q**3 * Eprime**13))**(1/18)

def getPayzoneHeight(mesh, solid):
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
    pos0 = None
    pos1 = None
    for i in range(len(mySigma)-1):
        currentSigma = mySigma[i]
        nextSigma = mySigma[i+1]
        if currentSigma != nextSigma:
            if pos0 == None:
                pos0 = (myYcoords[i+1] + myYcoords[i])/2.
            else:
                pos1 = (myYcoords[i + 1] + myYcoords[i]) / 2.
    return np.abs(pos1 - pos0)/2.

def export(simulation_name, myfolder):
    ####################################
    # exporting to multiple json files #
    ####################################

    export_results = True
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

        # 1) decide the names of the Json files:
        if 1 in to_export:
            myJsonName_1 = "./Data/SJ_"+simulation_name+"_export.json"           # I will export here most of the infos
            myJsonName_2 = "./Data/SJ_VEL_as_vector"+simulation_name+"_export.json"        # I will export here the velocity infos




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
            timetouch = 0
            payZoneHeight = 0
            stressJump_val = 0
            pNetCenter_ttouch = 0
            wCenter_ttouch = 0
            dimK_ttouch = 0
            frHeight = []
            frLength = []
            for fr in Fr_list:
                # ---- 0 ----
                L, H = getFrBounds(fr)
                frHeight.append(H)
                frLength.append(L)

                # ---- 1 ----
                # get ribbon element
                EltTip = fr.EltTip
                if remeshing == True:
                    Solid = MaterialProperties(fr.mesh,
                                               Solid.Eprime,
                                               Solid.K1c.max(),
                                               0.,
                                               confining_stress_func=sigmaO_func,
                                               minimum_width=1.e-8)
                # get center coord at tip
                EltTip_Sigvalues = Solid.SigmaO[EltTip]

                # if in all the ribbon KIc is constant we did not touch the heterogeneity
                EltTip_Sigvalues = np.asarray(EltTip_Sigvalues)
                if EltTip_Sigvalues.max() != EltTip_Sigvalues.min() and timetouch is 0:
                    timetouch = fr.time
                    stressJump_val = np.abs(EltTip_Sigvalues.max() - EltTip_Sigvalues.min())
                    payZoneHeight = getPayzoneHeight(fr.mesh, Solid)
                    pNetCenter_ttouch = fr.pNet[fr.mesh.locate_element(0., 0.)][0]
                    wCenter_ttouch = fr.w[fr.mesh.locate_element(0., 0.)][0]
                    dimK_ttouch = getdimK(timetouch,Solid.Eprime, Injection.injectionRate.max(), Fluid.muPrime,Solid.Kprime.max())


            simul_info = {'Eprime': Solid.Eprime,
                          'max_KIc': Solid.K1c.max(),
                          'min_KIc': Solid.K1c.min(),
                          'max_Sigma0': Solid.SigmaO.max(),
                          'min_Sigma0': Solid.SigmaO.min(),
                          'stressJump_val' : stressJump_val,
                          'viscosity': Fluid.viscosity,
                          'total_injection_rate': Injection.injectionRate.max(),
                          'sources_coordinates_lastFR': Fr_list[-1].mesh.CenterCoor[Injection.sourceElem].tolist(),
                          't_max': time_srs.max(),
                          't_min': time_srs.min(),
                          't_touching_interface': timetouch,
                          'payZoneHeight': payZoneHeight,
                          'pNetCenter_ttouch': pNetCenter_ttouch,
                          'wCenter_ttouch': wCenter_ttouch,
                          'dimK_ttouch' : dimK_ttouch,
                          'frHeight' : frHeight,
                          'frLength' : frLength}

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
            my_X = 0.0 ; my_Y = 0.
            w_at_my_point, time_list_at_my_point = get_fracture_variable_at_point(Fr_list, variable='w', point=[my_X, my_Y])
            append_to_json_file(myJsonName_1, w_at_my_point, 'append2keyASnewlist', key='w_at_my_point')
            append_to_json_file(myJsonName_1, time_list_at_my_point, 'append2keyASnewlist', key='time_list_W_at_my_point')
            print(" <-- DONE\n")



        # 5) get the fluid pressure p(t) versus time (t) at a point of coordinates myX and myY
        if 5 in to_export:
            print("\n 4) get pn(t) at a point... ")
            my_X = 0. ; my_Y = 0.
            pn_at_my_point, time_list_at_my_point = get_fracture_variable_at_point(Fr_list, variable='pn', point=[my_X, my_Y])
            append_to_json_file(myJsonName_1, pn_at_my_point, 'append2keyASnewlist', key='pn_at_my_point_A')
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
        # print("DONE! in " + myJsonName_2)
        plt.show(block=True)



### export multiple files

#for simID in ["01","02","03","04","05","06","07","08","09","10","11","12","14","15","16","17","18","19","20","21"]:
#for simID in ["10", "16", "17","18","20","21"]:
for simID in ["07"]:
    simulation_name = "B"+str(simID)
    myfolder = "./Data/"+simulation_name
    export(simulation_name, myfolder)