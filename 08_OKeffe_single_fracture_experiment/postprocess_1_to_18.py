# -*- coding: utf-8 -*-
"""
Created by Carlo Peruzzo.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2019.
All rights reserved. See the LICENSE.TXT file for more details.
"""

import numpy as np

# local imports
from mesh_obj import CartesianMesh
from properties import MaterialProperties, FluidProperties, InjectionProperties, SimulationProperties
from fracture_obj import Fracture
from controller import Controller
from fracture_initialization import Geometry, InitializationParameters

####################
# plotting results #
####################
catchthetimeonly = False

from visualization import *
from postprocess_fracture import append_to_json_file

def getHSlice(Fr_list):
    # this function gets w given an horizontal line passing for y=0 and for a given fracture
    ext_pnts = np.empty((2, 2), dtype=np.float64)
    fracture_list_slice = plot_fracture_list_slice(Fr_list,
                                                   variable='w',
                                                   projection='2D',
                                                   plot_cell_center=True,
                                                   extreme_points=ext_pnts,
                                                   orientation='horizontal',
                                                   point1=[0.0, 0.0],
                                                   export2Json=True)
    return fracture_list_slice

def getSETofSlices(myR,R_list,Fr_list,catchthetimeonly,expindx):
    # this function is needed because we need both to find the time (t*) where the diameter is equal to myR and we need to find
    # also the time indexes of the subsequent 6 times later than (t*) and far from each other of about 4 s
    # then we need to take a section of the fracture to get w(x) at these times
    myindexes = []
    # find the time correstponding to the closest radious to myR
    minlist = np.abs([x - myR for x in R_list])
    timeindex = np.where(minlist == np.amin(minlist))
    myindexes.append(timeindex[0][0])
    if catchthetimeonly:
        print("\n exp"+str(expindx))
        print("\n RnumPREVIOUS = " + str(R_list[timeindex[0][0] - 1]) + " at time = " + str(
            time_srs[timeindex[0][0] - 1]))
        print("\n Rnum = " + str(R_list[timeindex[0][0]]) + " ~ R = "+str(myR)+" m at time = " + str(
            time_srs[timeindex[0][0]]))
        print("\n RnumNEXT = " + str(R_list[timeindex[0][0] + 1]) + " at time = " + str(time_srs[timeindex[0][0] + 1]))
    for j in range(1, 7, 1):
        mytime = time_srs[timeindex[0][0]] + j * 4
        minlist = np.abs([x - mytime for x in time_srs])
        timeindexnext = np.where(minlist == np.amin(minlist))
        myindexes.append(timeindexnext[0][0])
        if catchthetimeonly:
            print("\n next time t = " + str(time_srs[timeindexnext[0][0]]) + " ~ time needed = " + str(
                mytime) + " diff " + str(time_srs[timeindexnext[0][0]] - mytime) + "s")
    if not catchthetimeonly:
        Hslicelist = []
        for x in myindexes:
           Hslicelist.append(getHSlice([Fr_list[x]] ))
        return Hslicelist
    else: return True

# DECIDE THE NAME OF THE OUTPUT JSON FILE
pathtofile= "/home/carlo/Desktop/PyFrac/Paper_OKeffe_single_fracture/"
FILENAME= "single_fracture_data_experiments_1to18.json"

myJsonName = pathtofile+FILENAME
foldername=[]
for i in range(1,19,1):
    if i<10:
        string="./Data/Exp_0"+str(i)+"_single_fracture"
    else:
        string = "./Data/Exp_" + str(i) + "_single_fracture"
    foldername.append(string)

#special requests for experiments  8  15: w(x,y)
#special requests for experiments 10  16: w(x/R)
#experiments from 1 to 18 get the average radius

if not catchthetimeonly:
    notes = 'this file contains results from numerical simulations of 18 experiments. All simulations involved remeshing'
    print("\n 2) writing the complete footpronts serie to json ... ")
    append_to_json_file(myJsonName, notes, 'append2keyASnewlist', key='notes',delete_existing_filename=True)

simul_indx=1
GlobalList_of_R=[]
GlobalList_of_t=[]
for path in foldername:

    # loading simulation results
    Fr_list, properties = load_fractures(path)
    Solid, Fluid, Injection, simulProp = properties

    time_srs = get_fracture_variable(Fr_list, variable='time')                      # list of times
    GlobalList_of_t.append(time_srs)

    # compute the mean radius
    R_list=[]
    for i in Fr_list:
        front_intersect_dist = np.sqrt((i.Ffront[::, [0, 2]].flatten() - 0.) ** 2
                                       + (i.Ffront[::, [1, 3]].flatten() - 0.) ** 2)
        Rmean = float(np.mean(front_intersect_dist))
        R_list.append(Rmean)

        # decide if we neet to take the w(x) in case of experiments

    GlobalList_of_R.append(R_list)
    if not catchthetimeonly:
        # 1st special case: export w(x,y)
        if simul_indx == 8 or simul_indx==15:
            for frac in Fr_list:
                if simul_indx==8 and frac.time == 39.4:
                    print("\n - get w(x,y,t) for simulation " + str(simul_indx) + " t = 39.4 s")
                    CenterCoor = np.ndarray.tolist(frac.mesh.CenterCoor)
                    wofxyandt = np.ndarray.tolist(frac.w)
                    footprint = np.ndarray.tolist(frac.Ffront)
                    append_to_json_file(myJsonName, CenterCoor, 'append2keyASnewlist', key='center_coor_exp8_t39p4')
                    append_to_json_file(myJsonName, wofxyandt, 'append2keyASnewlist', key='w_exp8_t39p4')
                    append_to_json_file(myJsonName, footprint, 'append2keyASnewlist', key='footprint_exp8_t39p4')
                elif simul_indx == 15 and frac.time == 40.6:
                    print("\n - get w(x,y,t) for simulation " + str(simul_indx) + " t = 40.6 s")
                    CenterCoor = np.ndarray.tolist(frac.mesh.CenterCoor)
                    wofxyandt = np.ndarray.tolist(frac.w)
                    footprint = np.ndarray.tolist(frac.Ffront)
                    append_to_json_file(myJsonName, CenterCoor, 'append2keyASnewlist', key='center_coor_exp15_t40p6')
                    append_to_json_file(myJsonName, wofxyandt, 'append2keyASnewlist', key='w_exp15_t40p6')
                    append_to_json_file(myJsonName, footprint, 'append2keyASnewlist', key='footprint_exp15_t40p6')

        # 2nd special case: export w(x) - TAKE A HORIZONTAL SECTION  (along x axis)  TO GET w AT THE MIDDLE
        if simul_indx == 10 or simul_indx==16:
            for frac in Fr_list:
                if simul_indx == 10 and frac.time == 14:
                    print("\n - get w(x,t) for simulation " + str(simul_indx) + " t = 14. s")
                    fracture_list_slice = getHSlice([frac])
                    towrite = {'wofx_exp10_t_14': fracture_list_slice}
                    append_to_json_file(myJsonName, towrite, 'extend_dictionary')
                elif simul_indx == 10 and frac.time == 48:
                    print("\n - get w(x,t) for simulation " + str(simul_indx) + " t = 48. s")
                    fracture_list_slice = getHSlice([frac])
                    towrite = {'wofx_exp10_t_48': fracture_list_slice}
                    append_to_json_file(myJsonName, towrite, 'extend_dictionary')
                elif simul_indx == 16 and frac.time == 0.76:
                    print("\n - get w(x,t) for simulation " + str(simul_indx) + " t = 0.76 s")
                    fracture_list_slice = getHSlice([frac])
                    towrite = {'wofx_exp16_t_0p76': fracture_list_slice}
                    append_to_json_file(myJsonName, towrite, 'extend_dictionary')
                elif simul_indx == 16 and frac.time == 27.5:
                    print("\n - get w(x,t) for simulation " + str(simul_indx) + " t = 27.5 s")
                    fracture_list_slice = getHSlice([frac])
                    towrite = {'wofx_exp16_t_27p5': fracture_list_slice}
                    append_to_json_file(myJsonName, towrite, 'extend_dictionary')


    # 3nd special case: export w(x) at different times - TAKE A HORIZONTAL SECTION  (along x axis)  TO GET w AT THE MIDDLE
    if simul_indx == 9:
        if not catchthetimeonly:
            fracture_list_slice = getSETofSlices(0.0207357,R_list,Fr_list,catchthetimeonly,9)
            towrite = {'wofx_exp9_varioust': fracture_list_slice}
            append_to_json_file(myJsonName, towrite, 'extend_dictionary')
        else:
            getSETofSlices(0.0207357,R_list,Fr_list,catchthetimeonly,9)

    elif simul_indx==16:
        if not catchthetimeonly:
            fracture_list_slice = getSETofSlices(0.01770,R_list,Fr_list,catchthetimeonly,16)
            towrite = {'wofx_exp16_varioust': fracture_list_slice}
            append_to_json_file(myJsonName, towrite, 'extend_dictionary')
        else:
            getSETofSlices(0.01770,R_list,Fr_list,catchthetimeonly,16)

    #update the simulation index for the next step
    simul_indx=simul_indx+1

if not catchthetimeonly:
    append_to_json_file(myJsonName, GlobalList_of_R, 'append2keyASnewlist', key='R_lists_1_to_18')
    append_to_json_file(myJsonName, GlobalList_of_t, 'append2keyASnewlist', key='t_lists_1_to_18')