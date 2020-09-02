# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Peruzzo Carlo
Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory, 2016-2019.
All rights reserved. See the LICENSE.TXT file for more details.
"""
import numpy as np
from elastohydrodynamic_solver import calculate_fluid_flow_characteristics_laminar
from postprocess_fracture import get_fracture_variable_slice_cell_center

def get_velocity_as_vector(Solid, Fluid, Fr_list):
    fluid_vel_list = []
    time_srs = []
    for i in Fr_list:

        fluid_flux, \
        fluid_vel, \
        Rey_num, \
        fluid_flux_components, \
        fluid_vel_components = calculate_fluid_flow_characteristics_laminar(i.w,
                                                                            i.pFluid,
                                                                            Solid.SigmaO,
                                                                            i.mesh,
                                                                            i.EltCrack,
                                                                            i.InCrack,
                                                                            Fluid.muPrime,
                                                                            Fluid.density)
        fluid_vel_list.append(fluid_vel_components)
        time_srs.append(i.time)

    return fluid_vel_list, time_srs


def get_velocity_slice(Solid, Fluid, Fr_list, initial_point, vel_direction = 'ux',orientation='horizontal'):
    # initial_point - of the slice
    fluid_vel_list, time_srs = get_velocity_as_vector(Solid, Fluid, Fr_list)

    # fluid_vel_list is a list containing a matrix with the information about the fluid velocity for each of the edges of any mesh element
    # fluid_vel_components_for_one_elem = [fx left edge, fy left edge, fx right edge, fy right edge, fx bottom edge, fy bottom edge, fx top edge, fy top edge]
    #
    #                 6  7
    #               (ux,uy)
    #           o---top edge---o
    #     0  1  |              |    2  3
    #   (ux,uy)left          right(ux,uy)
    #           |              |
    #           o-bottom edge--o
    #               (ux,uy)
    #                 4  5
    #
    vector_to_be_lost = np.zeros(Fr_list[0].mesh.NumberOfElts,dtype=np.int)
    NotUsd_var_values, sampling_line_center, sampling_cells = get_fracture_variable_slice_cell_center(vector_to_be_lost,
                                                                                        Fr_list[0].mesh,
                                                                                        point = initial_point,
                                                                                        orientation = orientation)
    hx = Fr_list[0].mesh.hx
    hy = Fr_list[0].mesh.hy
    if vel_direction ==  'ux' and orientation == 'horizontal': # take ux on the vertical edges
        indx1 = 0 #left
        indx2 = 2 #right
        sampling_line_center1=sampling_line_center-hx*.5
        sampling_line_center2=sampling_line_center+hx*.5
    elif vel_direction ==  'ux' and orientation == 'vertical': # take ux on the horizontal edges
        indx1 = 4 #bottom
        indx2 = 6 #top
        sampling_line_center1=sampling_line_center-hy*.5
        sampling_line_center2=sampling_line_center+hy*.5
    elif vel_direction == 'uy' and orientation == 'horizontal': # take uy on the vertical edges
        indx1 = 1 #left
        indx2 = 3 #rigt
        sampling_line_center1=sampling_line_center-hx*.5
        sampling_line_center2=sampling_line_center+hx*.5
    elif vel_direction == 'uy' and orientation == 'vertical': # take uy on the horizontal edges
        indx1 = 5 #bottom
        indx2 = 7 #top
        sampling_line_center1=sampling_line_center-hy*.5
        sampling_line_center2=sampling_line_center+hy*.5

    sampling_line = [None] * (len(sampling_line_center1) + len(sampling_line_center2))
    sampling_line[::2] = sampling_line_center1
    sampling_line[1::2] = sampling_line_center2

    fluid_vel_list_final = []
    for i in range(len(time_srs)):
        EltCrack_i = Fr_list[i].EltCrack
        fluid_vel_list_i=fluid_vel_list[i]

        vector_to_be_lost1  = np.zeros(Fr_list[0].mesh.NumberOfElts, dtype=np.float)
        vector_to_be_lost1[EltCrack_i] = fluid_vel_list_i[indx1,:]
        vector_to_be_lost2 = np.zeros(Fr_list[0].mesh.NumberOfElts, dtype=np.float)
        vector_to_be_lost2[EltCrack_i] = fluid_vel_list_i[indx2,:]

        fluid_vel_list_final_i = [None] * (len(vector_to_be_lost1[sampling_cells]) + len(vector_to_be_lost2[sampling_cells]))
        fluid_vel_list_final_i[::2] = vector_to_be_lost1[sampling_cells]
        fluid_vel_list_final_i[1::2] = vector_to_be_lost2[sampling_cells]
        fluid_vel_list_final.append(fluid_vel_list_final_i)

    return  fluid_vel_list_final, time_srs, sampling_line