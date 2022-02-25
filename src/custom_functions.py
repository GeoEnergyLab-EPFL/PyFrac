# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Carlo Peruzzo on Tue Nov 2 15:09:38 2021.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2020.
All rights reserved. See the LICENSE.TXT file for more details.
"""

# External imports
import numpy as np
import matplotlib.pyplot as plt




# # --------------------------------------------------------------
# def get_fracture_sizes(Fr):
#     # Now we are at a given time step.
#     # This function returns the coordinates of the smallest rectangle containing the fracture footprint
#
#     x_min_temp = 0.
#     x_max_temp = 0.
#     y_min_temp = 0.
#     y_max_temp = 0.
#     hx = Fr.mesh.hx; hy = Fr.mesh.hy
#     # loop over the segments defining the fracture front
#     for i in range(Fr.Ffront.shape[0]):
#         segment = Fr.Ffront[i]
#
#         # to find the x_max at this segment:
#         if segment[0] > x_max_temp and np.abs(segment[1])<2.*hy:
#             x_max_temp = segment[0]
#         if segment[2] > x_max_temp and np.abs(segment[3])<2.*hy:
#             x_max_temp = segment[2]
#
#         # to find the n_min at this segment:
#         if segment[0] < x_min_temp and np.abs(segment[1])<2.*hy:
#             x_min_temp = segment[0]
#         if segment[2] < x_min_temp and np.abs(segment[3])<2.*hy:
#             x_min_temp = segment[2]
#
#         # to find the y_max at this segment:
#         if segment[1] > y_max_temp and np.abs(segment[0])<2.*hx:
#             y_max_temp = segment[1]
#         if segment[3] > y_max_temp and np.abs(segment[2])<2.*hx:
#             y_max_temp = segment[3]
#
#         # to find the y_min at this segment:
#         if segment[1] < y_min_temp and np.abs(segment[0])<2.*hx:
#             y_min_temp = segment[1]
#         if segment[3] < y_min_temp and np.abs(segment[2])<2.*hx:
#             y_min_temp = segment[3]
#
#     return x_min_temp, x_max_temp, y_min_temp, y_max_temp
#
#
# def getH__(Ffront):
#     return np.abs(np.max(np.hstack((Ffront[::, 1], Ffront[::, 3]))) - np.min(np.hstack((Ffront[::, 1], Ffront[::, 3]))))
#
# def update_limits(x1, x2, xmax, xmin):
#     #the following function updates x and y
#     if x1 > xmax:
#         xmax = x1
#     if x1 < xmin:
#         xmin = x1
#     if x2 > xmax:
#         xmax = x2
#     if x2 < xmin:
#         xmin = x2
#     return xmax, xmin
#
# def getFfrontBounds__(Ffront):
#     xmax = 0.
#     xmin = 0.
#     ymax = 0.
#     ymin = 0.
#
#     for segment in Ffront:
#         x1, y1, x2, y2 = segment
#         xmax, xmin = update_limits(x1, x2, xmax, xmin)
#         ymax, ymin = update_limits(y1, y2, ymax, ymin)
#     return xmax, xmin, ymax, ymin
#
# def getL__(Ffront):
#     xmax, xmin, ymax, ymin = getFfrontBounds__(Ffront)
#     Lhalf = (np.abs(xmax)+np.abs(xmin))/2.
#     return Lhalf
#
# def getASPECTRATIO__(Ffront):
#     xmax, xmin, ymax, ymin = getFfrontBounds__(Ffront)
#     Lhalf = (np.abs(xmax) + np.abs(xmin))/2.
#     Hhalf = (np.abs(ymax) + np.abs(ymin)) / 2.
#     return Hhalf/Lhalf
#
# def getwAtMovingCenter__(fr):
#     xmax, xmin, ymax, ymin = getFfrontBounds__(fr.Ffront)
#     xmean = 0.5 * (xmax + xmin)
#     ymean = 0.5 * (ymax + ymin)
#     elem_id = fr.mesh.locate_element(xmean, ymean)
#     return fr.w[elem_id]
#
#
# def getwmax__(w):
#     return w.max()
#
# def apply_custom_prop(sim_prop, fr):
#     x_min_n, x_max_n, y_min_n, y_max_n = get_fracture_sizes(fr)
#     sim_prop.tHyst__.append(y_max_n/sim_prop.xlim)
#     #sim_prop.LHyst__.append(getL__(fr.Ffront))
#     #sim_prop.LHyst__.append(getwmax__(fr.w))
#     #sim_prop.LHyst__.append(getwmax__(fr.pFluid))
#     #sim_prop.LHyst__.append(getwAtMovingCenter__(fr))
#     sim_prop.LHyst__.append(x_max_n/sim_prop.xlim)
#
# def custom_plot(sim_prop, fig = None):
#     if fig is None:
#         fig = plt.figure()
#         ax = fig.gca()
#     else:
#         ax = fig.get_axes()[0]
#     # plot L vs time
#     xlabel = 'time [s]'
#     #ylabel = 'H [m]'
#     ylabel = 'L/H [-]'
#     ax.scatter(sim_prop.tHyst__, sim_prop.LHyst__, color='k')
#     # straight line
#     # sl= []
#     # sl_ana = []
#     # K_Ic = 0.5e6
#     # H=2*1.48
#     # p_limit = 1.47*K_Ic/np.sqrt(np.pi*H/2)
#     # p_ana = 2.*K_Ic/np.sqrt(np.pi*H)
#     # for i in range(len(sim_prop.tHyst__)):
#     #     sl.append(p_limit)
#     #     sl_ana.append(p_ana)
#     # ax.plot(sim_prop.tHyst__, sl, color='r')
#     # ax.plot(sim_prop.tHyst__, sl_ana, color='g')
#     ax.set_xlabel(xlabel)
#     ax.set_ylabel(ylabel)
#     ax.set_yscale('log')
#     ax.set_xscale('log')
#     return fig