# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on Wed Apr 19 11:07:38 2017.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2017. All rights reserved.
See the LICENSE.TXT file for more details.
"""

import sys
if "win32" in sys.platform or "win64" in sys.platform:
    slash = "\\"
else:
    slash = "/"
if not '..' + slash + 'src' in sys.path:
    sys.path.append('.' + slash + 'src')
if not '.' + slash + 'src' in sys.path:
    sys.path.append('.' + slash + 'src')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from src.CartesianMesh import *
from src.Fracture import *
from src.VolIntegral import *
from src.Properties import *
from src.Utility import *


def plot_radius(address, r_type='mean', fig_r=None, sol_t_srs=None, time_period=0.,
                      loglog=True, plt_symbol='o', analytical_sol='E', anltcl_clr='k'):

    if not slash in address[-2:]:
        address = address + slash

    # read properties
    filename = address + "properties"
    try:
        with open(filename, 'rb') as input:
            (Solid, Fluid, Injection, SimulProp) = pickle.load(input)
    except FileNotFoundError:
        raise SystemExit("Data not found. The address might be incorrect")
    except AttributeError:
        # todo: get the serialised anisotropic function
        # import marshal
        # code = marshal.loads(Solid.KpFunString)
        import __main__
        setattr(__main__, 'Kprime_function', None)
        with open(filename, 'rb') as input:
            (Solid, Fluid, Injection, SimulProp) = pickle.load(input)

    fileNo = 0
    nxt_plt_t = 0.0
    t_srs_indx = 0
    t_srs_given = isinstance(sol_t_srs, np.ndarray)
    if t_srs_given:
        nxt_plt_t = sol_t_srs[t_srs_indx]

    r_numrcl = np.asarray([])
    r_anltcl = np.asarray([])
    time_srs = np.asarray([])
    
    while fileNo < 5000:

        # trying to load next file. exit loop if not found
        try:
            ff = ReadFracture(address + "file_" + repr(fileNo))
        except FileNotFoundError:
            break
        fileNo+=1

        if ff.time - nxt_plt_t > -1e-8:
            # if the current fracture time has advanced the output time period

            time_srs = np.append(time_srs, ff.time)
            tipVrtxCoord = ff.mesh.VertexCoor[ff.mesh.Connectivity[ff.EltTip, ff.ZeroVertex]]
            if r_type is 'mean': 
                r_numrcl = np.append(r_numrcl, np.mean((tipVrtxCoord[:, 0] ** 2 + tipVrtxCoord[:, 1] ** 2) ** 0.5
                                                        + ff.l))
            elif r_type is 'max':
                r_numrcl = np.append(r_numrcl, max((tipVrtxCoord[:, 0] ** 2 + tipVrtxCoord[:, 1] ** 2) ** 0.5
                                                       + ff.l))
            elif r_type is 'min':
                r_numrcl = np.append(r_numrcl, min((tipVrtxCoord[:, 0] ** 2 + tipVrtxCoord[:, 1] ** 2) ** 0.5
                                                       + ff.l))

            if not analytical_sol is 'n':
                if analytical_sol in ('M', 'Mt', 'K', 'Kt', 'E'):  # radial fracture
                    t, R, p, w, v, actvElts = HF_analytical_sol(analytical_sol,
                                                                ff.mesh,
                                                                Solid.Eprime,
                                                                Injection.injectionRate[1, 0],
                                                                muPrime=Fluid.muPrime,
                                                                Kprime=Solid.Kprime[ff.mesh.CenterElts],
                                                                Cprime=Solid.Cprime[ff.mesh.CenterElts],
                                                                t=ff.time,
                                                                KIc_min=Solid.K1c_perp)
                elif analytical_sol == 'PKN':
                    print("PKN is to be implemented.")
                else:
                    raise ValueError("Provided analytical solution is not supported")
                
                r_anltcl = np.append(r_anltcl, R)

            if t_srs_given:
                if t_srs_indx < len(sol_t_srs) - 1:
                    t_srs_indx += 1
                    nxt_plt_t = sol_t_srs[t_srs_indx]
                if ff.time > max(sol_t_srs):
                    break
            else:
                nxt_plt_t = ff.time + time_period


    print(repr(time_srs))
    if fig_r is None:
        fig_r = plt.figure()
        ax = fig_r.add_subplot(111)
    ax = fig_r.add_subplot(111)
    if loglog:
        if not analytical_sol is 'n':
            ax.semilogx(time_srs, r_anltcl, anltcl_clr)
            # ax.loglog(time_srs, a_anltcl, anltcl_clr)
        ax.semilogx(time_srs, r_numrcl, plt_symbol)
        # ax.loglog(time_srs, a_numrcl, plt_symbol)
    else:
        if not analytical_sol is 'n':
            ax.plot(time_srs, r_anltcl, anltcl_clr)
        ax.plot(time_srs, r_numrcl, plt_symbol)

    # plt_symbol=plt_symbol[0]+'^'
    # plt.ylabel('axis length')
    # plt.xlabel('time')
    
    return fig_r

#-----------------------------------------------------------------------------------------------------------------------

def plot_ellipse_data(address, fig_ab=None, fig_asp_rtio=None, sol_t_srs=None, time_period=0., maxFiles=1500,
                      loglog=True, plt_symbol='o', plt_anltcl=True, anltcl_sol='E', anltcl_clr='k', plt_tmk2=False):

    if not slash in address[-2:]:
        address = address + slash

    # read properties
    filename = address + "properties"
    try:
        with open(filename, 'rb') as input:
            (Solid, Fluid, Injection, SimulProp) = pickle.load(input)
    except FileNotFoundError:
        raise SystemExit("Data not found. The address might be incorrect")


    fileNo = 0
    nxt_plt_t = 0.0
    t_srs_indx = 0
    t_srs_given = isinstance(sol_t_srs, np.ndarray)
    if t_srs_given:
        nxt_plt_t = sol_t_srs[t_srs_indx]

    a_numrcl = np.asarray([])
    b_numrcl = np.asarray([])
    a_anltcl = np.asarray([])
    b_anltcl = np.asarray([])
    time_srs = np.asarray([])
    while fileNo < maxFiles:

        # trying to load next file. exit loop if not found
        try:
            ff = ReadFracture(address + "file_" + repr(fileNo))
        except FileNotFoundError:
            break
        fileNo+=1

        if ff.time - nxt_plt_t > -1e-8:
            # if the current fracture time has advanced the output time period

            # at_x_axis = np.where(ff.mesh.CenterCoor[ff.EltTip,1]==0)
            time_srs = np.append(time_srs, ff.time)
            tipVrtxCoord = ff.mesh.VertexCoor[ff.mesh.Connectivity[ff.EltTip, ff.ZeroVertex]]
            a_numrcl = np.append(a_numrcl, max((tipVrtxCoord[:, 0] ** 2 + tipVrtxCoord[:, 1] ** 2) ** 0.5 + ff.l))
            b_numrcl = np.append(b_numrcl, min((tipVrtxCoord[:, 0] ** 2 + tipVrtxCoord[:, 1] ** 2) ** 0.5 + ff.l))


            if anltcl_sol=='E':
                b, a, w, p = anisotropic_toughness_elliptical_solution(Solid.K1c,
                                                                   Solid.K1c_perp,
                                                                   Solid.Eprime,
                                                                   Injection.injectionRate[1, 0],
                                                                   ff.mesh,
                                                                   t=ff.time)
                a_anltcl = np.append(a_anltcl, a)
                b_anltcl = np.append(b_anltcl, b)

            elif anltcl_sol=='M':
                R, p, w, v = M_vertex_solution_t_given(Solid.Eprime,
                                                       Injection.injectionRate[1, 0],
                                                       Fluid.muPrime,
                                                       ff.mesh,
                                                       ff.time)

                a_anltcl = np.append(a_anltcl, R)
                b_anltcl = np.append(b_anltcl, R)

            elif anltcl_sol == "K":
                R, p, w, v = K_vertex_solution_t_given(Solid.Kprime,
                                                       Solid.Eprime,
                                                       Injection.injectionRate[1, 0],
                                                       ff.mesh,
                                                       ff.time)
                a_anltcl = np.append(a_anltcl, R)
                b_anltcl = np.append(b_anltcl, R)

            if t_srs_given:
                if t_srs_indx < len(sol_t_srs) - 1:
                    t_srs_indx += 1
                    nxt_plt_t = sol_t_srs[t_srs_indx]
                if ff.time > max(sol_t_srs):
                    break
            else:
                nxt_plt_t = ff.time + time_period

    # tmk = (Solid.Eprime**13 * Fluid.muPrime**5 * Injection.injectionRate[1, 0]**3 / ((32 / math.pi) ** 0.5* Solid.K1c_perp)**18)**0.5
    # tmk2 = (Solid.Eprime**13 * Fluid.muPrime**5 * Injection.injectionRate[1, 0]**3 / ((32 / math.pi) ** 0.5* Solid.K1c[0])**18)**0.5
    # ratio = tmk/tmk2
    # lmk = Solid.Eprime**3 * Fluid.muPrime * Injection.injectionRate[1, 0] / ((32 / math.pi) ** 0.5* Solid.K1c[0])**4
    # b_anltcl = b_anltcl/lmk
    # b_numrcl = b_numrcl/lmk
    # time_srs = time_srs/tmk2
    print(repr(time_srs))
    if fig_ab is None:
        fig_ab = plt.figure()
        ax = fig_ab.add_subplot(111)
    ax = fig_ab.add_subplot(111)
    if loglog:
        if plt_anltcl:
            ax.semilogx(time_srs, a_anltcl, anltcl_clr)
            # ax.loglog(time_srs, a_anltcl, anltcl_clr)
        ax.semilogx(time_srs, a_numrcl, plt_symbol)
        # ax.loglog(time_srs, a_numrcl, plt_symbol)
    else:
        if plt_anltcl:
            ax.plot(time_srs, a_anltcl, anltcl_clr)
        ax.plot(time_srs, a_numrcl, plt_symbol)

    # plt_symbol=plt_symbol[0]+'^'
    if loglog:
        if plt_anltcl:
            ax.semilogx(time_srs, b_anltcl, anltcl_clr)
            # ax.loglog(time_srs, b_anltcl, anltcl_clr)
        ax.semilogx(time_srs, b_numrcl, plt_symbol)
        # ax.loglog(time_srs, b_numrcl, plt_symbol)
    else:
        if plt_anltcl:
            ax.plot(time_srs, b_anltcl, anltcl_clr)
        ax.plot(time_srs, b_numrcl, plt_symbol)
    # plt.ylabel('axis length')
    # plt.xlabel('time')
    if plt_tmk2:
        ax.plot(tmk2/tmk, 0.1, 'k.')
        ax.plot(7000*tmk2 / tmk, 0.1, 'k.')
        ax.plot(7000, 0.1, 'k.')
    if fig_asp_rtio is None:
        fig_asp_rtio = plt.figure()
        ax_asp_rtio = fig_asp_rtio.add_subplot(111)
    ax_asp_rtio = fig_asp_rtio.add_subplot(111)
    # ax_asp_rtio.plot(time_srs, a_anltcl / b_anltcl)
    # ax_asp_rtio.semilogx(time_srs, 4/(a_numrcl / b_numrcl), plt_symbol, ms=3)
    ax_asp_rtio.semilogx(time_srs, (a_anltcl - a_numrcl)/a_anltcl, plt_symbol)
    plt_symbol='b.'
    ax_asp_rtio.semilogx(time_srs, (b_anltcl - b_numrcl)/a_anltcl, plt_symbol)

    # plt.ylabel('aspect ratio')
    # plt.xlabel('time')
    return fig_ab, fig_asp_rtio

#-----------------------------------------------------------------------------------------------------------------------


def plot_leak_off(address, fig_lkOff=None, fig_efficiency=None, sol_t_srs=None, time_period=0., maxFiles=1500,
                      loglog=True, plt_symbol='o'):

    if not slash in address[-2:]:
        address = address + slash

    # read properties
    filename = address + "properties"
    try:
        with open(filename, 'rb') as input:
            (Solid, Fluid, Injection, SimulProp) = pickle.load(input)
    except FileNotFoundError:
        raise SystemExit("Data not found. The address might be incorrect")


    fileNo = 0
    nxt_plt_t = 0.0
    t_srs_indx = 0
    t_srs_given = isinstance(sol_t_srs, np.ndarray)
    if t_srs_given:
        nxt_plt_t = sol_t_srs[t_srs_indx]

    time_srs = np.asarray([])
    lk_off = np.asarray([])
    efficiency = np.asarray([])
    while fileNo < maxFiles:

        # trying to load next file. exit loop if not found
        try:
            ff = ReadFracture(address + "file_" + repr(fileNo))
        except FileNotFoundError:
            break
        fileNo+=1

        if ff.time - nxt_plt_t > -1e-8:
            # if the current fracture time has advanced the output time period

            # at_x_axis = np.where(ff.mesh.CenterCoor[ff.EltTip,1]==0)
            time_srs = np.append(time_srs, ff.time)
            lk_off = np.append(lk_off, sum(ff.LkOff_vol))
            efficiency = np.append(efficiency, ff.efficiency)


            if t_srs_given:
                if t_srs_indx < len(sol_t_srs) - 1:
                    t_srs_indx += 1
                    nxt_plt_t = sol_t_srs[t_srs_indx]
                if ff.time > max(sol_t_srs):
                    break
            else:
                nxt_plt_t = ff.time + time_period

    # tmk = (Solid.Eprime**13 * Fluid.muPrime**5 * Injection.injectionRate[1, 0]**3 / ((32 / math.pi) ** 0.5* Solid.K1c_perp)**18)**0.5
    # tmk2 = (Solid.Eprime**13 * Fluid.muPrime**5 * Injection.injectionRate[1, 0]**3 / ((32 / math.pi) ** 0.5* Solid.K1c[0])**18)**0.5
    # ratio = tmk/tmk2
    # lmk = Solid.Eprime**3 * Fluid.muPrime * Injection.injectionRate[1, 0] / ((32 / math.pi) ** 0.5* Solid.K1c[0])**4
    # time_srs = time_srs/tmk2

    if fig_lkOff is None:
        fig_lkOff = plt.figure()
        ax_lkOff = fig_lkOff.add_subplot(111)
    ax_lkOff = fig_lkOff.add_subplot(111)
    if loglog:
        ax_lkOff.semilogx(time_srs, lk_off, plt_symbol)
        # ax.loglog(time_srs, a_numrcl, plt_symbol)
    else:
        ax_lkOff.plot(time_srs, lk_off, plt_symbol)

    plt.ylabel('leaked off volume')
    plt.xlabel('time')

    if fig_efficiency is None:
        fig_efficiency = plt.figure()
        ax_efficiency = fig_efficiency.add_subplot(111)
    ax_efficiency = fig_efficiency.add_subplot(111)
    if loglog:
        ax_efficiency.semilogx(time_srs, efficiency, plt_symbol)
    plt.ylabel('fracture efficiency')
    plt.xlabel('time')
    return fig_lkOff, fig_efficiency