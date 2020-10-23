# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on 04.03.18.
Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory, 2016-2020.
All rights reserved. See the LICENSE.TXT file for more details.
"""

import sys
if "win32" in sys.platform or "win64" in sys.platform:
    slash = "\\"
else:
    slash = "/"


import dill
import numpy as np
import logging
import re
import os

from properties import PlotProperties
from visualization import plot_variable_vs_time


def load_performance_data(address, sim_name='simulation'):
    """
    This function loads the performance data in the given simulation. If no simulation name is provided, the most
    recent simulation will be loaded.

    Arguments:
        address (string):       -- the disk address where the simulation results are saved
        sim_name (string):      -- the name of the simulation

    returns:
        perf_data (list):       -- the loaded performance data in the form of a list of IterationProperties objects.( \
                                    see :py:class:`properties.IterationProperties` for details).
    """
    log = logging.getLogger('PyFrac.load_performace_data')
    log.info("---loading performance data---\n")

    if address is None:
        address = '.' + slash + '_simulation_data_PyFrac'

    if address[-1] != slash:
        address = address + slash

    if re.match('\d+-\d+-\d+__\d+_\d+_\d+', sim_name[-20:]):
        sim_full_name = sim_name
    else:
        simulations = os.listdir(address)
        time_stamps = []
        for i in simulations:
            if re.match(sim_name + '__\d+-\d+-\d+__\d+_\d+_\d+', i):
                time_stamps.append(i[-20:])
        if len(time_stamps) == 0:
            raise ValueError('Simulation not found! The address might be incorrect.')

        Tmst_sorted = sorted(time_stamps)
        sim_full_name = sim_name + '__' + Tmst_sorted[-1]

    filename = address + sim_full_name + slash + 'perf_data.dat'
    try:
        with open(filename, 'rb') as inp:
            perf_data = dill.load(inp)
    except FileNotFoundError:
        raise ValueError("Performance data not found! Check if it's saving is enabled in simulation properties.")

    return perf_data

#-----------------------------------------------------------------------------------------------------------------------

def get_performance_variable(perf_data, iteration, variable):
    """
    This function gets the required variable from the specified iteration.

    Arguments:
        perf_data (list):       -- the loaded performance data in the form of a list of IterationProperties objects.(see
                                   :py:class:`properties.IterationProperties` for details).
        iteration (string):     -- the type of iteration (see :py:class:`properties.IterationProperties` for details).
        variable (string):      -- the name of the variable to be retrieved.

    returns:
        - var_list (list)      -- the loaded variable.
        - time_list (list)     -- the corresponding simulated times at which the variable was collected.
        - N_list (list)        -- the corresponding number of elements in the fracture at the time step at which the \
                                   variable was collected.
    """
    var_list = []
    time_list = []
    N_list = []

    def append_variable(Iteration, variable):
        var_list.append(getattr(Iteration, variable))
        time_list.append(node.time)
        N_list.append(node.NumbOfElts)

    for i_node, node in enumerate(perf_data):
        if iteration == 'time step':
            append_variable(node, variable)
        else:
            for i_TS_attempt, TS_attempt in enumerate(node.attempts_data):
                if iteration == 'time step attempt':
                    append_variable(TS_attempt, variable)
                else:
                    for i_sameFP_inj, sameFP_inj in enumerate(TS_attempt.sameFront_data):
                        if iteration == 'same front':
                            append_variable(sameFP_inj, variable)
                        else:
                            for i_nonLinSolve, nonLinSolve_itr in enumerate(sameFP_inj.nonLinSolve_data):
                                if iteration == 'nonlinear system solve':
                                    append_variable(nonLinSolve_itr, variable)
                                else:
                                    for i_widthConstraint, widthConstraint_Itr in enumerate(
                                                                            nonLinSolve_itr.widthConstraintItr_data):
                                        if iteration == 'width constraint iteration':
                                            append_variable(widthConstraint_Itr, variable)
                                        else:
                                            for i_linearSolve, linearSolve_Itr in enumerate(
                                                                                widthConstraint_Itr.linearSolve_data):
                                                if iteration == 'linear system solve':
                                                    append_variable(linearSolve_Itr, variable)
                                            for i_RKLSolve, RKLSolve_Itr in enumerate(
                                                                                widthConstraint_Itr.RKL_data):
                                                if iteration == 'RKL time step':
                                                    append_variable(RKLSolve_Itr, variable)

                    for i_extFP_inj, extFP_inj in enumerate(TS_attempt.extendedFront_data):
                        if iteration == 'extended front':
                            append_variable(extFP_inj, variable)
                        else:
                            for i_tipInv_itr, tipInv_itr in enumerate(extFP_inj.tipInv_data):
                                if iteration == 'tip inversion':
                                    append_variable(tipInv_itr, variable)
                                else:
                                    for i_brentq_itr, brentq_itr in enumerate(tipInv_itr.brentMethod_data):
                                        if iteration == 'Brent method':
                                            append_variable(brentq_itr, variable)

                            for i_tipWidth_itr, tipWidth_itr in enumerate(extFP_inj.tipWidth_data):
                                if iteration == 'tip width':
                                    append_variable(tipWidth_itr, variable)
                                else:
                                    for i_brentq_itr, brentq_itr in enumerate(tipWidth_itr.brentMethod_data):
                                        if iteration == 'Brent method':
                                            append_variable(brentq_itr, variable)

                            for i_nonLinSolve, nonLinSolve_itr in enumerate(extFP_inj.nonLinSolve_data):
                                if iteration == 'nonlinear system solve':
                                    append_variable(nonLinSolve_itr, variable)
                                else:
                                    for i_widthConstraint, widthConstraint_Itr in enumerate(
                                                                            nonLinSolve_itr.widthConstraintItr_data):
                                        if iteration == 'width constraint iteration':
                                            append_variable(widthConstraint_Itr, variable)
                                        else:
                                            for i_linearSolve, linearSolve_Itr in enumerate(
                                                                                widthConstraint_Itr.linearSolve_data):
                                                if iteration == 'linear system solve':
                                                    append_variable(linearSolve_Itr, variable)
                                            for i_RKLSolve, RKLSolve_Itr in enumerate(
                                                                                widthConstraint_Itr.RKL_data):
                                                if iteration == 'RKL time step':
                                                    append_variable(RKLSolve_Itr, variable)

    return var_list, time_list, N_list


#-----------------------------------------------------------------------------------------------------------------------

def plot_performance(address, variable, sim_name='simulation', fig=None, plot_prop=None, plot_vs_N=False):
    """
    This function plot the performance variable from the given simulation.

    Arguments:
         address (string):              -- the disk location where the results of the simulation were saved.
         variable (string):             -- Currently, the following variables are supported:

            ===============================     ================================================
            variable                                                meaning
            ===============================     ================================================
            'time step attempts'                the number of attempts taken for the time step
            'fracture front iterations'         fracture front iterations (including the fixed front iteration)
            'tip inversion iterations'          the iterations taken by the brentq method to converge while inverting \
                                                the tip asymptote
            'width constraint iterations'       the iterations taken to converge on closed cells
            'Picard iterations'                 the number of times the linear system is solved
            'CPU time: time steps'              the CPU time taken by each of the time steps
            'CPU time: time step attempts'      the CPU time taken by each of the time step attempt
            ===============================     ================================================
         sim_name(string):              -- the name of the simulation.
         fig (Figure):                  -- a figure to superimpose on
         plot_prop (PlotProperties):    -- a PlotProperties object
         plot_vs_N (bool):              -- if True, a plot of the variable versus the number of cells will also be
                                           plotted.
    """

    perf_data = load_performance_data(address, sim_name)

    if variable in ['time step attempts']:
        var_list, time_list, N_list = get_performance_variable(perf_data, 'time step', 'iterations')
    elif variable in ['fracture front iterations']:
        var_list, time_list, N_list = get_performance_variable(perf_data, 'time step attempt', 'iterations')
    elif variable in ['tip inversion iterations']:
        var_list, time_list, N_list = get_performance_variable(perf_data, 'Brent method', 'iterations')
    elif variable in ['width constraint iterations']:
        var_list, time_list, N_list = get_performance_variable(perf_data, 'nonlinear system solve', 'iterations')
    elif variable in ['Picard iterations']:
        var_list, time_list, N_list = get_performance_variable(perf_data, 'width constraint iteration', 'iterations')
    elif variable in ['RKL substeps']:
        var_list, time_list, N_list = get_performance_variable(perf_data, 'RKL time step', 'iterations')
    elif variable in ['CPU time: time steps']:
        t_start_list, time_list, N_list = get_performance_variable(perf_data, 'time step', 'CpuTime_start')
        del time_list, N_list
        t_end_list, time_list, N_list = get_performance_variable(perf_data, 'time step', 'CpuTime_end')
        var_list = [i - j for i, j in zip(t_end_list, t_start_list)]
    elif variable in ['CPU time: time step attempts']:
        t_start_list, time_list, N_list = get_performance_variable(perf_data, 'time step attempt', 'CpuTime_start')
        del time_list, N_list
        t_end_list, time_list, N_list = get_performance_variable(perf_data, 'time step attempt', 'CpuTime_end')
        var_list = [i - j for i, j in zip(t_end_list, t_start_list)]
    else:
        raise ValueError("Cannot recognize the required variable.")

    var_list_np = np.asarray(var_list)
    del var_list
    time_list_np = np.asarray(time_list)
    del time_list
    N_list_np = np.asarray(N_list)
    del N_list

    if plot_prop is None:
        plot_prop = PlotProperties(line_style='.')

    if plot_vs_N:
        fig = plot_variable_vs_time(N_list_np,
                                     var_list_np,
                                     fig=fig,
                                     plot_prop=plot_prop,
                                     label='fracture front iterations')

        ax = fig.get_axes()[0]
        ax.set_ylabel(variable)
        ax.set_xlabel('number of elements')
    else:
        fig = plot_variable_vs_time(time_list_np,
                                             var_list_np,
                                             fig=fig,
                                             plot_prop=plot_prop,
                                             label=variable)

        ax = fig.get_axes()[0]
        ax.set_ylabel(variable)
        ax.set_xlabel('time')

    return fig
#-----------------------------------------------------------------------------------------------------------------------

def print_performance_data(address, sim_name=None):
    """
    This function generate a file with details of all the iterations and the data collected regarding their preformance

    Arguments:
        address (string):              -- the disk location where the results of the simulation were saved.
        sim_name(string):              -- the name of the simulation.
    """
    log = logging.getLogger('PyFrac.print_performance_data')
    perf_data = load_performance_data(address, sim_name)

    log.info("---saving iterations data---\n")

    f = open('performance_data.txt', 'w+')

    def print_non_linear_system_performance(iteration_prop, tabs):
        f.write(tabs + "--->Non linear system solve" + '\n')
        f.write(tabs + "\tnumber of width constraint iterations to solve non linear system = " + repr(iteration_prop.iterations) + '\n')
        f.write(tabs + "\tCPU time taken: " + repr(iteration_prop.CpuTime_end - iteration_prop.CpuTime_start) + " seconds" + '\n')
        f.write(tabs + "\tnorm for the iteration = " + repr(iteration_prop.norm) + '\n')

        for i_widthConstraint, widthConstraint_Itr in enumerate(iteration_prop.widthConstraintItr_data):
            f.write(tabs + "\t--->width constraint iteration " + repr(i_widthConstraint + 1) + '\n')
            f.write(tabs + "\t\tnumber of linear system solved for the Picard iteration = " + repr(widthConstraint_Itr.iterations) + '\n')
            f.write(tabs + "\t\tCPU time taken: " + repr(widthConstraint_Itr.CpuTime_end - widthConstraint_Itr.CpuTime_start) + " seconds" + '\n')
            f.write(tabs + "\t\tnorm for the iteration = " + repr(widthConstraint_Itr.norm) + '\n')

            for i_linearSolve, linearSolve_Itr in enumerate(widthConstraint_Itr.linearSolve_data):
                f.write(tabs + "\t\t--->linear system solve: iteration no " + repr(i_linearSolve + 1) + '\n')
                f.write(tabs + "\t\t\tsub-iteration data not collected" + '\n')
                f.write(tabs + "\t\t\tCPU time taken: " + repr(linearSolve_Itr.CpuTime_end - linearSolve_Itr.CpuTime_start) + " seconds" + '\n')
                f.write(tabs + "\t\t\tnorm of the iteration = " + repr(linearSolve_Itr.norm) + '\n')

    for i_node, node in enumerate(perf_data):
        status = 'successful' if node.status else 'unsuccessful'
        f.write("time step status = " + status + '\n')
        f.write("number of attempts = " + repr(node.iterations) + '\n')

        for i_TS_attempt, TS_attempt in enumerate(node.attempts_data):
            f.write("--->attempt number: " + repr(i_TS_attempt + 1) + '\n')
            f.write("\tattempt to advance to: " + repr(TS_attempt.time) + " seconds" + '\n')
            f.write("\tCPU time taken: " + repr(TS_attempt.CpuTime_end - TS_attempt.CpuTime_start) + " seconds" + '\n')
            status = 'successful' if node.status else 'unsuccessful'
            f.write("\tattempt status: " + status + '\n')

            for i_sameFP_inj, sameFP_inj in enumerate(TS_attempt.sameFront_data):
                f.write("\t--->same footprint injection" + '\n')
                f.write("\t\tCPU time taken: " + repr(sameFP_inj.CpuTime_end - sameFP_inj.CpuTime_start) + " seconds" + '\n')

                for i_nonLinSolve, nonLinSolve_itr in enumerate(sameFP_inj.nonLinSolve_data):
                    print_non_linear_system_performance(nonLinSolve_itr, '\t\t')

            for i_extFP_inj, extFP_inj in enumerate(TS_attempt.extendedFront_data):
                f.write("\t--->extended footprint injection" + '\n')
                f.write("\t\tCPU time taken: " + repr(extFP_inj.CpuTime_end - extFP_inj.CpuTime_start) + " seconds" + '\n')

                for i_tipInv_itr, tipInv_itr in enumerate(extFP_inj.tipInv_data):
                    f.write("\t\t--->tip inversion" + '\n')
                    f.write("\t\t\tCPU time taken: " + repr(tipInv_itr.CpuTime_end - tipInv_itr.CpuTime_start) + " seconds" + '\n')
                    f.write("\t\t\tnumber of root findings = " + repr(tipInv_itr.iterations) + '\n')

                    for i_brentq_itr, brentq_itr in enumerate(tipInv_itr.brentMethod_data):
                        f.write("\t\t\t--->root finding through brentq method: " + repr(i_brentq_itr + 1) + '\n')
                        f.write("\t\t\t\tCPU time taken: " + repr(brentq_itr.CpuTime_end - brentq_itr.CpuTime_start) + " seconds" + '\n')
                        f.write("\t\t\t\tnumber of iterations = " + repr(brentq_itr.iterations) + '\n')

                for i_tipWidth_itr, tipWidth_itr in enumerate(extFP_inj.tipWidth_data):
                    f.write("\t\t--->tip width" + '\n')
                    f.write("\t\t\tCPU time taken: " + repr(tipWidth_itr.CpuTime_end - tipWidth_itr.CpuTime_start) + " seconds" + '\n')
                    f.write("\t\t\tnumber of root findings = " + repr(tipWidth_itr.iterations) + '\n')

                    for i_brentq_itr, brentq_itr in enumerate(tipWidth_itr.brentMethod_data):
                        f.write("\t\t--->root finding through brentq method: " + repr(i_brentq_itr + 1) + '\n')
                        f.write("\t\t\t\tCPU time taken: " + repr(brentq_itr.CpuTime_end - brentq_itr.CpuTime_start) + " seconds" + '\n')
                        f.write("\t\t\t\tnumber of iterations = " + repr(brentq_itr.iterations) + '\n')

                for i_nonLinSolve, nonLinSolve_itr in enumerate(extFP_inj.nonLinSolve_data):
                    print_non_linear_system_performance(nonLinSolve_itr, '\t\t')

        f.write("\n")
    f.close()

#-----------------------------------------------------------------------------------------------------------------------

