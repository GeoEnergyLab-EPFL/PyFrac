#
# This file is part of PyFrac.
#
# Created by Haseeb Zia on 04.03.18.
# Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory, 2016-2017.  All rights
# reserved. See the LICENSE.TXT file for more details.
#

errorMessages = ("Propagation not attempted!",
                     "Time step successful!",
                     "Evaluated level set is not valid!",
                     "Front is not tracked correctly!",
                     "Evaluated tip volume is not valid!",
                     "Solution obtained from the elastohydrodynamic solver is not valid!",
                     "Did not converge after max iterations!",
                     "Tip inversion is not correct!",
                     "Ribbon element not found in the enclosure of the tip cell!",
                     "Filling fraction not correct!",
                     "Toughness iteration did not converge!",
                     "projection could not be found!",
                     "Reached end of grid!",
                     "Leak off can't be evaluated!"
                     )

import sys
if "win32" in sys.platform or "win64" in sys.platform:
    slash = "\\"
else:
    slash = "/"
import matplotlib.pyplot as plt
import dill
from src.PostProcess import to_precision
import numpy as np

def plot_fracture_front_iterations(address=None, fig_itr=None, fig_norm=None, plt_lnStyle='.', plt_norm=True,
                                   loglog=True, alpha=1):

    print("---Plotting fracture front iterations---\n\n")

    if address is None:
        address = "." + slash + "_simulation_data_PyFrac"

    if not slash in address[-2:]:
        address = address + slash


    try:
        with open(address + "perf_data.dat", 'rb') as input:
            perf_data = dill.load(input)
    except FileNotFoundError:
        raise SystemExit("Data not found. The address might be incorrect")

    frac_front_itrs = []
    frac_time = []
    if plt_norm:
        norms_list = []
    for i in perf_data:

        tm_Stp_attmpt = i.subIterations[-1]
        if tm_Stp_attmpt.status == 'successful':
            frac_front_itrs.append(tm_Stp_attmpt.iterations)
            frac_time.append(tm_Stp_attmpt.time)
            if plt_norm:
                norms_list.append(tm_Stp_attmpt.normList[2:])

    print("total number of iterations = " + repr(sum(frac_front_itrs)) + "; average = "
          + repr(sum(frac_front_itrs) / len(frac_front_itrs)))

    if fig_itr is None:
        fig_itr = plt.figure()
        ax_itr = fig_itr.add_subplot(111)
    else:
        ax_itr = fig_itr.get_axes()[0]

    if loglog:
        ax_itr.semilogx(frac_time, frac_front_itrs, plt_lnStyle, alpha=alpha)
    else:
        ax_itr.plot(frac_time, frac_front_itrs, plt_lnStyle, alpha=alpha)
    ax_itr.set_ylabel('number of fracture front iterations')
    ax_itr.set_xlabel('time step number')
    ax_itr.set_title('Fracture front iterations for each time step')

    if plt_norm:
        if fig_norm is None:
            fig_norm = plt.figure()
            ax_norm = fig_norm.add_subplot(111)
        else:
            ax_norm = fig_norm.get_axes()[0]

        for i in norms_list:
            ax_norm.plot(i, 'o-')

    return fig_itr, fig_norm

#-----------------------------------------------------------------------------------------------------------------------


def print_performance_data(address=None):

    print("---saving iterations data---\n\n")

    if address is None:
        address = "." + slash + "_simulation_data_PyFrac"

    if not slash in address[-2:]:
        address = address + slash


    try:
        with open(address + "perf_data.dat", 'rb') as input:
            perf_data = dill.load(input)
    except FileNotFoundError:
        raise SystemExit("Data not found. The address might be incorrect")

    f = open('performance_data.txt', 'w+')

    for i in perf_data:
        f.write("time step status = " + i.status + '\n')
        f.write("number of attempts = " + repr(i.iterations) + '\n')
        for attmpt_no in range(len(i.subIterations)):
            tmStp_attmpt = i.subIterations[attmpt_no]
            f.write("--->attempt number: " + repr(attmpt_no + 1) + '\n')
            f.write("\tattempt to advance to: " + repr(tmStp_attmpt.time) + " seconds" + '\n')
            f.write("\tCPU time taken: " + repr(tmStp_attmpt.CpuTime_end - tmStp_attmpt.CpuTime_start)
                    + " seconds" + '\n')
            f.write("\tattempt status: " + tmStp_attmpt.status + '\n')
            f.write("\tnumber of sub-iterations: " + repr(tmStp_attmpt.iterations) + '\n')
            f.write("\tnorm list: [")
            for norm_entry in tmStp_attmpt.normList:
                if np.isnan(norm_entry):
                    f.write(repr(np.nan) + ' ')
                else:
                    f.write(to_precision(norm_entry, 4) + ' ')
            f.write("]\n")
            if tmStp_attmpt.status == 'failed':
                f.write("\tCause of failure = " + errorMessages[tmStp_attmpt.failure_cause] + '\n')

            if len(tmStp_attmpt.subIterations[0]) == 1:
                expl_frnt_itrs = tmStp_attmpt.subIterations[0]
                f.write("\t--->sub-iteration: 1" + '\n')
                f.write("\t\tsub-iteration type: " + expl_frnt_itrs[0].itrType + '\n')
                f.write("\t\tnumber of iterations = 1" + '\n')
                f.write("\t\tCPU time taken: " + repr(expl_frnt_itrs[0].CpuTime_end - expl_frnt_itrs[0].CpuTime_start)
                        + " seconds" + '\n')

                picard_itrs = expl_frnt_itrs[0].subIterations[2]
                if len(picard_itrs) > 0:
                    f.write("\t\t--->sub-iteration: 1" + '\n')
                    f.write("\t\t\tsub-iteration type: " + picard_itrs[0].itrType + '\n')
                    f.write("\t\t\tno. of iterations: " + repr(picard_itrs[0].iterations) + '\n')
                    f.write("\t\t\tCPU time taken: " + repr(picard_itrs[0].CpuTime_end - picard_itrs[0].CpuTime_start)
                        + " seconds" + '\n')
                    f.write("\t\t\tnorm list: [")
                    for norm_entry in picard_itrs[0].normList:
                        if np.isnan(norm_entry):
                            f.write(repr(np.nan) + ' ')
                        else:
                            f.write(to_precision(norm_entry, 4) + ' ')
                    f.write("]\n")
            if len(tmStp_attmpt.subIterations[1]) == 1:
                sameFP_injs = tmStp_attmpt.subIterations[1]
                f.write("\t--->sub-iteration: 1" + '\n')
                f.write("\t\tsub-iteration type: " + sameFP_injs[0].itrType + '\n')
                f.write("\t\tnumber of iterations = 1" + '\n')
                f.write("\t\tCPU time taken: " + repr(sameFP_injs[0].CpuTime_end - sameFP_injs[0].CpuTime_start)
                        + " seconds" + '\n')

                picard_itrs = sameFP_injs[0].subIterations
                f.write("\t\t--->sub-iteration: 1" + '\n')
                f.write("\t\t\tsub-iteration type: " + picard_itrs[0].itrType + '\n')
                f.write("\t\t\tno. of iterations: " + repr(picard_itrs[0].iterations) + '\n')
                f.write("\t\t\tCPU time taken: " + repr(picard_itrs[0].CpuTime_end - picard_itrs[0].CpuTime_start)
                        + " seconds" + '\n')
                f.write("\t\t\tnorm list: [")
                for norm_entry in picard_itrs[0].normList:
                    if np.isnan(norm_entry):
                        f.write(repr(np.nan) + ' ')
                    else:
                        f.write(to_precision(norm_entry, 4) + ' ')
                f.write("]\n")

            if len(tmStp_attmpt.subIterations[2]) >= 1:
                f.write("\t--->sub-iteration: 2" + '\n')
                f.write("\t\tsub-iteration type = injection with extended footprint" + '\n')
                f.write("\t\tCPU time taken: " + repr(tmStp_attmpt.CpuTime_end - tmStp_attmpt.CpuTime_start)
                        + " seconds" + '\n')
                f.write("\t\tnumber of iterations " + repr(tmStp_attmpt.iterations - 1) + '\n')

                for front_itr_no in range(len(tmStp_attmpt.subIterations[2])):
                    extndFP_inj = tmStp_attmpt.subIterations[2][front_itr_no]
                    picard_itrs = extndFP_inj.subIterations[2]
                    if len(picard_itrs)>0:
                        f.write("\t\t--->sub-iteration: " + repr(front_itr_no + 1) + '\n')
                        f.write("\t\t\tsub-iteration type: " + picard_itrs[0].itrType + '\n')
                        f.write("\t\t\tCPU time taken: " + repr(picard_itrs[0].CpuTime_end - picard_itrs[0].CpuTime_start)
                                + " seconds" + '\n')
                        f.write("\t\t\tno. of iterations: " + repr(picard_itrs[0].iterations) + '\n')
                        f.write("\t\t\tnorm list: [" )
                        for norm_entry in picard_itrs[0].normList:
                            if np.isnan(norm_entry):
                                f.write(repr(np.nan) + ' ')
                            else:
                                f.write(to_precision(norm_entry, 4) + ' ')
                        f.write("]\n")
        f.write("\n")
    f.close()

#-----------------------------------------------------------------------------------------------------------------------


def plot_reattempts(address=None, fig=None, plt_lnStyle='.', alpha=1):
    print("---plotting reattempts---\n\n")

    if address is None:
        address = "." + slash + "_simulation_data_PyFrac"

    if not slash in address[-2:]:
        address = address + slash

    try:
        with open(address + "perf_data.dat", 'rb') as input:
            perf_data = dill.load(input)
    except FileNotFoundError:
        raise SystemExit("Data not found. The address might be incorrect")

    attempts = []
    time = []
    for i in perf_data:
        tm_Stp_attmpt = i.subIterations[-1]
        attempts.append(i.iterations)
        time.append(tm_Stp_attmpt.time)

    if fig is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        ax = fig.get_axes()[0]

    ax.plot(time, attempts, plt_lnStyle, alpha=alpha)
    ax.set_ylabel('number of re-attempts')
    ax.set_xlabel('time step number')
    ax.set_title('Re-attempts for each time step')

    return fig

#-----------------------------------------------------------------------------------------------------------------------


def plot_timeStep_CPU_time(address=None, fig=None, plt_lnStyle='.', loglog=True, alpha=1):
    print("---Plotting CPU time---\n\n")

    if address is None:
        address = "." + slash + "_simulation_data_PyFrac"

    if not slash in address[-2:]:
        address = address + slash

    try:
        with open(address + "perf_data.dat", 'rb') as input:
            perf_data = dill.load(input)
    except FileNotFoundError:
        raise SystemExit("Data not found. The address might be incorrect")

    CPU_time_list = []
    time_list = []
    for i in perf_data:
        successful_attmpt = i.subIterations[-1]
        CPU_time_list.append(successful_attmpt.CpuTime_end - successful_attmpt.CpuTime_start)
        time_list.append(successful_attmpt.time)
    if fig is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        ax = fig.get_axes()[0]

    if loglog:
        ax.semilogx(time_list, CPU_time_list, plt_lnStyle, alpha=alpha)
    else:
        ax.plot(time_list, CPU_time_list, plt_lnStyle, alpha=alpha)
    ax.set_ylabel('CPU time (seconds)')
    ax.set_xlabel('time step number')
    ax.set_title('CPU time for each time step')

    print("Total CPU time = " + repr(sum(CPU_time_list)))
    return fig

#-----------------------------------------------------------------------------------------------------------------------


def plot_picard_iterations_number(address=None, fig=None, plt_lnStyle='.', loglog=True, alpha=1):

    print("---Ploting picard iterations ---\n\n")

    if address is None:
        address = "." + slash + "_simulation_data_PyFrac"

    if not slash in address[-2:]:
        address = address + slash


    try:
        with open(address + "perf_data.dat", 'rb') as input:
            perf_data = dill.load(input)
    except FileNotFoundError:
        raise SystemExit("Data not found. The address might be incorrect")

    no_itrs = []
    time_list = []
    for i in perf_data:

        for attmpt_no in range(len(i.subIterations)):
            tmStp_attmpt = i.subIterations[attmpt_no]

            if len(tmStp_attmpt.subIterations[0]) == 1:
                expl_frnt_itrs = tmStp_attmpt.subIterations[0]
                picard_itrs = expl_frnt_itrs[0].subIterations[2]
                if len(picard_itrs) > 0:
                    no_itrs.append(picard_itrs[0].iterations)
                    time_list.append(tmStp_attmpt.time)

            if len(tmStp_attmpt.subIterations[1]) == 1:
                sameFP_injs = tmStp_attmpt.subIterations[1]
                picard_itrs = sameFP_injs[0].subIterations
                no_itrs.append(picard_itrs[0].iterations)
                time_list.append(tmStp_attmpt.time)

            if len(tmStp_attmpt.subIterations[2]) >= 1:
                for front_itr_no in range(len(tmStp_attmpt.subIterations[2])):
                    extndFP_inj = tmStp_attmpt.subIterations[2][front_itr_no]
                    picard_itrs = extndFP_inj.subIterations[2]
                    if len(picard_itrs)>0:
                       no_itrs.append(picard_itrs[0].iterations)
                       time_list.append(tmStp_attmpt.time)

    no_itrs_np = np.asarray(no_itrs)
    time_list_np = np.asarray(time_list)
    to_delete = np.where(no_itrs_np >= 48)[0]
    no_itrs_np = np.delete(no_itrs_np, to_delete)
    time_list_np = np.delete(time_list_np, to_delete)
    if fig is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        ax = fig.get_axes()[0]

    if loglog:
        ax.semilogx(time_list_np, no_itrs_np, plt_lnStyle, alpha=alpha)
    else:
        ax.plot(time_list_np, no_itrs_np, plt_lnStyle, alpha=alpha)
    ax.set_ylabel('number of iterations')
    ax.set_xlabel('iteration number')
    ax.set_title('Number of Picard iterations ')

    return fig

#-----------------------------------------------------------------------------------------------------------------------


def plot_time_steps(address=None, fig=None, plt_lnStyle='.', alpha=1):
    print("---plotting time steps---\n\n")

    if address is None:
        address = "." + slash + "_simulation_data_PyFrac"

    if not slash in address[-2:]:
        address = address + slash

    try:
        with open(address + "perf_data.dat", 'rb') as input:
            perf_data = dill.load(input)
    except FileNotFoundError:
        raise SystemExit("Data not found. The address might be incorrect")

    times = []
    for i in perf_data:
        tm_Stp_attmpt = i.subIterations[-1]
        times.append(tm_Stp_attmpt.time)

    if fig is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
    else:
        ax = fig.get_axes()[0]

    ax.plot(times, plt_lnStyle, alpha=alpha)
    ax.set_ylabel('time')
    ax.set_xlabel('time step number')
    ax.set_title('Re-attempts for each time step')

    return fig
