# local

import numpy as np
from scipy.interpolate import griddata
import dill
import os
import re
import sys
import csv
from postprocess_fracture import *
#import pandas as pd

from utility import ReadFracture
from HF_analytical import HF_analytical_sol, get_fracture_dimensions_analytical
from labels import *
# import FractureInitialization


if 'win32' in sys.platform or 'win64' in sys.platform:
    slash = '\\'
else:
    slash = '/'

#-----------------------------------------------------------------------------------------------------------------------


def get_bar_values_and_scales(properties,rho_s):
    """
    This function returns a list of the overbar values defined by GeGa14.

    Args:
        properties (tuple):             -- a tuple containing the Material, FLuid, Injection and Simulation properties
                                           of the simulation
        rho_s (scalar or array):        -- function to calculate the density at each point of the domain

    Returns:
        bar_values (list):              --  A list of two lists:
                                            - list containing the Kbar as a array for every cell and Ebar, mubar as
                                              floats, Deltarho as array and Vtot (none if constant injection) and Qo
                                              (last non zero injection rate for finite release) as floats
                                            - Lbar, b*, w*, p*, v*, t*, V* and Q*


    """

    print('Returning bar values and scales')

    if type(properties) is not tuple:
        raise ValueError('Provided properties incorrect!')


    # we get the bar values for all the parameters
    bar_values = [[np.sqrt(2/np.pi)*properties[0].K1c]] # Kbar index 0
    bar_values[0].append(1 / np.pi * properties[0].Eprime) # Ebar index 1
    bar_values[0].append(np.pi**2 * properties[1].viscosity) # mubar index 2
    bar_values[0].append(float(rho_s-properties[1].density)) #delta rho index 3

    if type(properties[2].injectionRate[0]) is not 'np.ndarray':
        bar_values[0].append(None)
        bar_values[0].append(properties[2].injectionRate)
    else:
        if properties[2].injectionRate[1,-1] == 0:
            delta_t = np.diff(properties[2].injectionRate[0])
            bar_values[0].append(np.sum(np.multiply(properties[2].injectionRate[1][:-1],delta_t)))
            bar_values[0].append(properties[2].injectionRate[1,-2])
        else:
            raise ValueError('Not able to resolve injected volume or rate!')

    # finite volume index 4 and injection rate index 5

    # evaluate scales

    alpha = 0.249 # from GeGa14
    bar_values.append([(bar_values[0][0]/(bar_values[0][3]*9.81))**(2/3)]) # Lb index 0
    bar_values[1].append(alpha**(2/3)*bar_values[1][0]) # bstar index 1
    bar_values[1].append(bar_values[0][0]*np.sqrt(bar_values[1][1])/bar_values[0][1]) # wstar index 2
    bar_values[1].append(bar_values[0][0]/np.sqrt(bar_values[1][1])) # pstar index 3
    bar_values[1].append((bar_values[0][0] / bar_values[0][1]) ** 2 * bar_values[1][3] / bar_values[0][2]) # vstar index 4
    bar_values[1].append(bar_values[1][1] / bar_values[1][4]) # tstar index 5
    bar_values[1].append(bar_values[1][2] * bar_values[1][1] ** 2) # vstar index 6
    bar_values[1].append(bar_values[1][6] / bar_values[1][5]) # Qstar index 7

    return bar_values

#-----------------------------------------------------------------------------------------------------------------------


def load_shutin_numerical_values(properties,fracture_list):
    """
    This function returns a list of the overbar values defined by GeGa14.

    Args:
        properties (tuple):             -- a tuple containing the Material, FLuid, Injection and Simulation properties
                                           of the simulation
        fracture_list (list):           -- list of fractures

    Returns:
        num_results (list):             --  A list containing:
                                            - the simulation name (string)
                                            - the theoretical radius of arrest (float)
                                            - the radius at shut-in (float)
                                            - the numerical radius of arrest (float)
                                            - the exponential alpha of the approximation
                                            - the pre-factor beta of the approximation, R(t>ts) = beta * t^alpha scaled
                                              scaled by the time indpendent part of L_vo (such that beta * LmVo = R(t)
                                              for t > ts)
                                            - the dimensionless toughness at shut-in
                                            - the theoretical time of arrest

    """
    # if os.path.isfile(properties[3]._SimulationProperties__outputAddress+'/num_data.csv'):
    #     df = pd.read_csv(properties[3]._SimulationProperties__outputAddress+'/num_data.csv',
    #                            header=None,squeeze=True)
    #     num_results_all = df.values.tolist()

    num_results = [None] * 8

    # calculate some constants
    gamma_o = 0.6955 # from SaDe 02
    gamma_s = (3/(8*np.sqrt(np.pi)))**(2/5)

    # calculate the values
    num_results[0] = properties[3]._SimulationProperties__simName #simulation name

    # get the theoretical values
    theoretical_values = get_shutin_parameters(properties)
    num_results[1] = theoretical_values[0] # Ra
    num_results[2] = theoretical_values[1] # Rs
    num_results[6] = theoretical_values[2] # Ks
    num_results[7] = theoretical_values[3] # ta

    # calculate here the interpolation
    time_srs = get_fracture_variable(fracture_list,  # list of times
                                     variable='time')
    d_mean = get_fracture_variable(fracture_list, 'd_mean')

    ind_shut = (np.abs(time_srs - properties[2].injectionRate[0,1])).argmin()
    ind_1p = np.argmax(np.asarray(d_mean) > num_results[1] * 0.95)

    num_results[4:6] = np.polyfit(np.log10(time_srs[ind_shut:ind_1p]), np.log10(d_mean[ind_shut:ind_1p]), 1)
    Lm_Vo = (properties[0].Eprime * (properties[2].injectionRate[0,1] * properties[2].injectionRate[1,0]) ** 3 /
             properties[1].muPrime) ** (1/9)
    num_results[5] = 10**num_results[5]/Lm_Vo

    intercepts = get_front_intercepts(fracture_list, [0,0])
    num_results[3] = (intercepts[-1][3] - intercepts[-1][2])/2

    # if 'num_results_all' in locals():
    #     if any(num_results[0] in sl for sl in num_results_all):
    #         ind_csv = next(i for i,v in enumerate(num_results_all) if num_results[0] in v)
    #         num_results_all[ind_csv] = num_results
    #     else:
    #         num_results_all.append(num_results)
    #
    #     with open(properties[3]._SimulationProperties__outputAddress + '/num_data.csv', "w", newline="") as f:
    #         writer = csv.writer(f)
    #         for item in num_results_all:
    #             writer.writerow(item)
    #
    # else:
    #     with open(properties[3]._SimulationProperties__outputAddress+'/num_data.csv', "w", newline="") as f:
    #         writer = csv.writer(f)
    #         writer.writerow(num_results)

    return num_results

#-----------------------------------------------------------------------------------------------------------------------


def get_shutin_parameters(properties):
    """
    This function returns a list of the overbar values defined by GeGa14.

    Args:
        properties (tuple):             -- a tuple containing the Material, FLuid, Injection and Simulation properties
                                           of the simulation

    Returns:
        shut_in_parameters (list):      --  A list containing:
                                            - the theoretical radius of arrest (float)
                                            - the radius at shut-in (float)
                                            - the dimensionless toughness at shut-in
                                            - the theoretical time of arrest

    """
    shut_in_parameters = [None] * 4
    # calculate some constants
    gamma_o = 0.6955 # from SaDe 02
    gamma_s = (3/(8*np.sqrt(np.pi)))**(2/5)

    # calculate the values
    shut_in_parameters[0] = gamma_s*(properties[0].Eprime * properties[2].injectionRate[1,0] *
                              properties[2].injectionRate[0,1] / properties[0].K1c[0]) ** (2/5) # Ra
    shut_in_parameters[1] = gamma_o*(properties[0].Eprime * properties[2].injectionRate[1,0] ** 3 *
                              properties[2].injectionRate[0,1] ** 4 / properties[1].muPrime) ** (1/9) # Rs
    shut_in_parameters[2] = properties[0].Kprime[0] * (properties[2].injectionRate[0,1] ** 2 /
                                                (properties[1].muPrime ** 5 * properties[0].Eprime ** 13 *
                                                 properties[2].injectionRate[1,0] ** 3)) ** (1/18) # Ks
    shut_in_parameters[3] = properties[2].injectionRate[0,1] * shut_in_parameters[2] ** (-18/5)# ta

    return shut_in_parameters