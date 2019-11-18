# local

import numpy as np
from scipy.interpolate import griddata
import dill
import os
import re
import sys

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

    print('Returning Kbar values...')

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

