#
# This file is part of PyFrac.
#
# Created by Brice Lecampion on 03.04.17.
# Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory, 2016-2017.  All rights reserved.
# See the LICENSE.TXT file for more details. 
#

import math
import numpy as np

from src.CartesianMesh import *

class MaterialProperties :
    """
    Class defining  the solid Material properties

    instance variables:

    methods:
    """
    def __init__(self,Eprime,Toughness,Cl,SigmaO,Mesh) :   # add Mesh as input directly here.

        if isinstance(Eprime, np.ndarray):  # check if float or ndarray
            print("Eprime  can not be an array as input ! - homogeneous medium only ")
        else:
            self.Eprime = Eprime

        if isinstance(Toughness, np.ndarray):  # check if float or ndarray
            if Toughness.size ==Mesh.NumberOfElts :
                self.K1c = Toughness
                self.Kprime = (32 / math.pi) ** 0.5 * Toughness
            else:
            # error
                print(' Error in the size of Toughness input ')
                return
        else:
            self.K1c = Toughness * np.ones((Mesh.NumberOfElts,), float)
            self.Kprime = (32 / math.pi) ** 0.5 * Toughness* np.ones((Mesh.NumberOfElts,), float)

        if isinstance(Cl, np.ndarray):  # check if float or ndarray
            if Cl.size == Mesh.NumberOfElts:
                self.Cprime = 2.* Cl
            else:
                print(' Error in the size of Leak-Off coefs input ')
                return
        else:
            self.Cprime = 2. * Cl * np.ones((Mesh.NumberOfElts,), float)

        if isinstance(SigmaO, np.ndarray):  # check if float or ndarray
            if SigmaO.size == Mesh.NumberOfElts:
                self.SigmaO = SigmaO
            else:
                print(' Error in the size of Sigma coefs input ')
                return
        else:
            self.SigmaO =  SigmaO * np.ones((Mesh.NumberOfElts,), float)


#--------------------------------------------------------------------------------------------------------

class FluidProperties :
    """
       Class defining  the fluid  properties

       instance variables:
            viscosity: float
            rheology : string
            muprime : 12 viscosity (// plates viscosity factor)
            density : float
       methods:
       """

    # todo: checks input
    def __init__(self,viscosity,density=1000.,rheology = "Newtonian", turbulence=False):

        self.viscosity = viscosity
        self.rheology = rheology
        self.muprime = 12. * viscosity   # the geometric viscosity in the parallel plate solution
        self.density = density
        self.turbulence = turbulence


#--------------------------------------------------------------------------------------------------------
class InjectionProperties:
    """
        Class defining the injection schedule

        instance variables:
            rate: float (could be extended to an array of time, rate changes - not implemented yet)
            source_coordinates: np array (float)
    """
# TODO: checks input, generalize to a an array of source, to the case of time varying rate...
    def __init__(self,rate,source_coordinates,Mesh):  # add Mesh as input directly here to ensure consistency
         self.injectionrate = rate
         self.source_coordinates = source_coordinates
         self.source_location = Mesh.locate_element(source_coordinates[0],source_coordinates[1])


#--------------------------------------------------------------------------------------------------------
class SimulationParameters:
    """
        Class defining the simulation parameters

        instance variables
            MaximumNumberOfTimeSteps : integer (default 10)
            ToleranceFractureFront : float (default 1.e-3)
            ToleranceEHL : float (default 1.e-5)
            maximum_steps: int (default 10)
            cfl_factor  : factor for time-step adaptivity (default 0.8)
            final_time: float (default 1000.)
            tip_asymptote (string):    propagation regime. Possible options:
                        regime -- A  gives the area (fill fraction)
                        regime -- K  gives tip volume according to the square root assymptote
                        regime -- M  gives tip volume according to the viscocity dominated assymptote
                        regime -- Lk is used to calculate the leak off given the distance of the front l (note, its not tip volume)
                        regime -- Mt gives tip volume according to the viscocity, Leak-off assymptote
                        regime -- U  gives tip volume according to the Universal assymptote (Donstov and Pierce, 2017)
                        regime -- MK gives tip volume according to the M-K transition assymptote

    """
    # todo : checks input, ensure variable name consistency...
    def __init__(self, toleranceFractureFront=1.0e-3, toleranceEHL=1.0e-5, maxfront_its = 20, cfl_factor=0.8,
                 tip_asymptote='U',final_time=1000.,maximum_steps=10):
        self.MaximumNumberOfTimeSteps = maximum_steps
        self.ToleranceFractureFront =toleranceFractureFront
        self.ToleranceEHL = toleranceEHL
        self.CFL_factor =cfl_factor
        self.FinalTime = final_time
        self.tip_asymptote=tip_asymptote
        self.MaximumFrontIts = maxfront_its


#--------------------------------------------------------------------------------------------------------

