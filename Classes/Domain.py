# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on Thu Dec 22 12:42:24 2016.
Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory, 2016-2017. All rights reserved.
See the LICENSE.TXT file for more details.
"""

import numpy as np

class Domain:
    """
    Class defining a domain being simulated. Is extended with fracture class
    
    instance variables:
    
    methods:
    """
    
    def __init__(self,mesh,Eprime,Kprime,sigma0,Cprime):
        
        self.Eprime = Eprime
        self.mesh   = mesh

        if isinstance(Kprime, np.ndarray): #check if float or ndarray
            self.Kprime = Kprime
        else:
            self.Kprime = Kprime*np.ones((mesh.NumberOfElts,),float)
        
        if isinstance(Cprime, np.ndarray): #check if float or ndarray
            self.Cprime = Cprime
        else:
            self.Cprime = Cprime*np.ones((mesh.NumberOfElts,),float)
            
        if isinstance(sigma0, np.ndarray): #check if float or ndarray
            self.sigma0 = sigma0
        else:
            self.sigma0 = sigma0*np.ones((mesh.NumberOfElts,),float)
               