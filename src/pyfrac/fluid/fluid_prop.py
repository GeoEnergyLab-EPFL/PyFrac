# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on 03.04.17.
Copyright (c) ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory, 2016-2020.
All rights reserved. See the LICENSE.TXT file for more details.
"""

# external imports
import numpy as np

class FluidProperties:
    """
    Class defining the fluid properties.

    Arguments:
        viscosity (ndarray):     -- viscosity of the fluid .
        density (float):         -- density of the fluid.
        rheology (string):       -- string specifying rheology of the fluid. Possible options:

                                     - "Newtonian"
                                     - "non-Newtonian"
        turbulence (bool):       -- turbulence flag. If true, turbulence will be taken into account
        compressibility (float): -- the compressibility of the fluid.

    Attributes:
        viscosity (ndarray):     -- Viscosity of the fluid (note its different from local viscosity, see
                                    fracture class for local viscosity).
        muPrime (float):         -- 12 * viscosity (parallel plates viscosity factor).
        rheology (string):       -- string specifying rheology of the fluid. Possible options:

                                     - "Newtonian"
                                     - "Herschel-Bulkley" or "HBF"
                                     - "power-law" or "PLF"
        density (float):         -- density of the fluid.
        turbulence (bool):       -- turbulence flag. If true, turbulence will be taken into account.
        compressibility (float): -- the compressibility of the fluid.
        n (float):               -- flow index of the Herschel-Bulkey fluid.
        k (float):               -- consistency index of the Herschel-Bulkey fluid.
        T0 (float):              -- yield stress of the Herschel-Bulkey fluid.
        Mprime                   -- 2**(n + 1) * (2 * n + 1)**n / n**n  * k

    """

    def __init__(self, viscosity=None, density=1000., rheology="Newtonian", turbulence=False, compressibility=0,
                 n=None, k=None, T0=None):
        """
        Constructor function.

        """
        if viscosity is None:
            # uniform viscosity
            self.viscosity = None
            self.muPrime = None
        elif isinstance(viscosity, np.ndarray):  # check if float or ndarray
            raise ValueError('Viscosity of the fluid can not be an array!. Note that'
                             ' its different from local viscosity')
        else:
            self.viscosity = viscosity
            self.muPrime = 12. * self.viscosity  # the geometric viscosity in the parallel plate solution

        rheologyOptions = ["Newtonian", "Herschel-Bulkley", "HBF", "power-law", "PLF"]
        if rheology in rheologyOptions:  # check if rheology match to any rheology option
            self.rheology = rheology
            if rheology in ["Herschel-Bulkley", "HBF"]:
                if n is None or k is None or T0 is None:
                    raise ValueError("n (flow index), k(consistency index) and T0 (yield stress) are required for a \
                                     Herscel-Bulkley type fluid!")
                self.n = n
                self.k = k
                self.T0 = T0
                self.Mprime = 2 ** (n + 1) * (2 * n + 1) ** n / n ** n * k
                self.var1 = self.Mprime ** (-1 / n)
                self.var2 = 1 / n - 1.
                self.var3 = 2. + 1 / n
                self.var4 = 1. + 1 / n
                self.var5 = n / (n + 1.)
            elif rheology in ["power-law", "PLF"]:
                if n is None or k is None:
                    raise ValueError("n (flow index) and k(consistency index) are required for a power-law type fluid!")
                self.n = n
                self.k = k
                self.Mprime = 2 ** (n + 1) * (2 * n + 1) ** n / n ** n * k
        else:  # error
            raise ValueError('Invalid input for fluid rheology. Possible options: ' + repr(rheologyOptions))

        self.density = density

        if isinstance(turbulence, bool):
            self.turbulence = turbulence
        else:
            # error
            raise ValueError('Invalid turbulence flag. Can be either True or False')

        self.compressibility = compressibility

# ----------------------------------------------------------------------------------------------------------------------

