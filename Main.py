# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on Fri Dec 23 17:49:21 2016.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2017. All rights reserved.
See the LICENSE.TXT file for more details.
"""

# imports
import numpy as np
import math
import copy

import sys
if not './src' in sys.path:
    sys.path.append('./src')


from src.CartesianMesh import *
from src.Fracture import *
from src.LevelSet import *
from src.VolIntegral import *
from src.Elasticity import *
from src.Properties import *
from src.FractureFrontLoop import *

Mesh  = CartesianMesh(10,10,25,25)

Fluid=FluidProperties(1.1e-3)

nu      = 0.4
Eprime  = 3.3e10/(1-nu**2)
K_Ic    = 0.005e6
sigma0  = 0*1e6
Solid =MaterialProperties(Eprime,K_Ic,0.,sigma0,Mesh)

Q0 = 0.027 # injection rate
well_location = np.array([0.,0.])

Injection = InjectionProperties(Q0,well_location,Mesh)

# initial radius of fracture
initRad = 5

# tol_FrntPos = 0.1e-5
# tol_Picard  = 1e-5
# Tend        = 1000

simul_p= SimulationParameters()

# prntcount   = 0
# timeout     = 0
# timeinc     = 5
# first       = 1
# cfl         = 0.6


# create fracture object
Fr      = Fracture(Mesh,Fluid,Solid)
Fr.InitializeRadialFracture(initRad,'radius','M',Solid,Fluid,Injection)


Fr.PlotFracture('complete','footPrint')

#plt.pause(1)
#Fr.InitializePKN(t0,0,h)
#Fr.PlotFracture('complete','footPrint',l_cr,evol=1,Solid)

timeout = Fr.time

# elasticity matrix

C = LoadElastMatrix(Mesh,Solid.Eprime)


MaximumTimeSteps = 4
TimeStep = 0.5
i=0
Tend=10.


while (Fr.time<Tend) and (i<MaximumTimeSteps) :

    i=i+1

    print('\n*********************\ntime = ' + repr(Fr.time))
    Fr_k = copy.deepcopy(Fr)

    status=FractureFrontLoop(Fr_k, C, Solid, Fluid, simul_p, Injection,TimeStep)

    if status != 1:
        print("Fracture front loop not converged")
        break
    else:
        Fr=copy.deepcopy(Fr_k)

    # we need to use functions for the analytical solution not such inline stuff, come on ! what the heck with np.mean(muPrime ) ?
    R_Msol  = 0.6976*Solid.Eprime**(1/9)*(Injection.injectionrate)**(1/3)*Fr.time**(4/9)/(Fluid.muprime)**(1/9)     #Viscoity dominated
    Fr.PlotFracture('complete', 'footPrint', R_Msol)

#    R_Mtsol = (2*sum(Fr.Q)/np.mean(Fr.Cprime))**0.5*Fr.time**0.25/np.pi
#    R_Ksol  = (3/2**0.5/np.pi * Q0*Eprime*Fr.time/Kprime)**0.4 
    
#     if Fr.time>timeout:
#         print(repr(Fr.time//timeout)+' cnt '+repr(prntcount))
#         # l_cr    = (Eprime*Q0**3*Fr.time**4/(4*np.pi**3*(h+2*Mesh.hx)**4*(muPrime/12)))**(1/5)
#         # plt.close("all")
#         Fr.PlotFracture('complete','footPrint',R_Msol)
#         # Fr.Q = 1.1*Fr.Q
# #        Fr.SaveFracture('..\\StressJumpData\\Vertical2\\file'+repr(prntcount))
#         prntcount+=1
#         timeout+=timeinc

    fract   = np.where(Fr.w>1e-10)[0]   # hummm we know which element are in the frac now ?
    print('injected = '+repr(Q0*Fr.time)+' leaked off '+repr(sum(Fr.Leakedoff))+' in Fracture '+repr(Fr.mesh.EltArea*sum(Fr.w[Fr.EltCrack])))
    print('diff = '+repr(1-(sum(Fr.Leakedoff)+Fr.mesh.EltArea*sum(Fr.w))/(Injection.injectionrate*Fr.time)))
    print('Q_inj = ' + repr(Injection.injectionrate))

