# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on Fri June 16 17:49:21 2017.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2017. All rights reserved.
See the LICENSE.TXT file for more details.
"""

# imports
from src.Fracture import *
from src.Controller import *
from src.PostProcess import *
import numpy as np
# creating mesh
Mesh = CartesianMesh(10, 10, 51, 51)

# solid properties
nu = 0.25                            # Poisson's ratio
youngs_mod = 1e8               # Young's modulus
Eprime = youngs_mod / (1 - nu ** 2) # plain strain modulus
K_Ic = 1e4                          # fracture toughness of the material
Kprime = 4*(2/np.pi)**(1/2)*K_Ic
g = 9.81
def sigmaO_func(x, y):
    #The function providing the confining stress
    density = 400
    sigma0_ini = 29430000
    return sigma0_ini - density * g * y

Solid = MaterialProperties(Mesh,
                           Eprime,
                           K_Ic,
                           SigmaO_func=sigmaO_func)

# injection parameters
Q0 = 1  # injection rate
Injection = InjectionProperties(np.array([[0], [Q0]]), Mesh)

# fluid properties
Fluid = FluidProperties(viscosity=50, density=400)

# simulation properties
simulProp = SimulationParameters()
simulProp.FinalTime = 100000          # the time at which the simulation stops
simulProp.outputTimePeriod = 1e-10    # the time after which the next fracture file is saved
simulProp.bckColor = 'sigma0'   # the parameter according to which the mesh is color coded
simulProp.set_tipAsymptote('M')
simulProp.plotFigure = False
simulProp.gravity = False
simulProp.set_outFileAddress(".\\Data\\WithBuoyancyConstantStress")
simulProp.verbosity = 2
simulProp.plotFigure = True
simulProp.analyticalSol = 'M'
simulProp.plotAnalytical = True
#simulProp.tmStpFactLimit = 1.5

# initializing fracture
initTime = 10
init_param = ('M', "time", initTime)


"""# Calculating dimentionless numbers
RhoRock = 3200
DeltaRho = RhoRock - Fluid.density
t = 1000

Km = Kprime*((t**2)/(Eprime**13*Q0**3*Fluid.muPrime))**(1/18)
Bm = DeltaRho*9.81*(t/(Eprime**2*Fluid.muPrime))**(1/3)
print('Km = ',Km, '\n', 'Bm = ', Bm)

Mk = Fluid.muPrime*((Eprime**13*Q0**3)/(Kprime**18*t**2))**(1/5)
Bk = DeltaRho*9.81/Fluid.muPrime * ((Kprime**12*t**3)/(Eprime**12*Q0**2))**(1/5)
print('Mk = ', Mk, '\n', 'Bk = ', Bk)

Kb = Kprime/Eprime * ((t**3*(DeltaRho*9.81)**5)/(Q0**2*Fluid.muPrime**5))**(1/12)
Mb = (DeltaRho*9.81)**(3/2)/(Eprime*Fluid.muPrime**(1/2))*t**(1/2)
Tmk = Eprime**(13/2)*Q0**(3/2)*Fluid.muPrime**(5/2)/(Kprime**9)
Tkb = t/Kb**4
Tmb = t/Mb**2
l = (Kprime**3/(Fluid.muPrime*Eprime**2*1))**2
gT = DeltaRho*g*l**(3/2)/Kprime
KP = 1/2*gT**(1/8)
print('Kb = ',Kb, '\n', 'Mb = ', Mb, '\n', 'Tkb = ', Tkb, '\n', 'Tmb = ', Tmb, 'Tmk = ', Tmk, 'KP = ', KP)

"""
# creating fracture object
Fr = Fracture(Mesh,
              init_param,
              Solid,
              Fluid,
              Injection,
              simulProp)


# create a Controller
controller = Controller(Fr,
                        Solid,
                        Fluid,
                        Injection,
                        simulProp)

# run the simulation
controller.run()

# plot results

plot_at = np.linspace(1, 1000)
plot_footprint(simulProp.get_outFileAddress(), time_period=0.1,
                    plot_at_times = plot_at)
#plot_footprint(simulProp.get_outFileAddress(), fig=figure1, time_period=0.1,
 #                   plot_at_times = plot_at, analytical_sol='K', plt_mesh=False, anltcl_lnStyle='r')
#plot_leakOff(simulProp.get_outFileAddress(),loglog=False, plot_at_times=10**np.linspace(-9, 2, 10))
plt.show()
