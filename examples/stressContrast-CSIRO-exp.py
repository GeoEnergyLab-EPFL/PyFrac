# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on Fri Dec 23 17:49:21 2016.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2017. All rights reserved.
See the LICENSE.TXT file for more details.
"""

# This example model the stress jump experiment reported in Wu et al. 2008 ARMA 08-267
#
# PMMA interface with CNC machined normal stress
# top layer Sig0=11.2MPa
# middle layer (H=50mm) , injection point in the middle,  Sig0= 7MP
# bottom layer   Sig0=5MPa

# adding src folder to the path
import sys
if "wind" in sys.platform:
    slash = "\\"
else:
    slash = "/"
if not '..' + slash + 'src' in sys.path:
    sys.path.append('.' + slash + 'src')
if not '.' + slash + 'src' in sys.path:
    sys.path.append('.' + slash + 'src')

# imports
import numpy as np
from src.CartesianMesh import *
from src.Fracture import *
from src.LevelSet import *
from src.VolIntegral import *
from src.Elasticity import *
from src.Properties import *
from src.FractureFrontLoop import *

# creating mesh
Mesh = CartesianMesh(.14,.14,66,66)

# solid properties
nu = 0.4
Eprime = 3.3e9 / (1 - nu ** 2)
K_Ic = 0.01e6

sigma0 = np.full((Mesh.NumberOfElts,), 7e6, dtype=np.float64)
# high stressed layers
stressed_layer_1 = np.where(Mesh.CenterCoor[:,1] > 0.025 )[0]
stressed_layer_2 = np.where(Mesh.CenterCoor[:,1] < -(0.025 ))[0]

sigma0[stressed_layer_1] = 11.2e6
sigma0[stressed_layer_2] = 5e6

d_grain = 1e-5
Solid = MaterialProperties(Eprime, K_Ic, 0., sigma0, d_grain, Mesh)

# injection parameters

well_location = np.array([0., 0.])   # todo: ensure initialization can be done for a fracture not at the center of the grid

myQo = np.array([ [0.,31.,151.],[ 0.0009*1.e-6 , 0.0065*1.e-6 ,0.0023*1.e-6 ]])

simulation_name = "LowStressJump_1"
Injection = InjectionProperties(myQo, well_location, Mesh)

# fluid properties
Fluid = FluidProperties(30, Mesh, turbulence=False)

# simulation properties
simulProp = SimulationParameters(tip_asymptote="U",
                                 output_time_period=10,
                                 plot_figure=True,
                                 save_to_disk=True,
                                 out_file_folder="./examples/"+simulation_name, # e.g. "./Data/Laminar" for linux or mac
                                 plot_analytical=False,
                                 tmStp_prefactor=0.4,
                                 analytical_sol="M",plot_evolution=False)


# initializing fracture
#initRad = 0.01 # initial radius of fracture

initTime = 20.;

# creating fracture object
Fr = Fracture(Mesh,
              initTime,
              'time',
              'M',
              Solid,
              Fluid,
              Injection,
              simulProp)


# elasticity matrix
C = load_elasticity_matrix(Mesh, Solid.Eprime)

# starting time stepping loop
i = 0
Fr_k = Fr

simulProp.FinalTime=30

while (Fr.time < simulProp.FinalTime) and (i < simulProp.maxTimeSteps):

    i = i + 1

    TimeStep = simulProp.tmStpPrefactor * min(Fr.mesh.hx, Fr.mesh.hy) / np.max(Fr.v)
    status, Fr_k = advance_time_step(Fr_k, C, Solid, Fluid, simulProp, Injection, TimeStep)

    Fr = copy.deepcopy(Fr_k)


#### post processing

# read fract from file


fileNo = 0
maxFiles=20

fraclist = [];

while fileNo < maxFiles:

    # trying to load next file. exit loop if not found
    try:
        ff = ReadFracture( './examples/Data/' + "file_" + repr(fileNo))
    except FileNotFoundError:
        break
    fileNo+=1
    fraclist.append(ff)


    g=ff.plot_fracture('complete', 'footPrint', mat_Properties=Solid)



ff=ReadFracture('./examples/Data/file_0')

ff.plot_fracture('complete','footPrint',mat_Properties=Solid)

ff.plot_fracture('complete','width')

ff.plot_fracture('complete','pressure')  #net pressure




###
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig, ax = plt.subplots()
line, = ax.plot(np.random.rand(10))
ax.set_ylim(0, 1)


def update(data):
    line.set_ydata(data)
    return line,


def data_gen():
    while True:
        yield np.random.rand(10)

ani = animation.FuncAnimation(fig, update, data_gen, interval=100)
plt.show()
