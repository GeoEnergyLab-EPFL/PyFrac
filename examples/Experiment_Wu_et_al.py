# -*- coding: utf-8 -*-
"""
This file is part of PyFrac.

Created by Haseeb Zia on Fri March 13 2018.
Copyright (c) "ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE, Switzerland, Geo-Energy Laboratory", 2016-2017. All rights reserved.
See the LICENSE.TXT file for more details.
"""

# imports
from src.Fracture import *
from src.Controller import *
from src.PostProcess import *


# creating mesh
Mesh = CartesianMesh(0.13, 0.17, 51, 67)

# solid properties
nu = 0.4                            # Poisson's ratio
youngs_mod = 3.3e9                  # Young's modulus
Eprime = youngs_mod / (1 - nu ** 2) # plain strain modulus
K_Ic = 10000                        # set toughness to a very low value

def sigmaO_func(x, y):
    """ The function providing the confining stress"""
    if y > 0.025:
        return 11.2e6
    elif y < -0.025:
        return 5.0e6
    else:
        return 7.0e6

Solid = MaterialProperties(Mesh,
                           Eprime,
                           K_Ic,
                           SigmaO_func=sigmaO_func)

# injection parameters
Q0 = np.asarray([[0, 31, 151], [0.0009e-6, 0.0065e-6, 0.0023e-6]])
Injection = InjectionProperties(Q0, Mesh)

# fluid properties
Fluid = FluidProperties(viscosity=30)

# simulation properties
simulProp = SimulationParameters()
simulProp.outputTimePeriod = 0.1        # Setting it small so the file is saved after every time step
simulProp.bckColor = 'sigma0'           # the parameter according to which the background is color coded
simulProp.tmStpPrefactor = 0.5          # set the time step prefactor
# simulProp.set_outFileAddress('.\\Data\\Wu_et_al')
simulProp.set_solTimeSeries(np.asarray([22., 60., 144., 376., 665.]))

# initializing fracture
initRad = 0.014
init_param = ('M', "length", initRad)

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

# loading the experiment data file
import csv
with open('./wu_et_al_data.csv', 'r') as f:
    reader = csv.reader(f)
    data = np.asarray(list(reader), dtype=np.float64)

# plotting fracture footprint
Fig = plot_footprint(simulProp.get_outFileAddress(),
                     plot_at_times=simulProp.get_solTimeSeries(),
                     plt_color='b')

# plotting footprint from experiment
ax = Fig.get_axes()[0]
ax.plot(data[:, 0]*1e-3, -1e-3*data[:, 1], 'k')
ax.plot(data[:, 2]*1e-3, -1e-3*data[:, 3], 'k')
ax.plot(data[:, 4]*1e-3, -1e-3*data[:, 5], 'k')
ax.plot(data[:, 6]*1e-3, -1e-3*data[:, 7], 'k')
ax.plot(data[:, 8]*1e-3, -1e-3*data[:, 9], 'k')

blue_patch = mpatches.mlines.Line2D([], [], color='k', label='experiment (Wu et al. 2008)')
black_patch = mpatches.mlines.Line2D([], [], color='b', label='numerical')
plt.legend(handles=[blue_patch, black_patch])
ax.set_ylim(-170e-3, 50e-3,)

plt.show()

#### The code below will save images to the given folder and then make a video showing the growth of the fracture along
#  with the experimental data. You will need to install openCv to make the video from images.

# # make a movie
# from src.Utility import ReadFracture
# from src.PostProcess import to_precision
# tm_srs_index = 0
# exp_data_index = 8
#
# loop to read fracture data
# for i in range(52):
#     name = simulProp.get_outFileAddress() + "fracture_" + repr(i)
#     ff = ReadFracture(name)
#
#     # plotting the current footprint
#     fig = plot_footprint(simulProp.get_outFileAddress(),
#                    plot_at_times=ff.time,
#                    plt_color='k')
#
#     # plotting the traversed footprints
#     fig = plot_footprint(simulProp.get_outFileAddress(),
#                          plot_at_times=simulProp.get_solTimeSeries()[0:tm_srs_index],
#                          fig=fig,
#                          plt_color='k',
#                          plt_mesh=False)
#
#     ax = fig.get_axes()[0]
#     time = to_precision(ff.time, 3)
#     ax.set_title(time + 's.')
#
#     if ff.time == simulProp.get_solTimeSeries()[tm_srs_index]:
#         tm_srs_index += 1
#
#     # plotting traversed experiment data
#     exp_data_index = 8
#     for j in range(tm_srs_index):
#         line1 = ax.plot(data[:, exp_data_index] * 1e-3, -1e-3 * data[:, exp_data_index + 1],
#                         'b', linewidth=0.5, label="Wu et al. 2008")
#         exp_data_index -= 2
#
#     blue_patch = mpatches.mlines.Line2D([], [], color='b', label='experiment (Wu et al. 2008)')
#     black_patch = mpatches.mlines.Line2D([], [], color='k', label='numerical')
#     plt.legend(handles=[blue_patch, black_patch])
#     ax.set_ylim(-170e-3, 50e-3,)
#
#     # saving file
#     print("\nSaving image number: " + str(i).zfill(4))
#     fig.savefig(".\\images\\" + str(i).zfill(4) + '.png', dpi=300)
#     plt.close(fig)
#
# # making a movie from the saved images
# save_images_to_video('.\\images')